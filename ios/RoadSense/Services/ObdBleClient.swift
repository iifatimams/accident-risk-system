import Foundation
import CoreBluetooth

public protocol ObdDelegate: AnyObject {
    func didUpdateObdData(speed: Double?, rpm: Double?, throttle: Double?, load: Double?)
    func didChangeConnectionStatus(isConnected: Bool)
}

public class ObdBleClient: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    public weak var delegate: ObdDelegate?
    public var targetDeviceIdentifier: String?
    
    private var centralManager: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var txCharacteristic: CBCharacteristic?
    private var rxCharacteristic: CBCharacteristic?
    
    private var pollTimer: Timer?
    private var scanTimeoutWorkItem: DispatchWorkItem?
    
    // Strict ELM327 initialization sequence
    private let initCommands = [
        "ATZ\r",     // Reset adapter
        "ATE0\r",    // Echo off
        "ATL0\r",    // Linefeeds off[cite: 1]
        "ATS0\r",    // Spaces off[cite: 1]
        "ATH0\r",    // Headers off[cite: 1]
        "ATSP0\r",   // Auto protocol negotiation[cite: 1]
        "0100\r"     // Test ECU link[cite: 1]
    ]
    private var initStep = 0
    private var handshakeStarted = false
    private var isHandshakeDone = false
    private var isScanning = false
    private var isPollingActive = false
    
    // Round-robin query cycle[cite: 1]
    private let pids = ["010D\r", "010C\r", "0111\r", "0104\r"]
    private var currentPidIndex = 0
    
    private var responseAccumulator = ""
    private var lastSpeedMps: Double?
    private var lastRpm: Double?
    private var lastThrottle: Double?
    private var lastLoad: Double?
    
    public override init() {
        super.init()
        centralManager = CBCentralManager(
            delegate: self,
            queue: nil, // Run on Main queue to guarantee timely delegate execution
            options: [CBCentralManagerOptionShowPowerAlertKey: true]
        )
    }
    
    public func start() {
        if centralManager.state == .poweredOn {
            startScanWithTimeout()
        }
    }
    
    public func centralManagerDidUpdateState(_ central: CBCentralManager) {
        print("[OBD] Bluetooth state: \(central.state.rawValue)")
        if central.state == .poweredOn {
            startScanWithTimeout()
        } else {
            delegate?.didChangeConnectionStatus(isConnected: false)
        }
    }
    
    private func matchesTarget(_ name: String) -> Bool {
        let upper = name.uppercased()
        if let target = targetDeviceIdentifier?.uppercased(), !target.isEmpty {
            return upper.contains(target)
        }
        return upper.contains("OBDBLE") || upper.contains("OBD-BLE") || upper.contains("OBDII") || upper.contains("IOS-VLINK")
    }
    
    private func startScanWithTimeout() {
        guard centralManager.state == .poweredOn, peripheral == nil, !isScanning else { return }
        
        isScanning = true
        centralManager.stopScan()
        scanTimeoutWorkItem?.cancel()
        
        print("[OBD] Scanning for OBD adapter...")
        centralManager.scanForPeripherals(withServices: nil, options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])
        
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self else { return }
            self.isScanning = false
            if self.peripheral == nil {
                self.centralManager.stopScan()
                self.startScanWithTimeout()
            }
        }
        scanTimeoutWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 8.0, execute: workItem)
    }
    
    public func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        let peripheralName = peripheral.name ?? ""
        let localName = (advertisementData[CBAdvertisementDataLocalNameKey] as? String) ?? ""
        let combined = "\(peripheralName) \(localName)".trimmingCharacters(in: .whitespaces)
        
        print("[OBD] Discovered device: '\(combined)' (RSSI: \(RSSI))")
        
        if matchesTarget(combined) {
            print("[OBD] Matching adapter found: '\(combined)'. Connecting...")
            scanTimeoutWorkItem?.cancel()
            isScanning = false
            centralManager.stopScan()
            self.peripheral = peripheral
            centralManager.connect(peripheral, options: nil)
        }
    }
    
    public func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("[OBD] Connected to: \(peripheral.name ?? "OBDBLE"). Discovering services...")
        self.peripheral = peripheral
        peripheral.delegate = self
        self.initStep = 0
        self.handshakeStarted = false
        self.isHandshakeDone = false
        self.isPollingActive = false
        self.responseAccumulator = ""
        
        // Notify UI that the BLE link is established
        delegate?.didChangeConnectionStatus(isConnected: true)
        peripheral.discoverServices(nil)
    }
    
    public func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("[OBD] Failed to connect: \(error?.localizedDescription ?? "error")")
        cleanup()
        startScanWithTimeout()
    }
    
    public func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("[OBD] Disconnected. Retrying...")
        cleanup()
        delegate?.didChangeConnectionStatus(isConnected: false)
        startScanWithTimeout()
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services, error == nil else { return }
        for service in services {
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics, error == nil else { return }
        
        for char in characteristics {
            if char.properties.contains(.notify) || char.properties.contains(.indicate) {
                self.rxCharacteristic = char
                peripheral.setNotifyValue(true, for: char)
            }
            if char.properties.contains(.write) || char.properties.contains(.writeWithoutResponse) {
                self.txCharacteristic = char
            }
        }
        
        if txCharacteristic != nil && rxCharacteristic != nil && !handshakeStarted {
            handshakeStarted = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                self?.sendNextInitCommand()
            }
        }
    }
    
    private func sendNextInitCommand() {
        guard let p = peripheral, let tx = txCharacteristic else { return }
        
        if initStep < initCommands.count {
            let cmd = initCommands[initStep]
            initStep += 1
            print("[OBD] Handshake TX: \(cmd.trimmingCharacters(in: .whitespacesAndNewlines))")
            writeRaw(cmd, to: p, characteristic: tx)
            
            let delay = (cmd.contains("ATZ") || cmd.contains("ATSP0")) ? 0.35 : 0.18
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                self?.sendNextInitCommand()
            }
        }
    }
    
    private func startPolling() {
        guard !isPollingActive else { return }
        isPollingActive = true
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.pollTimer?.invalidate()
            print("[OBD] >>> OBD POLLING TIMER STARTED <<<")
            self.pollTimer = Timer.scheduledTimer(withTimeInterval: 0.10, repeats: true) { [weak self] _ in
                self?.pollNextPid()
            }
        }
    }
    
    private func pollNextPid() {
        guard let p = peripheral, let tx = txCharacteristic, isHandshakeDone else { return }
        let pidCmd = pids[currentPidIndex]
        currentPidIndex = (currentPidIndex + 1) % pids.count
        writeRaw(pidCmd, to: p, characteristic: tx)
    }
    
    private func writeRaw(_ cmd: String, to peripheral: CBPeripheral, characteristic: CBCharacteristic) {
        guard let data = cmd.data(using: .ascii) else { return }
        let writeType: CBCharacteristicWriteType = characteristic.properties.contains(.writeWithoutResponse) ? .withoutResponse : .withResponse
        peripheral.writeValue(data, for: characteristic, type: writeType)
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value,
              let chunk = String(data: data, encoding: .ascii) else { return }
        
        responseAccumulator.append(chunk)
        
        if responseAccumulator.contains(">") {
            let message = responseAccumulator
            responseAccumulator = ""
            parseElm(message)
        }
    }
    
    private func parseElm(_ raw: String) {
        let clean = raw.replacingOccurrences(of: " ", with: "")
                       .replacingOccurrences(of: "\r", with: "")
                       .replacingOccurrences(of: "\n", with: "")
                       .replacingOccurrences(of: ">", with: "")
                       .uppercased()
        
        if clean.contains("4100") {
            print("[OBD] ECU validated: \(clean)")
            if !isHandshakeDone {
                isHandshakeDone = true
                delegate?.didChangeConnectionStatus(isConnected: true)
                startPolling()
            }
        }
        
        if clean.contains("UNABLETOCONNECT") || clean.contains("BUSINIT:ERROR") || clean.contains("CANERROR") {
            print("[OBD] ECU error: \(clean)")
            delegate?.didChangeConnectionStatus(isConnected: false)
            return
        }
        
        if let range = clean.range(of: "410D") {
            let payload = String(clean[range.upperBound...])
            if payload.count >= 2, let val = Int(payload.prefix(2), radix: 16) {
                lastSpeedMps = Double(val) / 3.6
                if !isHandshakeDone {
                    isHandshakeDone = true
                    delegate?.didChangeConnectionStatus(isConnected: true)
                    startPolling()
                }
            }
        }
        
        if let range = clean.range(of: "410C") {
            let payload = String(clean[range.upperBound...])
            if payload.count >= 4, let val = Int(payload.prefix(4), radix: 16) {
                lastRpm = Double(val) / 4.0
                if !isHandshakeDone {
                    isHandshakeDone = true
                    delegate?.didChangeConnectionStatus(isConnected: true)
                    startPolling()
                }
            }
        }
        
        if let range = clean.range(of: "4111") {
            let payload = String(clean[range.upperBound...])
            if payload.count >= 2, let val = Int(payload.prefix(2), radix: 16) {
                lastThrottle = (Double(val) * 100.0) / 255.0
            }
        }
        
        if let range = clean.range(of: "4104") {
            let payload = String(clean[range.upperBound...])
            if payload.count >= 2, let val = Int(payload.prefix(2), radix: 16) {
                lastLoad = (Double(val) * 100.0) / 255.0
            }
        }
        
        delegate?.didUpdateObdData(
            speed: lastSpeedMps,
            rpm: lastRpm,
            throttle: lastThrottle,
            load: lastLoad
        )
    }
    
    private func cleanup() {
        DispatchQueue.main.async { [weak self] in
            self?.pollTimer?.invalidate()
            self?.pollTimer = nil
        }
        txCharacteristic = nil
        rxCharacteristic = nil
        handshakeStarted = false
        isHandshakeDone = false
        isScanning = false
        isPollingActive = false
        initStep = 0
        responseAccumulator = ""
        if let p = peripheral {
            centralManager.cancelPeripheralConnection(p)
            self.peripheral = nil
        }
    }
    
    public func stop() {
        scanTimeoutWorkItem?.cancel()
        centralManager.stopScan()
        cleanup()
    }
}