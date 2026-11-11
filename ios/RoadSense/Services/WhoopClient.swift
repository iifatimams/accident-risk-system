import Foundation
import CoreBluetooth

public protocol WhoopDelegate: AnyObject {
    func didUpdateWhoop(heartRate: Double?, rrInterval: Double?)
}

public class WhoopClient: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    public weak var delegate: WhoopDelegate?
    public var targetDeviceIdentifier: String?
    
    private var centralManager: CBCentralManager!
    private var whoopPeripheral: CBPeripheral?
    
    private let heartRateServiceUUID = CBUUID(string: "180D")
    private let heartRateMeasurementCharacteristicUUID = CBUUID(string: "2A37")
    
    private var pollTimer: Timer?
    private var isStarted = false
    
    public override init() {
        super.init()
        // FIX: DO NOT initialize CBCentralManager here to prevent main-thread deadlocks on app launch!
    }
    
    public func startPolling() {
        guard !isStarted else { return }
        isStarted = true
        
        // FIX: Only initialize Bluetooth when the session actually starts
        if centralManager == nil {
            centralManager = CBCentralManager(delegate: self, queue: nil)
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.pollTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                self?.attemptConnection()
            }
        }
        
        if centralManager.state == .poweredOn {
            attemptConnection()
        }
    }
    
    private func attemptConnection() {
        guard whoopPeripheral == nil else { return }
        guard centralManager.state == .poweredOn else { return }
        
        let connectedPeripherals = centralManager.retrieveConnectedPeripherals(withServices: [heartRateServiceUUID])
        for strap in connectedPeripherals {
            if matchesTarget(strap.name) {
                print("  [WhoopClient] Found matched system-connected WHOOP: \(strap.name ?? "Unknown"). Connecting...")
                connectPeripheral(strap)
                return
            }
        }
        
        if !centralManager.isScanning {
            print("  [WhoopClient] Scanning for Heart Rate Broadcaster...")
            centralManager.scanForPeripherals(
                withServices: [heartRateServiceUUID],
                options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
            )
        }
    }
    
    private func matchesTarget(_ name: String?) -> Bool {
        guard let name = name?.uppercased() else { return false }
        if let target = targetDeviceIdentifier?.uppercased(), !target.isEmpty {
            return name.contains(target)
        }
        return name.contains("WHOOP")
    }
    
    private func connectPeripheral(_ peripheral: CBPeripheral) {
        centralManager.stopScan()
        whoopPeripheral = peripheral
        whoopPeripheral?.delegate = self
        centralManager.connect(peripheral, options: nil)
    }
    
    public func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn && isStarted {
            attemptConnection()
        }
    }
    
    public func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        let name = peripheral.name ?? ""
        let localName = advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? ""
        let combined = "\(name) \(localName)".uppercased()
        
        if matchesTarget(combined) {
            print("  [WhoopClient] Connecting to target: '\(combined)'")
            connectPeripheral(peripheral)
        }
    }
    
    public func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.discoverServices([heartRateServiceUUID])
    }
    
    public func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        whoopPeripheral = nil
    }
    
    public func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        whoopPeripheral = nil
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services where service.uuid == heartRateServiceUUID {
            peripheral.discoverCharacteristics([heartRateMeasurementCharacteristicUUID], for: service)
        }
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        for char in characteristics where char.uuid == heartRateMeasurementCharacteristicUUID {
            peripheral.setNotifyValue(true, for: char)
        }
    }
    
    public func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == heartRateMeasurementCharacteristicUUID,
              let data = characteristic.value,
              !data.isEmpty else { return }
        parseHeartRate(data)
    }
    
    private func parseHeartRate(_ data: Data) {
        let buffer = [UInt8](data)
        let flags = buffer[0]
        let isUInt16 = (flags & 0x01) != 0
        var offset = 1
        var bpm: Double = 0.0
        
        if isUInt16 && buffer.count >= 3 {
            let val = UInt16(buffer[1]) | (UInt16(buffer[2]) << 8)
            bpm = Double(val)
            offset = 3
        } else if buffer.count >= 2 {
            bpm = Double(buffer[1])
            offset = 2
        }
        
        if (flags & 0x08) != 0 {
            offset += 2
        }
        
        var rrMs: Double? = nil
        if (flags & 0x10) != 0 && buffer.count >= offset + 2 {
            let rawRR = UInt16(buffer[offset]) | (UInt16(buffer[offset + 1]) << 8)
            rrMs = (Double(rawRR) / 1024.0) * 1000.0
        }
        
        if bpm > 0 {
            delegate?.didUpdateWhoop(heartRate: bpm, rrInterval: rrMs)
        }
    }
    
    public func stop() {
        isStarted = false
        pollTimer?.invalidate()
        pollTimer = nil
        centralManager?.stopScan()
        if let p = whoopPeripheral {
            centralManager?.cancelPeripheralConnection(p)
            whoopPeripheral = nil
        }
    }
}