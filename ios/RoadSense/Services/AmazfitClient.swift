import Foundation
import CoreBluetooth

/// Phase 15: Amazfit Helio Strap support, added per the project directive
/// (Fitbit stays out of scope; Amazfit is explicitly in scope). Mirrors
/// WhoopClient.swift's already-working mechanism almost exactly: the
/// standard Bluetooth GATT Heart Rate Service (0x180D), not an
/// Amazfit/Zepp-proprietary protocol -- the directive is explicit that this
/// must use "supported device functionality, not undocumented cloud-
/// password scraping," and the standard service is exactly that. A
/// separate class (not a generalized "WearableClient" merging both) so
/// each device keeps its own identity/delegate, matching the directive's
/// own framing of WHOOP and Amazfit as two parallel, separately-identified
/// devices, and so the already-working WHOOP integration is not touched
/// beyond extracting the shared parsing logic (see WearableHeartRate.swift).
///
/// UNVERIFIED on physical hardware -- per the directive's own instruction,
/// this must be labeled unverified until tested against a real strap.
/// Specifically unconfirmed:
///   - Whether the Helio Strap actually exposes 0x180D/0x2A37 over BLE by
///     default, or requires a "Heart Rate Push" setting to be enabled in
///     the Zepp app first (the directive names this exact setting to
///     investigate). Nothing in this codebase can verify that without the
///     physical strap.
///   - Whether WHOOP and Amazfit can stream simultaneously -- each gets its
///     own CBCentralManager here (same as WhoopClient's own instance), which
///     should not conflict at the iOS BLE stack level, but real simultaneous
///     operation with both devices worn has not been tested.
///   - Only heart rate and RR interval are read (same fields WhoopClient
///     reads) -- no HRV, SpO2, or stress value is fabricated for data the
///     standard Heart Rate Service does not carry.
public protocol AmazfitDelegate: AnyObject {
  func didUpdateAmazfit(heartRate: Double?, rrInterval: Double?)
}

public class AmazfitClient: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
  public weak var delegate: AmazfitDelegate?
  public var targetDeviceIdentifier: String?

  private var centralManager: CBCentralManager!
  private var amazfitPeripheral: CBPeripheral?

  private let heartRateServiceUUID = CBUUID(string: "180D")
  private let heartRateMeasurementCharacteristicUUID = CBUUID(string: "2A37")

  private var pollTimer: Timer?
  private var isStarted = false

  public override init() {
    super.init()
    // Same fix as WhoopClient: never initialize CBCentralManager here, to
    // avoid the main-thread deadlock on app launch that comment documents.
  }

  public func startPolling() {
    guard !isStarted else { return }
    isStarted = true

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
    guard amazfitPeripheral == nil else { return }
    guard centralManager.state == .poweredOn else { return }

    let connectedPeripherals = centralManager.retrieveConnectedPeripherals(withServices: [heartRateServiceUUID])
    for strap in connectedPeripherals {
      if matchesTarget(strap.name) {
        print("  [AmazfitClient] Found matched system-connected strap: \(strap.name ?? "Unknown"). Connecting...")
        connectPeripheral(strap)
        return
      }
    }

    if !centralManager.isScanning {
      print("  [AmazfitClient] Scanning for Heart Rate Broadcaster...")
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
    return name.contains("AMAZFIT") || name.contains("HELIO")
  }

  private func connectPeripheral(_ peripheral: CBPeripheral) {
    centralManager.stopScan()
    amazfitPeripheral = peripheral
    amazfitPeripheral?.delegate = self
    centralManager.connect(peripheral, options: nil)
  }

  public func centralManagerDidUpdateState(_ central: CBCentralManager) {
    if central.state == .poweredOn && isStarted {
      attemptConnection()
    }
  }

  public func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
    let name = peripheral.name ?? ""
    let localName = advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? ""
    let combined = "\(name) \(localName)".uppercased()

    if matchesTarget(combined) {
      print("  [AmazfitClient] Connecting to target: '\(combined)'")
      connectPeripheral(peripheral)
    }
  }

  public func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
    peripheral.discoverServices([heartRateServiceUUID])
  }

  public func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
    amazfitPeripheral = nil
  }

  public func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
    amazfitPeripheral = nil
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
      let data = characteristic.value, !data.isEmpty
    else { return }
    guard let reading = WearableHeartRate.parseHeartRateMeasurement([UInt8](data)) else {
      return
    }
    delegate?.didUpdateAmazfit(heartRate: reading.heartRateBpm, rrInterval: reading.rrIntervalMs)
  }

  public func stop() {
    isStarted = false
    pollTimer?.invalidate()
    pollTimer = nil
    centralManager?.stopScan()
    if let p = amazfitPeripheral {
      centralManager?.cancelPeripheralConnection(p)
      amazfitPeripheral = nil
    }
  }
}
