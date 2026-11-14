import Foundation

/// Persists one `CalibrationBaseline` per driver account id, so swapping
/// drivers in the same car (or the same driver on a different phone,
/// once they sign in) doesn't apply someone else's calibration. Same
/// UserDefaults-backed pattern as CollisionIncidentStore.swift -- this
/// project's own precedent for "no new local database/framework" at
/// this scale (a handful of drivers per device, not thousands).
public final class DriverCalibrationStore {
  public static let shared = DriverCalibrationStore()
  private let defaultsKey = "roadsense_driver_calibrations"
  private let queue = DispatchQueue(label: "ae.aus.roadsense.calibrationStore")
  private let defaults: UserDefaults

  public init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
  }

  public func baseline(forDriverId driverId: Int) -> CalibrationBaseline? {
    queue.sync {
      loadAllLocked()[String(driverId)]
    }
  }

  public func save(_ baseline: CalibrationBaseline, forDriverId driverId: Int) {
    queue.sync {
      var all = loadAllLocked()
      all[String(driverId)] = baseline
      saveAllLocked(all)
    }
  }

  public func clear(forDriverId driverId: Int) {
    queue.sync {
      var all = loadAllLocked()
      all.removeValue(forKey: String(driverId))
      saveAllLocked(all)
    }
  }

  private func loadAllLocked() -> [String: CalibrationBaseline] {
    guard let data = defaults.data(forKey: defaultsKey),
      let decoded = try? JSONDecoder().decode([String: CalibrationBaseline].self, from: data)
    else {
      return [:]
    }
    return decoded
  }

  private func saveAllLocked(_ all: [String: CalibrationBaseline]) {
    guard let data = try? JSONEncoder().encode(all) else {
      return
    }
    defaults.set(data, forKey: defaultsKey)
  }
}
