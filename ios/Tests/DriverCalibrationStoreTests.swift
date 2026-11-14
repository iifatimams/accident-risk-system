import XCTest
@testable import RoadSenseAuth

final class DriverCalibrationStoreTests: XCTestCase {
    func makeStore() -> DriverCalibrationStore {
        DriverCalibrationStore(defaults: UserDefaults(suiteName: "roadsense-calibration-test-\(UUID().uuidString)")!)
    }

    func sampleBaseline() -> CalibrationBaseline {
        CalibrationBaseline(pitch: -0.1, yaw: 0.05, roll: 0.0, gazeHorizontal: 0.02, gazeVertical: -0.03, sampleCount: 30, calibratedAt: 1000.0)
    }

    func testBaselineIsNilBeforeAnyCalibration() {
        let store = makeStore()
        XCTAssertNil(store.baseline(forDriverId: 42))
    }

    func testSaveThenLoadRoundTrips() {
        let store = makeStore()
        store.save(sampleBaseline(), forDriverId: 42)
        XCTAssertEqual(store.baseline(forDriverId: 42), sampleBaseline())
    }

    func testDifferentDriversHaveIndependentBaselines() {
        let store = makeStore()
        let driverA = CalibrationBaseline(pitch: -0.1, yaw: 0, roll: 0, gazeHorizontal: 0, gazeVertical: 0, sampleCount: 30, calibratedAt: 0)
        let driverB = CalibrationBaseline(pitch: 0.2, yaw: 0, roll: 0, gazeHorizontal: 0, gazeVertical: 0, sampleCount: 30, calibratedAt: 0)
        store.save(driverA, forDriverId: 1)
        store.save(driverB, forDriverId: 2)
        XCTAssertEqual(store.baseline(forDriverId: 1), driverA)
        XCTAssertEqual(store.baseline(forDriverId: 2), driverB)
    }

    func testSavingAgainOverwritesThePreviousBaselineForThatDriver() {
        let store = makeStore()
        store.save(sampleBaseline(), forDriverId: 42)
        let recalibrated = CalibrationBaseline(pitch: 0.5, yaw: 0, roll: 0, gazeHorizontal: 0, gazeVertical: 0, sampleCount: 40, calibratedAt: 2000.0)
        store.save(recalibrated, forDriverId: 42)
        XCTAssertEqual(store.baseline(forDriverId: 42), recalibrated)
    }

    func testClearRemovesTheBaseline() {
        let store = makeStore()
        store.save(sampleBaseline(), forDriverId: 42)
        store.clear(forDriverId: 42)
        XCTAssertNil(store.baseline(forDriverId: 42))
    }
}
