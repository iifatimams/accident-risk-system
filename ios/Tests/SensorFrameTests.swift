import XCTest
@testable import RoadSenseAuth

/// Regression test for the bug where `toDictionary()` computed accel_mag/
/// gyro_mag from the raw axes but never put the raw axes themselves into
/// the dictionary that gets sent to the backend and stored in
/// session_samples -- so accel_x/y/z and gyro_x/y/z always arrived as
/// null even though CMMotionManager was feeding them into SensorFrame.
final class SensorFrameTests: XCTestCase {
    func testToDictionaryIncludesRawAccelAndGyroAxes() {
        var frame = SensorFrame(streamId: "test-stream", timestamp: 100.0)
        frame.accelX = 1.5
        frame.accelY = -2.5
        frame.accelZ = 9.8
        frame.gyroX = 0.1
        frame.gyroY = -0.2
        frame.gyroZ = 0.3

        let dict = frame.toDictionary()

        XCTAssertEqual(dict["accel_x"] as? Double, 1.5)
        XCTAssertEqual(dict["accel_y"] as? Double, -2.5)
        XCTAssertEqual(dict["accel_z"] as? Double, 9.8)
        XCTAssertEqual(dict["gyro_x"] as? Double, 0.1)
        XCTAssertEqual(dict["gyro_y"] as? Double, -0.2)
        XCTAssertEqual(dict["gyro_z"] as? Double, 0.3)

        // The derived magnitudes must still be present and correct.
        let expectedAccelMag = sqrt(1.5 * 1.5 + 2.5 * 2.5 + 9.8 * 9.8)
        let expectedGyroMag = sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3)
        XCTAssertEqual(dict["accel_mag"] as? Double ?? 0, expectedAccelMag, accuracy: 1e-9)
        XCTAssertEqual(dict["gyro_mag"] as? Double ?? 0, expectedGyroMag, accuracy: 1e-9)
    }

    func testToDictionaryOmitsAccelAndGyroKeysWhenUnavailable() {
        let frame = SensorFrame(streamId: "test-stream", timestamp: 100.0)

        let dict = frame.toDictionary()

        XCTAssertNil(dict["accel_x"])
        XCTAssertNil(dict["accel_y"])
        XCTAssertNil(dict["accel_z"])
        XCTAssertNil(dict["gyro_x"])
        XCTAssertNil(dict["gyro_y"])
        XCTAssertNil(dict["gyro_z"])
        XCTAssertNil(dict["accel_mag"])
        XCTAssertNil(dict["gyro_mag"])
    }
}
