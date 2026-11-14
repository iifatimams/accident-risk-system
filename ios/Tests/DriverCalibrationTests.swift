import XCTest
@testable import RoadSenseAuth

final class CalibrationSessionTests: XCTestCase {
    func testNotCompleteUntilMinimumSamplesReached() {
        var session = CalibrationSession(minimumSamples: 3)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        XCTAssertFalse(session.isComplete)
        XCTAssertNil(session.finish(now: 0))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        XCTAssertTrue(session.isComplete)
        XCTAssertNotNil(session.finish(now: 0))
    }

    func testProgressReflectsFractionOfMinimumSamples() {
        var session = CalibrationSession(minimumSamples: 4)
        XCTAssertEqual(session.progress, 0.0)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        XCTAssertEqual(session.progress, 0.25)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        XCTAssertEqual(session.progress, 1.0)
    }

    func testProgressNeverExceedsOneEvenWithExtraSamples() {
        var session = CalibrationSession(minimumSamples: 2)
        for _ in 0..<10 {
            session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        }
        XCTAssertEqual(session.progress, 1.0)
    }

    func testFinishAveragesPitchYawRoll() {
        var session = CalibrationSession(minimumSamples: 2)
        session.record(CalibrationSample(pitch: -0.1, yaw: 0.2, roll: 0.0, gaze: nil))
        session.record(CalibrationSample(pitch: -0.3, yaw: 0.4, roll: 0.0, gaze: nil))
        let baseline = session.finish(now: 123.0)
        XCTAssertNotNil(baseline)
        XCTAssertEqual(baseline!.pitch, -0.2, accuracy: 1e-9)
        XCTAssertEqual(baseline!.yaw, 0.3, accuracy: 1e-9)
        XCTAssertEqual(baseline!.calibratedAt, 123.0)
        XCTAssertEqual(baseline!.sampleCount, 2)
    }

    func testFinishAveragesOnlyFramesWithAGazeReading() {
        var session = CalibrationSession(minimumSamples: 3)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: GazeOffset(horizontal: 0.1, vertical: -0.1)))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))  // excluded, not treated as (0,0)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: GazeOffset(horizontal: 0.3, vertical: -0.3)))
        let baseline = session.finish(now: 0)
        XCTAssertEqual(baseline?.gazeHorizontal ?? 99, 0.2, accuracy: 1e-9)
        XCTAssertEqual(baseline?.gazeVertical ?? 99, -0.2, accuracy: 1e-9)
    }

    func testFinishWithNoGazeReadingsAtAllDefaultsToZeroOffset() {
        var session = CalibrationSession(minimumSamples: 2)
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        session.record(CalibrationSample(pitch: 0, yaw: 0, roll: 0, gaze: nil))
        let baseline = session.finish(now: 0)
        XCTAssertEqual(baseline?.gazeHorizontal, 0.0)
        XCTAssertEqual(baseline?.gazeVertical, 0.0)
    }
}

final class ApplyCalibrationTests: XCTestCase {
    func testNilBaselineLeavesReadingUnchanged() {
        let result = applyCalibration(pitch: -0.3, yaw: 0.2, roll: 0.0, gaze: GazeOffset(horizontal: 0.1, vertical: -0.1), baseline: nil)
        XCTAssertEqual(result.pitch, -0.3)
        XCTAssertEqual(result.yaw, 0.2)
        XCTAssertEqual(result.gaze, GazeOffset(horizontal: 0.1, vertical: -0.1))
    }

    func testBaselineIsSubtractedFromRawReading() {
        let baseline = CalibrationBaseline(pitch: -0.2, yaw: 0.1, roll: 0.0, gazeHorizontal: 0.05, gazeVertical: -0.1, sampleCount: 30, calibratedAt: 0)
        let result = applyCalibration(pitch: -0.2, yaw: 0.1, roll: 0.0, gaze: GazeOffset(horizontal: 0.05, vertical: -0.1), baseline: baseline)
        // A driver's own personal "neutral" reading, after calibration, must read as exactly neutral (0).
        XCTAssertEqual(result.pitch, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result.yaw, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result.gaze?.horizontal ?? 99, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result.gaze?.vertical ?? 99, 0.0, accuracy: 1e-9)
    }

    func testCalibrationPreservesADeviationFromNeutral() {
        let baseline = CalibrationBaseline(pitch: -0.2, yaw: 0.0, roll: 0.0, gazeHorizontal: 0.0, gazeVertical: 0.0, sampleCount: 30, calibratedAt: 0)
        // Driver's neutral is pitch -0.2; a genuine further look-down to -0.5 must still show up as a -0.3 deviation.
        let result = applyCalibration(pitch: -0.5, yaw: 0.0, roll: 0.0, gaze: nil, baseline: baseline)
        XCTAssertEqual(result.pitch, -0.3, accuracy: 1e-9)
    }

    func testMissingGazeStaysMissingAfterCalibration() {
        let baseline = CalibrationBaseline(pitch: 0, yaw: 0, roll: 0, gazeHorizontal: 0.1, gazeVertical: 0.1, sampleCount: 30, calibratedAt: 0)
        let result = applyCalibration(pitch: 0, yaw: 0, roll: 0, gaze: nil, baseline: baseline)
        XCTAssertNil(result.gaze, "a frame with no gaze reading must not gain a fabricated one just because a baseline exists")
    }
}
