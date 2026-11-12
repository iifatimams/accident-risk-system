import XCTest
@testable import RoadSenseAuth

final class CameraOrientationTests: XCTestCase {
    func testPortraitIsNinetyDegreesForBothCameras() {
        XCTAssertEqual(videoRotationAngle(for: .portrait, camera: .front), 90)
        XCTAssertEqual(videoRotationAngle(for: .portrait, camera: .back), 90)
    }

    func testPortraitUpsideDownIsTwoSeventyForBothCameras() {
        XCTAssertEqual(videoRotationAngle(for: .portraitUpsideDown, camera: .front), 270)
        XCTAssertEqual(videoRotationAngle(for: .portraitUpsideDown, camera: .back), 270)
    }

    /// Landscape is where front and back MUST diverge -- the two sensors are
    /// mounted rotated 180 degrees relative to each other. A regression here
    /// (e.g. applying the same angle to both cameras, as the old hardcoded
    /// 90-degrees-always behavior effectively did) would produce an upright
    /// rear view and a sideways/upside-down front view, or vice versa.
    func testLandscapeAnglesDivergeBetweenFrontAndBackCamera() {
        XCTAssertNotEqual(
            videoRotationAngle(for: .landscapeLeft, camera: .front),
            videoRotationAngle(for: .landscapeLeft, camera: .back)
        )
        XCTAssertNotEqual(
            videoRotationAngle(for: .landscapeRight, camera: .front),
            videoRotationAngle(for: .landscapeRight, camera: .back)
        )
    }

    func testLandscapeLeftAndRightAreOppositeForEachCamera() {
        XCTAssertNotEqual(
            videoRotationAngle(for: .landscapeLeft, camera: .front),
            videoRotationAngle(for: .landscapeRight, camera: .front)
        )
        XCTAssertNotEqual(
            videoRotationAngle(for: .landscapeLeft, camera: .back),
            videoRotationAngle(for: .landscapeRight, camera: .back)
        )
    }

    func testAllAnglesAreValidAVCaptureConnectionAngles() {
        let validAngles: Set<Double> = [0, 90, 180, 270]
        let orientations: [PhoneOrientation] = [.portrait, .portraitUpsideDown, .landscapeLeft, .landscapeRight]
        for orientation in orientations {
            for camera: CameraPosition in [.front, .back] {
                XCTAssertTrue(
                    validAngles.contains(videoRotationAngle(for: orientation, camera: camera)),
                    "\(orientation)/\(camera) produced an angle AVCaptureConnection cannot accept"
                )
            }
        }
    }
}
