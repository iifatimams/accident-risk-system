import XCTest
@testable import RoadSenseAuth

final class TemporalTrackerTests: XCTestCase {
    func box(_ x: Double, _ y: Double, _ w: Double, _ h: Double) -> NormalizedRect {
        NormalizedRect(x: x, y: y, width: w, height: h)
    }

    func testIoUOfIdenticalBoxesIsOne() {
        let a = box(0.1, 0.1, 0.2, 0.2)
        XCTAssertEqual(NormalizedRect.iou(a, a), 1.0, accuracy: 1e-9)
    }

    func testIoUOfNonOverlappingBoxesIsZero() {
        XCTAssertEqual(NormalizedRect.iou(box(0, 0, 0.1, 0.1), box(0.5, 0.5, 0.1, 0.1)), 0.0)
    }

    func testFirstDetectionCreatesANewTrack() {
        var tracker = ObjectTracker()
        let tracks = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.0
        )
        XCTAssertEqual(tracks.count, 1)
        XCTAssertEqual(tracks[0].classLabel, "car")
        XCTAssertEqual(tracks[0].missedFrames, 0)
    }

    func testOverlappingDetectionInNextFrameKeepsTheSameTrackId() {
        var tracker = ObjectTracker()
        let first = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.0
        )
        let firstId = first[0].id
        // Slightly moved/grown box, still heavily overlapping -- same car, next frame.
        let second = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.41, 0.41, 0.22, 0.22), confidence: 0.9)],
            at: 0.1
        )
        XCTAssertEqual(second.count, 1)
        XCTAssertEqual(second[0].id, firstId, "a matching detection next frame must keep the same track identity")
        XCTAssertEqual(second[0].history.count, 2)
    }

    func testDifferentClassDetectionsNeverMergeIntoOneTrack() {
        var tracker = ObjectTracker()
        _ = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.0
        )
        let tracks = tracker.update(
            // Same box, but a pedestrian -- must not be matched to the car track.
            detections: [Detection(classLabel: "person", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.1
        )
        XCTAssertEqual(tracks.count, 2, "a different class at the same location must start its own track, not steal the car's")
    }

    func testASingleMissedFrameDoesNotFragmentTheTrack() {
        var tracker = ObjectTracker(maxMissedFrames: 2)
        let first = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.0
        )
        let firstId = first[0].id
        // Frame with no detections at all (motion blur, brief miss).
        let missed = tracker.update(detections: [], at: 0.1)
        XCTAssertEqual(missed.count, 1, "within maxMissedFrames, the track must survive a miss, not disappear")
        XCTAssertEqual(missed[0].missedFrames, 1)
        // Reappears in roughly the same place.
        let recovered = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.41, 0.41, 0.2, 0.2), confidence: 0.9)],
            at: 0.2
        )
        XCTAssertEqual(recovered.count, 1)
        XCTAssertEqual(recovered[0].id, firstId, "a brief miss must not assign a new identity on recovery")
        XCTAssertEqual(recovered[0].missedFrames, 0)
    }

    func testTrackIsDroppedAfterExceedingMaxMissedFrames() {
        var tracker = ObjectTracker(maxMissedFrames: 2)
        _ = tracker.update(
            detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
            at: 0.0
        )
        _ = tracker.update(detections: [], at: 0.1)  // missed=1
        _ = tracker.update(detections: [], at: 0.2)  // missed=2, still alive
        let afterThirdMiss = tracker.update(detections: [], at: 0.3)  // missed=3 > maxMissedFrames
        XCTAssertTrue(afterThirdMiss.isEmpty, "a track lost for longer than maxMissedFrames must be dropped")
    }

    func testHistoryIsBoundedToMaxHistory() {
        var tracker = ObjectTracker(maxHistory: 3)
        for i in 0..<10 {
            _ = tracker.update(
                detections: [Detection(classLabel: "car", box: box(0.4, 0.4, 0.2, 0.2), confidence: 0.9)],
                at: Double(i) * 0.1
            )
        }
        XCTAssertEqual(tracker.activeTracks[0].history.count, 3)
    }
}
