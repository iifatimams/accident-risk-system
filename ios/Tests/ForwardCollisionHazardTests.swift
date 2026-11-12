import XCTest
@testable import RoadSenseAuth

final class OpticalExpansionTests: XCTestCase {
    func history(_ samples: [(TimeInterval, Double)]) -> [BoxObservation] {
        // Square boxes of side `size`, centered, so area = size^2 and
        // sqrt(area) = size exactly -- makes the expected math exact.
        samples.map { (t, size) in
            BoxObservation(timestamp: t, box: NormalizedRect(x: 0.4, y: 0.4, width: size, height: size))
        }
    }

    func testKnownLinearGrowthProducesTheExpectedTTC() {
        // Size grows from 0.1 to 0.2 over 1 second -> growth rate 0.1/s,
        // TTC = sizeAtEnd / growthRate = 0.2 / 0.1 = 2.0s exactly.
        let h = history([(0.0, 0.1), (0.5, 0.15), (1.0, 0.2)])
        XCTAssertEqual(OpticalExpansion.timeToContactSeconds(history: h)!, 2.0, accuracy: 1e-9)
    }

    func testStaticSizeHasNoTTC() {
        let h = history([(0.0, 0.1), (0.5, 0.1), (1.0, 0.1)])
        XCTAssertNil(OpticalExpansion.timeToContactSeconds(history: h))
    }

    func testShrinkingSizeHasNoTTC() {
        let h = history([(0.0, 0.2), (0.5, 0.15), (1.0, 0.1)])
        XCTAssertNil(OpticalExpansion.timeToContactSeconds(history: h), "a shrinking/receding object must never report a closing TTC")
    }

    func testTooFewSamplesHasNoTTC() {
        let h = history([(0.0, 0.1), (1.0, 0.2)])
        XCTAssertNil(OpticalExpansion.timeToContactSeconds(history: h))
    }

    func testFasterGrowthProducesShorterTTC() {
        let slow = history([(0.0, 0.1), (0.5, 0.11), (1.0, 0.12)])
        let fast = history([(0.0, 0.1), (0.5, 0.3), (1.0, 0.5)])
        let slowTTC = OpticalExpansion.timeToContactSeconds(history: slow)!
        let fastTTC = OpticalExpansion.timeToContactSeconds(history: fast)!
        XCTAssertLessThan(fastTTC, slowTTC, "a faster-approaching object must report a SHORTER time-to-contact")
    }
}

final class ClassifyForwardHazardTests: XCTestCase {
    func growingTrack(samples: Int = 3) -> TrackedObject {
        var history: [BoxObservation] = []
        for i in 0..<samples {
            let size = 0.1 + Double(i) * 0.05
            history.append(BoxObservation(
                timestamp: Double(i) * 0.3,
                box: NormalizedRect(x: 0.4, y: 0.4, width: size, height: size)
            ))
        }
        return TrackedObject(
            id: UUID(), classLabel: "car", box: history.last!.box, confidence: 0.9,
            firstSeen: 0, lastSeen: history.last!.timestamp, missedFrames: 0, history: history
        )
    }

    func testNoTrackIsNoRelevantLeadObject() {
        XCTAssertEqual(classifyForwardHazard(leadTrack: nil, egoSpeedKmh: 50), .noRelevantLeadObject)
    }

    func testTrackOutsideForwardCorridorIsNoRelevantLeadObject() {
        var track = growingTrack()
        track.box = NormalizedRect(x: 0.0, y: 0.4, width: 0.1, height: 0.1)  // far left edge
        // history must also reflect an off-corridor box for centerX to read off-corridor
        track.history = track.history.map {
            BoxObservation(timestamp: $0.timestamp, box: NormalizedRect(x: 0.0, y: 0.4, width: $0.box.width, height: $0.box.height))
        }
        XCTAssertEqual(classifyForwardHazard(leadTrack: track, egoSpeedKmh: 50), .noRelevantLeadObject)
    }

    func testTooLittleHistoryIsInsufficientEvidence() {
        var track = growingTrack(samples: 3)
        track.history = Array(track.history.prefix(1))
        XCTAssertEqual(classifyForwardHazard(leadTrack: track, egoSpeedKmh: 50), .insufficientEvidence)
    }

    func testLowEgoSpeedNeverEscalatesPastTracked() {
        let track = growingTrack()
        // Below minEgoSpeedKmhForClosingEvidence -- even with a growing
        // box, this must not be reported as closing.
        XCTAssertEqual(classifyForwardHazard(leadTrack: track, egoSpeedKmh: 2.0), .leadObjectTracked)
    }

    func testMissingEgoSpeedNeverEscalatesPastTracked() {
        let track = growingTrack()
        XCTAssertEqual(classifyForwardHazard(leadTrack: track, egoSpeedKmh: nil), .leadObjectTracked)
    }

    func testStaticSizeWithGoodSpeedIsJustTracked() {
        let history = [
            BoxObservation(timestamp: 0.0, box: NormalizedRect(x: 0.4, y: 0.4, width: 0.1, height: 0.1)),
            BoxObservation(timestamp: 0.3, box: NormalizedRect(x: 0.4, y: 0.4, width: 0.1, height: 0.1)),
            BoxObservation(timestamp: 0.6, box: NormalizedRect(x: 0.4, y: 0.4, width: 0.1, height: 0.1)),
        ]
        let track = TrackedObject(
            id: UUID(), classLabel: "car", box: history.last!.box, confidence: 0.9,
            firstSeen: 0, lastSeen: 0.6, missedFrames: 0, history: history
        )
        XCTAssertEqual(classifyForwardHazard(leadTrack: track, egoSpeedKmh: 50), .leadObjectTracked)
    }

    func testRapidGrowthWithSufficientSpeedIsIncreasingHazard() {
        var history: [BoxObservation] = []
        for i in 0..<4 {
            let size = 0.1 + Double(i) * 0.15  // fast growth -> short TTC
            history.append(BoxObservation(
                timestamp: Double(i) * 0.3, box: NormalizedRect(x: 0.4, y: 0.4, width: size, height: size)
            ))
        }
        let track = TrackedObject(
            id: UUID(), classLabel: "car", box: history.last!.box, confidence: 0.9,
            firstSeen: 0, lastSeen: history.last!.timestamp, missedFrames: 0, history: history
        )
        let state = classifyForwardHazard(leadTrack: track, egoSpeedKmh: 60)
        XCTAssertEqual(state, .increasingClosingHazard)
    }

    func testSlowerGrowthWithSufficientSpeedIsPossibleClosingMotion() {
        var history: [BoxObservation] = []
        for i in 0..<4 {
            let size = 0.10 + Double(i) * 0.01  // slow growth -> TTC ~3.9s, in the "warning" band
            history.append(BoxObservation(
                timestamp: Double(i) * 0.3, box: NormalizedRect(x: 0.4, y: 0.4, width: size, height: size)
            ))
        }
        let track = TrackedObject(
            id: UUID(), classLabel: "car", box: history.last!.box, confidence: 0.9,
            firstSeen: 0, lastSeen: history.last!.timestamp, missedFrames: 0, history: history
        )
        let state = classifyForwardHazard(leadTrack: track, egoSpeedKmh: 60)
        XCTAssertEqual(state, .possibleClosingMotion)
    }
}
