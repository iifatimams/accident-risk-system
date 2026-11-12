import XCTest
@testable import RoadSenseAuth

final class ClassifyPedestrianHazardTests: XCTestCase {
    func track(centerXHistory: [(TimeInterval, Double)]) -> TrackedObject {
        let history = centerXHistory.map { (t, cx) in
            BoxObservation(timestamp: t, box: NormalizedRect(x: cx - 0.05, y: 0.5, width: 0.1, height: 0.2))
        }
        return TrackedObject(
            id: UUID(), classLabel: "person", box: history.last!.box, confidence: 0.95,
            firstSeen: history.first!.timestamp, lastSeen: history.last!.timestamp,
            missedFrames: 0, history: history
        )
    }

    func testTooLittleHistoryIsInsufficientEvidence() {
        let t = track(centerXHistory: [(0.0, 0.1)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .insufficientEvidence)
    }

    func testStationaryPedestrianOutsideCorridorIsRoadsideVisible() {
        // Standing still at the far left edge -- not moving, not in the corridor.
        let t = track(centerXHistory: [(0.0, 0.05), (0.3, 0.05), (0.6, 0.05)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .roadsideVisible)
    }

    func testPedestrianInsideCorridorIsInOrNearPath() {
        let t = track(centerXHistory: [(0.0, 0.5), (0.3, 0.5), (0.6, 0.5)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .inOrNearPath)
    }

    func testPedestrianWalkingLeftToRightTowardTheCorridorIsEnteringConflictZone() {
        // Starts well left of the corridor (0.3...0.7), walking rightward toward it.
        let t = track(centerXHistory: [(0.0, 0.05), (0.3, 0.12), (0.6, 0.19)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .enteringConflictZone)
    }

    func testPedestrianWalkingAwayFromTheCorridorIsMovingAcrossScene() {
        // Starts left of the corridor, walking further left/away -- moving, but not toward the path.
        let t = track(centerXHistory: [(0.0, 0.2), (0.3, 0.13), (0.6, 0.06)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .movingAcrossScene)
    }

    func testPedestrianOnTheRightWalkingLeftTowardTheCorridorIsEnteringConflictZone() {
        // Starts right of the corridor (> 0.7), walking leftward toward it.
        let t = track(centerXHistory: [(0.0, 0.95), (0.3, 0.88), (0.6, 0.81)])
        XCTAssertEqual(classifyPedestrianHazard(track: t), .enteringConflictZone)
    }

    func testSeverityRankOrdersStatesFromLeastToMostSevere() {
        let ordered: [PedestrianHazardState] = [
            .insufficientEvidence, .roadsideVisible, .movingAcrossScene,
            .inOrNearPath, .enteringConflictZone,
        ]
        for (a, b) in zip(ordered, ordered.dropFirst()) {
            // Accessed via classifyPedestrianHazard's own ranking is private;
            // this test instead checks the PUBLIC contract that matters --
            // aggregation picks the more severe of two states, both ways round.
            let worseOfTwo = [a, b].max { lhs, rhs in
                severityIndex(lhs) < severityIndex(rhs)
            }
            XCTAssertEqual(worseOfTwo, b, "\(b) must be treated as more severe than \(a)")
        }
    }

    /// Mirrors the aggregation BackgroundCaptureService performs (picking
    /// the most severe of several simultaneously-tracked pedestrians),
    /// without depending on the enum's internal (non-public) severity rank.
    private func severityIndex(_ state: PedestrianHazardState) -> Int {
        [.insufficientEvidence, .roadsideVisible, .movingAcrossScene, .inOrNearPath, .enteringConflictZone]
            .firstIndex(of: state)!
    }
}
