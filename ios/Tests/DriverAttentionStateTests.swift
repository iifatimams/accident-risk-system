import XCTest
@testable import RoadSenseAuth

final class ClassifyGazeTests: XCTestCase {
    let neutral = (pitch: 0.0, yaw: 0.0, roll: 0.0, ear: 0.3, confidence: Float(0.9))

    func testNeutralPoseIsAttentive() {
        XCTAssertEqual(
            classifyGaze(pitch: neutral.pitch, yaw: neutral.yaw, roll: neutral.roll,
                         eyeAspectRatio: neutral.ear, faceDetected: true, confidence: neutral.confidence),
            .attentive
        )
    }

    func testNoFaceIsFaceNotDetectedRegardlessOfOtherValues() {
        XCTAssertEqual(
            classifyGaze(pitch: 0, yaw: 0, roll: 0, eyeAspectRatio: 0.3, faceDetected: false, confidence: 0.9),
            .faceNotDetected
        )
    }

    /// The whole point of the confidence gate: a degraded (dark, occluded,
    /// glare) reading must not be silently trusted as a real pose reading,
    /// and must NOT default to "attentive" just because noisy pitch/yaw
    /// happen to land near zero.
    func testLowConfidenceIsUncertainEvenWithAttentivePose() {
        XCTAssertEqual(
            classifyGaze(pitch: 0, yaw: 0, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.1),
            .uncertain
        )
    }

    func testEyesClosedTakesPriorityOverSimultaneousHeadTurn() {
        XCTAssertEqual(
            classifyGaze(pitch: -0.5, yaw: 0.6, roll: 0, eyeAspectRatio: 0.05, faceDetected: true, confidence: 0.9),
            .eyesClosed
        )
    }

    func testPitchDownIsLookingDown() {
        XCTAssertEqual(
            classifyGaze(pitch: -0.5, yaw: 0, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.9),
            .lookingDown
        )
    }

    /// This case did not exist at all before Phase 2: the previous
    /// classifier only checked `pitch < -0.4`, so looking UP (e.g. at a
    /// vanity mirror) could never be flagged no matter how far back the
    /// head tilted.
    func testPitchUpIsLookingUp() {
        XCTAssertEqual(
            classifyGaze(pitch: 0.5, yaw: 0, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.9),
            .lookingUp
        )
    }

    func testYawBeyondThresholdIsLeftOrRight() {
        let right = classifyGaze(pitch: 0, yaw: 0.6, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.9)
        let left = classifyGaze(pitch: 0, yaw: -0.6, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.9)
        XCTAssertEqual(right, .lookingRight)
        XCTAssertEqual(left, .lookingLeft)
        XCTAssertNotEqual(right, left)
    }

    func testSmallYawWithinThresholdStaysAttentive() {
        XCTAssertEqual(
            classifyGaze(pitch: 0, yaw: 0.1, roll: 0, eyeAspectRatio: 0.3, faceDetected: true, confidence: 0.9),
            .attentive
        )
    }
}

final class AttentionHysteresisTests: XCTestCase {
    func testStartsAttentive() {
        let machine = AttentionHysteresis()
        XCTAssertEqual(machine.currentState, .attentive)
    }

    func testBriefNonAttentiveReadingDoesNotCommit() {
        var machine = AttentionHysteresis(enterDwell: 1.5, exitDwell: 0.5)
        let state = machine.update(instantaneous: .lookingDown, now: 0.0)
        XCTAssertEqual(state, .attentive, "a single instantaneous reading must not immediately commit")
    }

    func testSustainedNonAttentiveReadingCommitsAfterEnterDwell() {
        var machine = AttentionHysteresis(enterDwell: 1.5, exitDwell: 0.5)
        _ = machine.update(instantaneous: .lookingDown, now: 0.0)
        _ = machine.update(instantaneous: .lookingDown, now: 1.0)
        XCTAssertEqual(machine.currentState, .attentive, "not sustained long enough yet")
        let committed = machine.update(instantaneous: .lookingDown, now: 1.6)
        XCTAssertEqual(committed, .lookingDown)
    }

    /// The actual bug fix under test: previously, ANY single instantaneous
    /// "attentive" reading instantly cleared a pending/committed distraction
    /// with zero debounce, even mid-genuine-distraction. A single noisy
    /// frame must no longer be enough to flip back.
    func testBriefAttentiveBlipDuringSustainedDistractionDoesNotImmediatelyClear() {
        var machine = AttentionHysteresis(enterDwell: 1.5, exitDwell: 0.5)
        _ = machine.update(instantaneous: .lookingLeft, now: 0.0)
        let committed = machine.update(instantaneous: .lookingLeft, now: 1.6)
        XCTAssertEqual(committed, .lookingLeft)
        // One single noisy "attentive" frame, immediately followed by the
        // real distraction resuming.
        let afterBlip = machine.update(instantaneous: .attentive, now: 1.63)
        XCTAssertEqual(afterBlip, .lookingLeft, "a single frame must not instantly clear a committed distraction")
        let resumed = machine.update(instantaneous: .lookingLeft, now: 1.7)
        XCTAssertEqual(resumed, .lookingLeft)
    }

    func testSustainedAttentiveReadingClearsAfterExitDwell() {
        var machine = AttentionHysteresis(enterDwell: 1.5, exitDwell: 0.5)
        _ = machine.update(instantaneous: .lookingDown, now: 0.0)
        _ = machine.update(instantaneous: .lookingDown, now: 1.6)
        XCTAssertEqual(machine.currentState, .lookingDown)
        _ = machine.update(instantaneous: .attentive, now: 1.7)
        XCTAssertEqual(machine.currentState, .lookingDown, "not sustained long enough to clear yet")
        let cleared = machine.update(instantaneous: .attentive, now: 2.3)
        XCTAssertEqual(cleared, .attentive)
    }

    func testSwitchingCandidateBeforeDwellElapsesRestartsTheTimer() {
        var machine = AttentionHysteresis(enterDwell: 1.5, exitDwell: 0.5)
        _ = machine.update(instantaneous: .lookingDown, now: 0.0)
        _ = machine.update(instantaneous: .lookingLeft, now: 1.0) // switches candidate before committing
        let notYetDown = machine.update(instantaneous: .lookingLeft, now: 1.4)
        XCTAssertEqual(notYetDown, .attentive, "lookingDown never got enough dwell before the candidate changed")
        let committedLeft = machine.update(instantaneous: .lookingLeft, now: 2.6)
        XCTAssertEqual(committedLeft, .lookingLeft)
    }

    func testIsEyesOnRoadOnlyTrueForAttentive() {
        XCTAssertTrue(DriverAttentionState.attentive.isEyesOnRoad)
        for state: DriverAttentionState in [.lookingLeft, .lookingRight, .lookingDown, .lookingUp, .eyesClosed, .faceNotDetected, .uncertain] {
            XCTAssertFalse(state.isEyesOnRoad, "\(state) must not count as eyes-on-road")
        }
    }
}
