import XCTest
@testable import RoadSenseAuth

/// Phase 9.1: RiskFusionHeuristics.swift is a line-for-line port of what
/// used to be inline, untestable code inside RoadSenseOnDeviceRiskEngine
/// (which cannot be unit tested at all in this environment -- it imports
/// CoreML and generated model classes only available inside an Xcode
/// build). These tests pin down the exact behavior that file's own
/// comments describe, so this heuristic half can never silently drift
/// from what those comments claim.
final class RiskFusionHeuristicsTests: XCTestCase {
    func baseline(
        mlFusionBase: Double = 0.0, jerk: Double = 0.0, hrVelocity: Double = 0.0,
        trafficLevel: String = "empty", pedestriansDetected: Bool = false,
        eyesClosedSec: Double = 0.0, eyesOnRoad: Bool = true, speedKmh: Double = 50.0,
        leadClosingFast: Bool = false, gyroMag: Double = 0.0, effectiveAccel: Double = 9.8,
        inRoundabout: Bool = false, classPred: Int = 2, weatherMult: Double = 1.0
    ) -> RiskBreakdown {
        applyRiskHeuristics(
            mlFusionBase: mlFusionBase, jerk: jerk, hrVelocity: hrVelocity,
            trafficLevel: trafficLevel, pedestriansDetected: pedestriansDetected,
            eyesClosedSec: eyesClosedSec, eyesOnRoad: eyesOnRoad, speedKmh: speedKmh,
            leadClosingFast: leadClosingFast, gyroMag: gyroMag, effectiveAccel: effectiveAccel,
            inRoundabout: inRoundabout, classPred: classPred, weatherMult: weatherMult
        )
    }

    func testPureMLFusionBaseWithNoAdjustmentsPassesThroughUnchanged() {
        let b = baseline(mlFusionBase: 15.0)
        XCTAssertEqual(b.total, 15.0, accuracy: 1e-9)
        XCTAssertEqual(b.jerkBoost, 0.0)
        XCTAssertEqual(b.hrVelocityBoost, 0.0)
    }

    func testJerkBoostOnlyAppliesAboveThresholdAndIsCapped() {
        XCTAssertEqual(baseline(jerk: 12.0).jerkBoost, 0.0, "exactly at the threshold must not trigger it")
        XCTAssertEqual(baseline(jerk: 13.0).jerkBoost, 13.0 * 1.5, accuracy: 1e-9)
        XCTAssertEqual(baseline(jerk: 100.0).jerkBoost, 25.0, "must cap at 25")
    }

    func testHrVelocityBoostOnlyAppliesAboveThresholdAndIsCapped() {
        XCTAssertEqual(baseline(hrVelocity: 3.0).hrVelocityBoost, 0.0)
        XCTAssertEqual(baseline(hrVelocity: 4.0).hrVelocityBoost, 4.0 * 2.5, accuracy: 1e-9)
        XCTAssertEqual(baseline(hrVelocity: 100.0).hrVelocityBoost, 15.0, "must cap at 15")
    }

    func testTrafficContextMatchesEachBand() {
        XCTAssertEqual(baseline(trafficLevel: "empty").trafficContext, 0.0)
        XCTAssertEqual(baseline(trafficLevel: "light").trafficContext, 2.0)
        XCTAssertEqual(baseline(trafficLevel: "moderate").trafficContext, 6.0)
        XCTAssertEqual(baseline(trafficLevel: "crowded").trafficContext, 12.0)
    }

    func testPedestrianContextIsAFlatEightWhenDetected() {
        XCTAssertEqual(baseline(pedestriansDetected: false).pedestrianContext, 0.0)
        XCTAssertEqual(baseline(pedestriansDetected: true).pedestrianContext, 8.0)
    }

    func testEyesClosedRampsContinuouslyBetweenGraceAndCeiling() {
        XCTAssertEqual(baseline(eyesClosedSec: 0.5).eyesClosedPenalty, 0.0, "at the grace period, no penalty yet")
        XCTAssertEqual(baseline(eyesClosedSec: 1.75).eyesClosedPenalty, 30.0, accuracy: 1e-9, "halfway to the 3.0s ceiling -> half of 60")
        XCTAssertEqual(baseline(eyesClosedSec: 3.0).eyesClosedPenalty, 60.0, accuracy: 1e-6)
    }

    func testAttentionPenaltyOnlyFiresAboveTenKmhAndEscalatesWithTrafficAndPedestrians() {
        XCTAssertEqual(baseline(eyesOnRoad: false, speedKmh: 10.0).attentionPenalty, 0.0, "at exactly 10 km/h, must not fire")
        let base = baseline(trafficLevel: "empty", pedestriansDetected: false, eyesOnRoad: false, speedKmh: 11.0)
        XCTAssertEqual(base.attentionPenalty, 15.0, accuracy: 1e-9)
        let withTraffic = baseline(trafficLevel: "crowded", pedestriansDetected: false, eyesOnRoad: false, speedKmh: 11.0)
        XCTAssertEqual(withTraffic.attentionPenalty, 15.0 + 12.0 * 2.0, accuracy: 1e-9)
        let withPedestrian = baseline(trafficLevel: "empty", pedestriansDetected: true, eyesOnRoad: false, speedKmh: 11.0)
        XCTAssertEqual(withPedestrian.attentionPenalty, 15.0 + 20.0, accuracy: 1e-9)
    }

    func testFalsePositiveSuppressorCapsRoundaboutAndSwerveCasesBelowTheHardCollisionThreshold() {
        // A high ML fusion base that would otherwise read as a near-miss,
        // but in a roundabout at normal accel -- must be capped to 18 *
        // weatherMult, not reported as a real high-risk event.
        let b = baseline(mlFusionBase: 80.0, effectiveAccel: 10.0, inRoundabout: true, classPred: 1)
        XCTAssertTrue(b.falsePositiveSuppressionApplied)
        XCTAssertEqual(b.total, 18.0, accuracy: 1e-9)
    }

    func testFalsePositiveSuppressorDoesNotApplyWhenNotRoundaboutOrSwerving() {
        let b = baseline(mlFusionBase: 80.0, gyroMag: 0.1, effectiveAccel: 10.0, inRoundabout: false, classPred: 1)
        XCTAssertFalse(b.falsePositiveSuppressionApplied)
        XCTAssertEqual(b.total, 80.0, accuracy: 1e-9)
    }

    func testEyesClosedThreeSecondsIsAHardOverrideToNinetyNine() {
        let b = baseline(mlFusionBase: 5.0, eyesClosedSec: 3.0)
        XCTAssertTrue(b.hardOverrideApplied)
        XCTAssertEqual(b.total, 99.0, accuracy: 1e-9)
    }

    func testEyesClosedHardOverrideWinsOverFalsePositiveSuppression() {
        // Even inside a roundabout, 3+ seconds of closed eyes must still
        // win -- a real safety condition must never be suppressed by the
        // false-positive guard meant for ordinary turning/braking noise.
        let b = baseline(mlFusionBase: 5.0, eyesClosedSec: 3.5, effectiveAccel: 10.0, inRoundabout: true, classPred: 1)
        XCTAssertEqual(b.total, 99.0, accuracy: 1e-9)
    }

    func testLeadClosingHardBoostOnlyFiresWithEyesOffRoadAndSufficientSpeed() {
        let notFast = baseline(mlFusionBase: 10.0, eyesOnRoad: false, speedKmh: 19.9, leadClosingFast: true)
        XCTAssertEqual(notFast.leadClosingHardBoost, 0.0)
        let fast = baseline(mlFusionBase: 10.0, eyesOnRoad: false, speedKmh: 20.0, leadClosingFast: true)
        XCTAssertEqual(fast.leadClosingHardBoost, 25.0, accuracy: 1e-9)
        // 10 (ml base) + 15 (attentionPenalty, since eyesOnRoad=false and
        // speedKmh=20 > 10 also triggers that block) + 25 (this boost).
        XCTAssertEqual(fast.total, 50.0, accuracy: 1e-9)
    }

    func testFinalTotalIsAlwaysClampedToTheDocumentedRange() {
        XCTAssertEqual(baseline(mlFusionBase: -50.0).total, 0.0, "must never go negative")
        XCTAssertEqual(baseline(mlFusionBase: 500.0).total, 99.9, "must never exceed the 99.9 ceiling")
    }
}
