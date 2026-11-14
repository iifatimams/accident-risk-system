import XCTest
@testable import RoadSenseAuth

final class MultimodalRiskFusionTests: XCTestCase {
    func testAllThreeModalitiesMissingGivesZeroRiskAndZeroEvidenceQuality() {
        let estimate = fuseMultimodalRisk(
            vehicle: .unavailable, attention: .unavailable, road: .unavailable,
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertEqual(estimate.currentRisk, 0.0)
        XCTAssertEqual(estimate.evidenceQuality, 0.0)
        XCTAssertEqual(estimate.riskLevel, .low)
        XCTAssertEqual(estimate.dominantFactor, "none")
    }

    func testMissingModalityIsNotTreatedAsSafeWhenOthersAreDangerous() {
        // Road camera occluded (unavailable), but attention+vehicle both
        // report real danger -- the missing road evidence must not drag
        // the total down to "safe."
        let estimate = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 80.0, confidence: 1.0),
            attention: ModalityEvidence(value: 80.0, confidence: 1.0),
            road: .unavailable,
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertGreaterThan(estimate.currentRisk, 50.0)
        XCTAssertLessThan(estimate.evidenceQuality, 1.0)  // road modality absence is still visible
    }

    func testLowConfidenceEvidenceContributesLessThanHighConfidenceEvidence() {
        let lowConfidence = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 90.0, confidence: 0.1),
            attention: .unavailable, road: .unavailable,
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        let highConfidence = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 90.0, confidence: 1.0),
            attention: .unavailable, road: .unavailable,
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertLessThan(lowConfidence.currentRisk, highConfidence.currentRisk)
    }

    func testStationaryPedestrianAloneDoesNotProduceHighRisk() {
        // A low-magnitude road-hazard reading (pedestrian visible but not
        // in/near the path) must not alone push risk into "high."
        let estimate = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 5.0, confidence: 1.0),
            attention: ModalityEvidence(value: 0.0, confidence: 1.0),
            road: ModalityEvidence(value: 8.0, confidence: 1.0),
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertEqual(estimate.riskLevel, .low)
    }

    func testFiveSecondAheadIsReportedSeparatelyFromCurrentRisk() {
        let estimate = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 10.0, confidence: 1.0),
            attention: ModalityEvidence(value: 5.0, confidence: 1.0),
            road: ModalityEvidence(value: 5.0, confidence: 1.0),
            fiveSecondVehicleHazardProb: 0.9, contextualAdjustment: 0.0
        )
        XCTAssertNotEqual(estimate.fiveSecondAheadRisk, estimate.currentRisk)
        XCTAssertGreaterThan(estimate.fiveSecondAheadRisk ?? 0, estimate.currentRisk)
    }

    func testFiveSecondAheadIsNilWhenVehicleModelInputUnavailable() {
        let estimate = fuseMultimodalRisk(
            vehicle: .unavailable, attention: .unavailable, road: .unavailable,
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertNil(estimate.fiveSecondAheadRisk)
    }

    func testDominantFactorNamesTheLargestWeightedContribution() {
        let estimate = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 5.0, confidence: 1.0),
            attention: ModalityEvidence(value: 70.0, confidence: 1.0),
            road: ModalityEvidence(value: 10.0, confidence: 1.0),
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertEqual(estimate.dominantFactor, "driver-attention")
    }

    func testDominantFactorIsNoneWhenEverythingIsNearZero() {
        let estimate = fuseMultimodalRisk(
            vehicle: ModalityEvidence(value: 0.2, confidence: 1.0),
            attention: ModalityEvidence(value: 0.1, confidence: 1.0),
            road: ModalityEvidence(value: 0.0, confidence: 1.0),
            fiveSecondVehicleHazardProb: nil, contextualAdjustment: 0.0
        )
        XCTAssertEqual(estimate.dominantFactor, "none")
    }

    func testRiskLevelThresholdsMatchClassification() {
        XCTAssertEqual(classifyRiskLevel(0.0), .low)
        XCTAssertEqual(classifyRiskLevel(19.9), .low)
        XCTAssertEqual(classifyRiskLevel(20.0), .medium)
        XCTAssertEqual(classifyRiskLevel(44.9), .medium)
        XCTAssertEqual(classifyRiskLevel(45.0), .high)
        XCTAssertEqual(classifyRiskLevel(69.9), .high)
        XCTAssertEqual(classifyRiskLevel(70.0), .veryHigh)
        XCTAssertEqual(classifyRiskLevel(99.9), .veryHigh)
    }

    func testModalityEvidenceConfidenceIsClampedToUnitRange() {
        let over = ModalityEvidence(value: 1.0, confidence: 5.0)
        let under = ModalityEvidence(value: 1.0, confidence: -5.0)
        XCTAssertEqual(over.confidence, 1.0)
        XCTAssertEqual(under.confidence, 0.0)
    }

    func testRollingRiskHistoryTrendIsZeroWithInsufficientSpan() {
        var history = RollingRiskHistory()
        history.record(total: 10.0, at: 0.0)
        history.record(total: 90.0, at: 0.2)  // spans only 0.2s -- below the 1.5s minimum
        XCTAssertEqual(history.recentTrend(), 0.0)
    }

    func testRollingRiskHistoryTrendReflectsRisingRisk() {
        var history = RollingRiskHistory()
        history.record(total: 10.0, at: 0.0)
        history.record(total: 50.0, at: 2.0)
        XCTAssertEqual(history.recentTrend(), 20.0, accuracy: 0.01)  // (50-10)/2.0
    }

    func testRollingRiskHistoryDropsEntriesOutsideWindow() {
        var history = RollingRiskHistory(windowSeconds: 5.0)
        history.record(total: 10.0, at: 0.0)
        history.record(total: 20.0, at: 10.0)  // first entry is now outside the 5s window
        // Only one entry remains -> no span -> trend is 0, not a huge spurious spike.
        XCTAssertEqual(history.recentTrend(), 0.0)
    }
}
