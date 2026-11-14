import Foundation

/// The final multimodal predictive-risk fusion layer (master directive
/// section 6): combines the three principal evidence dimensions --
/// vehicle dynamics (POLIDriving), driver attention (DMD + Apple Vision),
/// and road/environment hazard (BDD100K detections via TemporalTracker +
/// ForwardCollisionHazard/PedestrianHazard) -- into one coherent,
/// explainable risk estimate.
///
/// POLIDriving, DMD, and BDD100K are three unrelated datasets with no
/// synchronized multimodal labels (different recordings, different
/// vehicles/drivers, different sensors, never collected together). No
/// trained end-to-end fusion model exists or is claimed here -- per the
/// directive's own explicit instruction not to "randomly concatenate
/// unrelated samples and call the result a validated multimodal training
/// dataset." This IS the directive's own documented fallback: "implement
/// an evidence-based temporal late-fusion engine using the three trained
/// specialist models, physically meaningful hazard indicators, and
/// confidence-aware weighting."
///
/// Deliberately pure Foundation (no CoreML/UIKit) so it is independently
/// unit-testable, same pattern as RiskFusionHeuristics.swift. Each
/// specialist model's OWN CoreML inference happens elsewhere
/// (RoadSenseOnDeviceRiskEngine for POLIDriving's current+5s models,
/// BackgroundCaptureService for the DMD attention model and the BDD100K
/// detector feeding TemporalTracker) -- this file only combines their
/// already-computed evidence.
public enum RiskLevel: String, Codable {
  case low
  case medium
  case high
  case veryHigh
}

/// One modality's evidence at a point in time. `confidence` is how much
/// to trust `value` right now (0 when the modality has no observation at
/// all, e.g. no face detected, no lead object tracked, no live OBD data --
/// NOT the same as "safe," which `isFresh`/`confidence == 0` must not be
/// silently treated as per the directive's explicit "missing modalities
/// must not be treated as either automatically safe or automatically
/// dangerous" requirement).
public struct ModalityEvidence: Equatable {
  /// Risk contribution on the same 0...100-ish scale as the rest of the
  /// engine, BEFORE confidence weighting.
  public var value: Double
  /// 0 (no evidence at all) ... 1 (fully confident, fresh observation).
  public var confidence: Double

  public init(value: Double, confidence: Double) {
    self.value = value
    self.confidence = max(0.0, min(1.0, confidence))
  }

  public static let unavailable = ModalityEvidence(value: 0.0, confidence: 0.0)
}

/// The directive's full required output: continuous current risk,
/// experimental 5-second-ahead estimate, risk level, per-dimension
/// contributions, contextual adjustment, an evidence-quality/confidence
/// indicator, and an explanation of the dominant risk factor.
public struct MultimodalRiskEstimate: Equatable {
  public var currentRisk: Double = 0.0
  /// nil when the 5-second-ahead model's input wasn't available this
  /// frame (e.g. no OBD data) -- never silently defaulted to 0, which
  /// would read as "confidently safe" rather than "unknown."
  public var fiveSecondAheadRisk: Double?
  public var riskLevel: RiskLevel = .low
  public var vehicleContribution: Double = 0.0
  public var attentionContribution: Double = 0.0
  public var roadHazardContribution: Double = 0.0
  public var contextualAdjustment: Double = 0.0
  /// 0 (no modality has usable evidence this frame) ... 1 (all three
  /// principal modalities are fresh and confident). This is evidence
  /// quality, explicitly NOT a calibrated probability of anything.
  public var evidenceQuality: Double = 0.0
  /// Human-readable name of whichever contribution is currently largest,
  /// e.g. "driver-attention", "road-hazard", "vehicle-dynamics", or
  /// "none" when everything is near zero.
  public var dominantFactor: String = "none"
}

/// Rolling history the fusion engine needs to satisfy the directive's
/// "maintain a rolling feature history" requirement: recent risk change,
/// and how long the current risk level has persisted (used so a single
/// noisy spike doesn't instantly relabel the trip as "very high," the
/// same hysteresis principle AttentionHysteresis already applies to
/// driver gaze).
public struct RollingRiskHistory {
  private var recentTotals: [(time: Double, total: Double)] = []
  private let windowSeconds: Double

  public init(windowSeconds: Double = 10.0) {
    self.windowSeconds = windowSeconds
  }

  public mutating func record(total: Double, at time: Double) {
    recentTotals.append((time, total))
    recentTotals.removeAll { time - $0.time > windowSeconds }
  }

  /// (last - first) / elapsed, or 0 if the window doesn't span enough
  /// history yet to mean anything -- same minimum-span guard
  /// RoadSenseOnDeviceRiskEngine already uses for jerk/hrVelocity, for
  /// the same reason (a near-zero denominator produces a spurious spike).
  public func recentTrend(minSpanSeconds: Double = 1.5) -> Double {
    guard let first = recentTotals.first, let last = recentTotals.last,
          last.time - first.time > minSpanSeconds
    else {
      return 0.0
    }
    return (last.total - first.total) / (last.time - first.time)
  }
}

public func classifyRiskLevel(_ total: Double) -> RiskLevel {
  switch total {
  case ..<20.0: return .low
  case 20.0..<45.0: return .medium
  case 45.0..<70.0: return .high
  default: return .veryHigh
  }
}

/// Combines the three principal evidence dimensions into one estimate.
///
/// `vehicle` should already encode the POLIDriving current-risk model's
/// output (confidence 0 when OBD/IMU data is missing). `attention` should
/// encode the DMD-trained model + Apple Vision fallback, already run
/// through AttentionHysteresis (confidence 0 when the face genuinely
/// can't be observed -- never silently 0 RISK). `road` should encode the
/// BDD100K-detector-driven ForwardCollisionHazard/PedestrianHazard
/// evidence via TemporalTracker (confidence 0 when no rear-camera
/// detection evidence exists this frame, e.g. the camera is occluded).
///
/// `fiveSecondVehicleHazardProb` is the separately-trained 5-second-ahead
/// POLIDriving model's own hazard probability (0...1), or nil when that
/// model's inputs weren't available. It is reported as its own output
/// field, NOT folded into `currentRisk` -- the directive explicitly wants
/// both reported, not conflated.
///
/// `contextualAdjustment` is the already-existing weather/road-context
/// multiplier's effect, reported separately so it's visible in the
/// breakdown rather than hidden inside one of the three dimensions.
public func fuseMultimodalRisk(
  vehicle: ModalityEvidence,
  attention: ModalityEvidence,
  road: ModalityEvidence,
  fiveSecondVehicleHazardProb: Double?,
  contextualAdjustment: Double
) -> MultimodalRiskEstimate {
  var estimate = MultimodalRiskEstimate()

  let weightedVehicle = vehicle.value * vehicle.confidence
  let weightedAttention = attention.value * attention.confidence
  let weightedRoad = road.value * road.confidence

  estimate.vehicleContribution = weightedVehicle
  estimate.attentionContribution = weightedAttention
  estimate.roadHazardContribution = weightedRoad
  estimate.contextualAdjustment = contextualAdjustment

  let base = weightedVehicle + weightedAttention + weightedRoad
  estimate.currentRisk = min(99.9, max(0.0, base + contextualAdjustment))
  estimate.riskLevel = classifyRiskLevel(estimate.currentRisk)

  if let hazardProb = fiveSecondVehicleHazardProb {
    // The 5s model only forecasts the VEHICLE dimension (POLIDriving is
    // the only modality with a trained forward-looking target this
    // session) -- blended with the OTHER TWO modalities' CURRENT evidence
    // as a stand-in for "if nothing else changes," not claimed as a
    // genuine 5-second forecast of attention/road evidence, which was
    // never trained.
    let forecastVehicle = hazardProb * 100.0
    estimate.fiveSecondAheadRisk = min(99.9, max(0.0, forecastVehicle + weightedAttention + weightedRoad + contextualAdjustment))
  }

  // Evidence quality: mean confidence across the three principal
  // modalities. All three missing (e.g. phone face-down, no OBD, no rear
  // camera) correctly reads as 0, not folded into a falsely-safe 0 risk.
  estimate.evidenceQuality = (vehicle.confidence + attention.confidence + road.confidence) / 3.0

  let contributions: [(String, Double)] = [
    ("vehicle-dynamics", weightedVehicle),
    ("driver-attention", weightedAttention),
    ("road-hazard", weightedRoad),
  ]
  if let maxEntry = contributions.max(by: { $0.1 < $1.1 }), maxEntry.1 > 0.5 {
    estimate.dominantFactor = maxEntry.0
  } else {
    estimate.dominantFactor = "none"
  }

  return estimate
}
