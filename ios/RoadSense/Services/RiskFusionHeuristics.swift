import Foundation

/// Phase 9.1: the hand-authored heuristic half of the on-device risk
/// score, extracted from RoadSenseOnDeviceRiskEngine.swift so it is
/// testable (that file imports CoreML and references generated model
/// classes that only exist inside an Xcode build, so it cannot be part of
/// the `swift test`-able module at all) and so raw ML model output stays
/// visibly separate from hand-authored adjustments, per the project
/// directive's own instruction.
///
/// This is a line-for-line port of what was previously inline in
/// RoadSenseOnDeviceRiskEngine.predictRisk's steps 5-10 -- same constants,
/// same order of operations (order matters: the false-positive suppressor
/// clamps the running total AFTER road-context/attention additions, and
/// the hard overrides apply AFTER that clamp). It must not be
/// independently tuned without re-verifying against that history; see
/// RoadSenseOnDeviceRiskEngine's own tests (none exist in this
/// environment -- CoreML models aren't loadable outside an Xcode build --
/// which is exactly why keeping this half separately testable matters).
public struct RiskBreakdown: Equatable {
  /// (0.5*poliRisk + 0.5*vzRisk) * weatherMult -- the two CoreML models'
  /// raw, weighted-combined output. The ONE component here that is not a
  /// hand-authored heuristic.
  public var mlFusionBase: Double = 0.0
  public var jerkBoost: Double = 0.0
  public var hrVelocityBoost: Double = 0.0
  public var trafficContext: Double = 0.0
  public var pedestrianContext: Double = 0.0
  public var eyesClosedPenalty: Double = 0.0
  /// The "driver isn't watching AND moving fast enough to matter" penalty,
  /// including its own pedestrian escalation -- kept as one component
  /// since in the original code it is one conditional block, not two
  /// independently-triggerable additions.
  public var attentionPenalty: Double = 0.0
  /// Added by the lead-closing-fast + eyes-off-road + speed combination,
  /// only when the false-positive suppressor and the eyesClosedSec>=3
  /// hard override did not already apply.
  public var leadClosingHardBoost: Double = 0.0
  public var falsePositiveSuppressionApplied: Bool = false
  public var hardOverrideApplied: Bool = false
  /// Final value, clamped to [0, 99.9] -- same clamp and same ceiling as
  /// before. This is the number BackgroundCaptureService actually sends;
  /// every field above explains how it was reached, not a separate score.
  public var total: Double = 0.0
}

public func applyRiskHeuristics(
  mlFusionBase: Double,
  jerk: Double,
  hrVelocity: Double,
  trafficLevel: String,
  pedestriansDetected: Bool,
  eyesClosedSec: Double,
  eyesOnRoad: Bool,
  speedKmh: Double,
  leadClosingFast: Bool,
  gyroMag: Double,
  effectiveAccel: Double,
  inRoundabout: Bool,
  classPred: Int,
  weatherMult: Double
) -> RiskBreakdown {
  var breakdown = RiskBreakdown()
  breakdown.mlFusionBase = mlFusionBase
  var riskScore = mlFusionBase

  if jerk > 12.0 {
    breakdown.jerkBoost = min(25.0, jerk * 1.5)
    riskScore += breakdown.jerkBoost
  }

  if hrVelocity > 3.0 {
    breakdown.hrVelocityBoost = min(15.0, hrVelocity * 2.5)
    riskScore += breakdown.hrVelocityBoost
  }

  let trafficBase: Double
  switch trafficLevel {
  case "empty": trafficBase = 0.0
  case "light": trafficBase = 2.0
  case "moderate": trafficBase = 6.0
  case "crowded": trafficBase = 12.0
  default: trafficBase = 0.0
  }
  breakdown.trafficContext = trafficBase
  riskScore += trafficBase

  if pedestriansDetected {
    breakdown.pedestrianContext = 8.0
    riskScore += 8.0
  }

  let eyesClosedGraceSeconds = 0.5
  let eyesClosedRampCeilingSeconds = 3.0
  if eyesClosedSec > eyesClosedGraceSeconds {
    let rampProgress = min(
      1.0,
      (eyesClosedSec - eyesClosedGraceSeconds) / (eyesClosedRampCeilingSeconds - eyesClosedGraceSeconds)
    )
    breakdown.eyesClosedPenalty = rampProgress * 60.0
    riskScore += breakdown.eyesClosedPenalty
  }

  if !eyesOnRoad && speedKmh > 10.0 {
    var penalty = 15.0 + (trafficBase * 2.0)
    if pedestriansDetected {
      penalty += 20.0
    }
    breakdown.attentionPenalty = penalty
    riskScore += penalty
  }

  let isSwerving = (gyroMag > 0.8 && effectiveAccel < 24.0)
  if (inRoundabout || isSwerving) && (classPred == 0 || classPred == 1) && effectiveAccel < 24.0 {
    let capped = min(riskScore, 18.0 * weatherMult)
    if capped < riskScore {
      breakdown.falsePositiveSuppressionApplied = true
    }
    riskScore = capped
  }

  if eyesClosedSec >= 3.0 {
    riskScore = 99.0
    breakdown.hardOverrideApplied = true
  } else if !eyesOnRoad && leadClosingFast && speedKmh >= 20.0 {
    breakdown.leadClosingHardBoost = 25.0
    riskScore += 25.0
  }

  breakdown.total = min(99.9, max(0.0, riskScore))
  return breakdown
}
