import Foundation

/// Phase 6: experimental forward-hazard estimation from a tracked lead
/// object's apparent-size history, NOT from any fabricated metric
/// distance. A monocular camera cannot give trustworthy absolute distance
/// without validated calibration (per project directive); this module
/// never attempts to. Instead it uses the apparent-size growth rate of a
/// tracked box over time -- "optical time-to-contact" (sometimes called
/// Lee's tau in the perception literature): while an object is genuinely
/// approaching at roughly constant relative speed, dividing its current
/// apparent size by its own rate of growth gives a time-until-contact
/// estimate that needs no absolute size or distance calibration at all.
///
/// REAL ASSUMPTIONS this relies on, stated rather than hidden:
///   - The tracked object's true physical size is roughly constant (nothing
///     is actually changing size) and it's seen face-on-ish, not mostly
///     rotating/turning across the frame.
///   - Closing velocity is roughly constant over the measurement window --
///     a real approach is rarely perfectly constant, so this is a
///     short-horizon estimate, not a precise prediction.
///   - It depends entirely on the upstream detection/tracking being
///     correct. Today's rear-camera "car" detection in
///     BackgroundCaptureService.swift is a crude rectangle heuristic, not
///     a validated detector (see docs/benchmarks/bdd100k_*.md) -- this is
///     explicitly why the classifier below stays conservative
///     (`insufficientEvidence`/`leadObjectTracked`) rather than escalating
///     without strong, sustained evidence.
public enum ForwardHazardState: Equatable {
  case noRelevantLeadObject
  case leadObjectTracked
  case possibleClosingMotion
  case increasingClosingHazard
  case insufficientEvidence

  /// Stable wire/dashboard string, same rationale as
  /// DriverAttentionState.payloadValue: decoupled from the Swift case name
  /// so a future rename doesn't silently change the payload format.
  public var payloadValue: String {
    switch self {
    case .noRelevantLeadObject: return "no_relevant_lead_object"
    case .leadObjectTracked: return "lead_object_tracked"
    case .possibleClosingMotion: return "possible_closing_motion"
    case .increasingClosingHazard: return "increasing_closing_hazard"
    case .insufficientEvidence: return "insufficient_evidence"
    }
  }
}

/// Optical time-to-contact from a track's bounding-box history.
public enum OpticalExpansion {
  /// Returns nil whenever growth isn't measurably present: too little
  /// history, non-positive elapsed time, or the object is static/shrinking
  /// (not approaching) -- never invents a number for those cases.
  public static func timeToContactSeconds(history: [BoxObservation]) -> TimeInterval? {
    guard let first = history.first, let last = history.last, history.count >= 3 else {
      return nil
    }
    let dt = last.timestamp - first.timestamp
    guard dt > 0 else {
      return nil
    }
    // sqrt(area) (a geometric-mean linear size) is used rather than width
    // or height alone so ordinary aspect-ratio jitter between frames
    // (partial occlusion, a slightly different crop) doesn't masquerade as
    // genuine growth in one dimension.
    let sizeAtStart = sqrt(max(first.box.area, 0))
    let sizeAtEnd = sqrt(max(last.box.area, 0))
    guard sizeAtStart > 0, sizeAtEnd > sizeAtStart else {
      return nil
    }
    let growthPerSecond = (sizeAtEnd - sizeAtStart) / dt
    guard growthPerSecond > 0 else {
      return nil
    }
    return sizeAtEnd / growthPerSecond
  }
}

public struct ForwardHazardThresholds {
  /// Below this many history samples, a TTC estimate is too noisy to act
  /// on at all -- report insufficientEvidence rather than guess.
  public var minHistoryForEvidence: Int = 3
  public var closingTTCUrgentSeconds: Double = 2.0
  public var closingTTCWarningSeconds: Double = 4.0
  /// Below this ego speed, apparent growth in a lead object's box is more
  /// likely parking-lot/stop-and-go jitter than a real closing hazard, and
  /// the directive explicitly warns against assuming a stationary lead
  /// vehicle is dangerous just because it looks closer. Below this speed,
  /// this classifier deliberately never escalates past leadObjectTracked.
  public var minEgoSpeedKmhForClosingEvidence: Double = 8.0
  /// The lead object's box center must fall in this horizontal band
  /// (0...1, left-to-right) to count as "ahead" at all -- an approximate
  /// forward travel corridor, not a claim about actual lane geometry.
  public var forwardCorridorCenterXRange: ClosedRange<Double> = 0.3...0.7

  public static let `default` = ForwardHazardThresholds()
}

/// Classifies ONE lead candidate (already selected by the caller, e.g. the
/// nearest/largest vehicle-class track) against ego speed context. Pure,
/// stateless, no hidden smoothing -- the apparent-size-growth computation
/// above already spans the track's bounded history window, which is this
/// module's only smoothing; see its doc comment on real-world tuning
/// before adding more.
public func classifyForwardHazard(
  leadTrack: TrackedObject?,
  egoSpeedKmh: Double?,
  thresholds: ForwardHazardThresholds = .default
) -> ForwardHazardState {
  guard let track = leadTrack else {
    return .noRelevantLeadObject
  }
  let centerX = track.box.centerX
  guard thresholds.forwardCorridorCenterXRange.contains(centerX) else {
    return .noRelevantLeadObject
  }
  guard track.history.count >= thresholds.minHistoryForEvidence else {
    return .insufficientEvidence
  }
  guard let speed = egoSpeedKmh, speed >= thresholds.minEgoSpeedKmhForClosingEvidence else {
    // A tracked object ahead, but not enough speed context to say anything
    // about closing motion -- never defaults to "safe" or "closing" here.
    return .leadObjectTracked
  }
  guard let ttc = OpticalExpansion.timeToContactSeconds(history: track.history) else {
    return .leadObjectTracked
  }
  if ttc <= thresholds.closingTTCUrgentSeconds {
    return .increasingClosingHazard
  }
  if ttc <= thresholds.closingTTCWarningSeconds {
    return .possibleClosingMotion
  }
  return .leadObjectTracked
}
