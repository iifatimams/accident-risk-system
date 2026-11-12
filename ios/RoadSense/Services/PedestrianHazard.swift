import Foundation

/// Phase 7: pedestrian collision-risk context, reusing the Phase 5 tracker
/// over the REAL `VNDetectHumanRectanglesRequest` detections (unlike the
/// rectangle heuristic feeding Phase 6, this one is an Apple-trained
/// detector -- see BackgroundCaptureService.swift's own comment on that
/// distinction). Per the directive: "visibility alone is not an imminent
/// collision" -- the common classifications below (roadsideVisible,
/// movingAcrossScene) are deliberately unalarming, and the two
/// conflict-relevant states only fire from actual position/motion
/// evidence, never from a pedestrian simply being in frame.
public enum PedestrianHazardState: Equatable {
  case roadsideVisible
  case inOrNearPath
  case movingAcrossScene
  case enteringConflictZone
  case insufficientEvidence

  /// Severity order for aggregating several simultaneously-tracked
  /// pedestrians into one published field -- see
  /// BackgroundCaptureService.swift. Higher is more severe.
  var severityRank: Int {
    switch self {
    case .insufficientEvidence: return 0
    case .roadsideVisible: return 1
    case .movingAcrossScene: return 2
    case .inOrNearPath: return 3
    case .enteringConflictZone: return 4
    }
  }

  public var payloadValue: String {
    switch self {
    case .roadsideVisible: return "roadside_visible"
    case .inOrNearPath: return "in_or_near_path"
    case .movingAcrossScene: return "moving_across_scene"
    case .enteringConflictZone: return "entering_conflict_zone"
    case .insufficientEvidence: return "insufficient_evidence"
    }
  }
}

public struct PedestrianHazardThresholds {
  /// Below this many history samples, lateral motion direction is too
  /// noisy to classify at all.
  public var minHistoryForEvidence: Int = 3
  /// The ego vehicle's approximate forward travel corridor, same
  /// definition and same default band as Phase 6's vehicle corridor.
  public var corridorCenterXRange: ClosedRange<Double> = 0.3...0.7
  /// Normalized frame-widths per second of |centerX change|. Below this,
  /// a pedestrian reads as essentially stationary (roadside) rather than
  /// "moving" -- deliberately NOT gated on ego speed (unlike Phase 6):
  /// pedestrian path conflict is about the pedestrian's own motion
  /// relative to the corridor, not an image-growth artifact that only
  /// appears at speed, so there is no equivalent false-positive mechanism
  /// to guard against here.
  public var movingLateralVelocityThreshold: Double = 0.15

  public static let `default` = PedestrianHazardThresholds()
}

/// Classifies ONE already-tracked pedestrian. The caller decides which
/// track(s) to classify and how to combine several into one published
/// signal (see PedestrianHazardState.severityRank).
public func classifyPedestrianHazard(
  track: TrackedObject,
  thresholds: PedestrianHazardThresholds = .default
) -> PedestrianHazardState {
  guard track.history.count >= thresholds.minHistoryForEvidence else {
    return .insufficientEvidence
  }
  if thresholds.corridorCenterXRange.contains(track.box.centerX) {
    return .inOrNearPath
  }

  let first = track.history.first!
  let last = track.history.last!
  let dt = last.timestamp - first.timestamp
  let lateralVelocity = dt > 0 ? (last.box.centerX - first.box.centerX) / dt : 0.0
  let isMoving = abs(lateralVelocity) >= thresholds.movingLateralVelocityThreshold

  guard isMoving else {
    return .roadsideVisible
  }
  let movingTowardCorridor =
    track.box.centerX < thresholds.corridorCenterXRange.lowerBound
    ? lateralVelocity > 0
    : lateralVelocity < 0
  return movingTowardCorridor ? .enteringConflictZone : .movingAcrossScene
}
