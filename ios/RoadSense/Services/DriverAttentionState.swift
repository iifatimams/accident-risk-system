import Foundation

/// Discrete driver-attention states, replacing a single `eyesOnRoad` boolean
/// with something that can actually say WHY. `eyesOnRoad` is still derived
/// from this (see `BackgroundCaptureService`) for backend compatibility.
///
/// `.uncertain` and `.faceNotDetected` are deliberately distinct from
/// `.attentive`: a missing or low-confidence reading means we don't know
/// where the driver is looking, which is not the same claim as "looking at
/// the road." Per the project directive: never turn unknown into attentive.
public enum DriverAttentionState: Equatable {
  case attentive
  case lookingLeft
  case lookingRight
  case lookingDown
  case lookingUp
  case eyesClosed
  case faceNotDetected
  case uncertain

  /// For the existing backend `eyes_on_road` boolean field. Only `.attentive`
  /// counts as eyes-on-road; everything else -- including `.uncertain` --
  /// is NOT attentive, which is the fix for "unknown treated as attentive."
  public var isEyesOnRoad: Bool { self == .attentive }

  /// Stable wire/dashboard string. Deliberately separate from a raw
  /// `String(describing:)`/CaseIterable name so the payload format doesn't
  /// silently change if a case is ever renamed for Swift-side clarity.
  public var payloadValue: String {
    switch self {
    case .attentive: return "attentive"
    case .lookingLeft: return "looking_left"
    case .lookingRight: return "looking_right"
    case .lookingDown: return "looking_down"
    case .lookingUp: return "looking_up"
    case .eyesClosed: return "eyes_closed"
    case .faceNotDetected: return "face_not_detected"
    case .uncertain: return "uncertain"
    }
  }
}

/// Tunable thresholds for classifying one frame's head pose/EAR into an
/// instantaneous (pre-hysteresis) attention reading. Defaults are the
/// pre-existing values already running in `BackgroundCaptureService`
/// (pitch/yaw distraction, EAR eyes-closed) PLUS a few new ones documented
/// below. None of these have been tuned against labeled real-world footage
/// -- there is no such dataset available in this environment -- so they
/// are starting points for on-device calibration, not validated values.
public struct GazeThresholds {
  /// Head pitched down past this (radians) reads as looking down (e.g. at a
  /// phone in the lap). Pre-existing value.
  public var pitchDown: Double = -0.4
  /// Head pitched up past this reads as looking up (e.g. at a vanity
  /// mirror). NEW -- the previous classifier had no upward threshold at
  /// all, so looking up could never be flagged no matter how far the head
  /// tilted back. Chosen symmetric to `pitchDown` as a starting point.
  public var pitchUp: Double = 0.4
  /// Head yawed past this (either sign) reads as a left/right turn.
  /// Pre-existing value.
  public var yawTurn: Double = 0.45
  /// Eye-aspect-ratio below this reads as eyes closed. Pre-existing value.
  public var earClosed: Double = 0.18
  /// Below this face-detection confidence, pose/EAR readings are untrusted
  /// and the frame reads as `.uncertain` rather than whatever the noisy
  /// pitch/yaw/EAR values happen to say. NEW -- this is the mechanism for
  /// "report unknown states and reduced confidence when detection degrades"
  /// (nighttime, glasses glare, partial occlusion), instead of silently
  /// trusting degraded Vision output.
  public var minConfidence: Float = 0.4

  public static let `default` = GazeThresholds()
}

/// Classifies ONE frame. Pure function, no timing/state -- see
/// `AttentionHysteresis` below for how repeated instantaneous readings turn
/// into a stable, flicker-resistant `DriverAttentionState`.
///
/// Priority order matters: a closed-eye reading pre-empts a simultaneous
/// head-turn reading (can't usefully distinguish "turned and blinking" from
/// "turned with eyes closed" from pose alone, and eyes-closed is the more
/// safety-relevant state to surface).
///
/// KNOWN LIMITATION (documented per project directive, not fixed here):
/// this is head pose, not eye gaze. A driver can hold their head still and
/// move only their eyes to look at a phone in a cupholder, and this will
/// read `.attentive`. True gaze estimation would need pupil tracking, which
/// is not implemented -- see docs/DEVELOPMENT_PROGRESS.md Phase 2.4. This
/// also does not detect phone USE; `.lookingDown` means head pitch only,
/// not a visually confirmed phone in frame.
public func classifyGaze(
  pitch: Double,
  yaw: Double,
  roll: Double,
  eyeAspectRatio: Double,
  faceDetected: Bool,
  confidence: Float,
  thresholds: GazeThresholds = .default
) -> DriverAttentionState {
  guard faceDetected else {
    return .faceNotDetected
  }
  guard confidence >= thresholds.minConfidence else {
    return .uncertain
  }
  if eyeAspectRatio < thresholds.earClosed {
    return .eyesClosed
  }
  if pitch < thresholds.pitchDown {
    return .lookingDown
  }
  if pitch > thresholds.pitchUp {
    return .lookingUp
  }
  if yaw > thresholds.yawTurn {
    // UNVERIFIED sign convention: this labels positive yaw as a turn to the
    // viewer's right. Not confirmed against a physical device (the front
    // camera is not mirrored, so this may be reversed from what the driver
    // perceives as "their" left/right). Confirm by turning the head to a
    // known side and checking debugYaw's sign against this branch; swap
    // the two `return` values below if backward.
    return .lookingRight
  }
  if yaw < -thresholds.yawTurn {
    return .lookingLeft
  }
  return .attentive
}

/// Turns a stream of instantaneous `classifyGaze` readings into a stable,
/// debounced `DriverAttentionState`, fixing a real bug in the previous
/// implementation: entering a distracted state required 1.5s sustained
/// evidence, but a SINGLE frame reading back to "attentive" (Vision pose
/// noise, or a quick glance back mid-turn) instantly cleared it with no
/// debounce at all. During an actual large, sustained head turn, pose
/// noise crossing back under the threshold for just one frame out of
/// several could cancel the pending distraction state and immediately
/// report `eyesOnRoad == true` -- the exact "large head turns ... app
/// still appears to report eyes on the road" failure from the project
/// directive.
///
/// This machine requires sustained evidence in BOTH directions: `enterDwell`
/// to commit to any non-attentive state, `exitDwell` to return to
/// `.attentive`. `exitDwell` is intentionally shorter so a genuine return to
/// the road isn't sluggish, but a single noisy frame can no longer flip the
/// state instantly either way.
public struct AttentionHysteresis {
  public var enterDwell: TimeInterval
  public var exitDwell: TimeInterval
  private var committed: DriverAttentionState = .attentive
  private var candidate: DriverAttentionState = .attentive
  private var candidateSince: TimeInterval?

  public init(enterDwell: TimeInterval = 1.5, exitDwell: TimeInterval = 0.5) {
    self.enterDwell = enterDwell
    self.exitDwell = exitDwell
  }

  public var currentState: DriverAttentionState { committed }

  /// Feed one frame's instantaneous classification. `now` is a monotonic
  /// or wall-clock timestamp in seconds; callers must pass consistently
  /// increasing values (the existing call site uses
  /// `Date().timeIntervalSince1970`, matching the rest of this file).
  @discardableResult
  public mutating func update(instantaneous: DriverAttentionState, now: TimeInterval) -> DriverAttentionState {
    if instantaneous != candidate {
      candidate = instantaneous
      candidateSince = now
    }
    guard let since = candidateSince, candidate != committed else {
      return committed
    }
    let dwell = candidate == .attentive ? exitDwell : enterDwell
    if now - since >= dwell {
      committed = candidate
    }
    return committed
  }
}
