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
/// Where the pupil sits inside the eye socket, relative to the eye's own
/// bounding box -- a genuine EYE-GAZE signal, independent of head pose.
/// This is what lets the system catch "phone held below the steering
/// wheel, head relatively straight, eyes directed downward" -- the
/// capstone directive's own example of something head-pose-only
/// classification cannot see at all.
///
/// Both components are normalized by the eye's own width/height, in
/// Vision's normalized-image coordinate convention (origin bottom-left,
/// y increases upward) -- `vertical < 0` means the pupil sits toward the
/// BOTTOM of the eye socket, i.e. looking down, using the same sign
/// convention as `GazeThresholds.pitchDown` deliberately, so the two
/// signals combine without a sign-flip bug.
public struct GazeOffset: Equatable {
  public var horizontal: Double
  public var vertical: Double
}

/// Computes `GazeOffset` from one eye's pupil point and its own contour
/// points (both in Vision's normalized image coordinates). Returns nil
/// when the eye region is degenerate (zero width/height -- e.g. a
/// landmark failure) rather than dividing by zero and fabricating a
/// reading.
public func gazeOffset(pupil: (x: Double, y: Double), eyeContour: [(x: Double, y: Double)]) -> GazeOffset? {
  guard !eyeContour.isEmpty else {
    return nil
  }
  let minX = eyeContour.map(\.x).min()!
  let maxX = eyeContour.map(\.x).max()!
  let minY = eyeContour.map(\.y).min()!
  let maxY = eyeContour.map(\.y).max()!
  let width = maxX - minX
  let height = maxY - minY
  guard width > 0, height > 0 else {
    return nil
  }
  let centerX = (minX + maxX) / 2.0
  let centerY = (minY + maxY) / 2.0
  return GazeOffset(horizontal: (pupil.x - centerX) / width, vertical: (pupil.y - centerY) / height)
}

/// Averages two eyes' independently-computed offsets. Using both eyes
/// (not just one) is more robust to a single degenerate landmark read;
/// either eye missing still yields a usable reading from the other,
/// rather than discarding the whole frame.
public func averageGazeOffset(left: GazeOffset?, right: GazeOffset?) -> GazeOffset? {
  switch (left, right) {
  case (nil, nil):
    return nil
  case (let l?, nil):
    return l
  case (nil, let r?):
    return r
  case (let l?, let r?):
    return GazeOffset(horizontal: (l.horizontal + r.horizontal) / 2.0, vertical: (l.vertical + r.vertical) / 2.0)
  }
}

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
  /// Pupil-within-eye-socket vertical offset below this reads as looking
  /// down BY EYE GAZE ALONE, independent of head pitch -- this is the
  /// signal that catches a phone held below the wheel while the head
  /// stays level. A starting point, not tuned against labeled footage
  /// (same caveat as every other threshold in this struct).
  public var gazeVerticalDown: Double = -0.18

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
/// Combines head pose (pitch/yaw) with an actual eye-gaze signal (`gaze`,
/// pupil-within-eye-socket offset -- see `GazeOffset`/`gazeOffset(...)`
/// above) specifically so a driver who holds their head still and moves
/// only their eyes (e.g. a phone held below the wheel) is still caught,
/// which head-pose-only classification could not do at all. This still
/// does not detect phone USE; `.lookingDown` means gaze/pitch evidence
/// only, not a visually confirmed phone in frame. Only VERTICAL gaze
/// offset is used to override head pose so far (the directive's own
/// worked example); horizontal pupil offset is computed but not yet
/// used to override yaw -- left/right head turns were already covered
/// by yaw alone, and no labeled data exists in this environment to
/// validate a horizontal override threshold the way the DMD gaze-zone
/// dataset at least approximates for vertical.
public func classifyGaze(
  pitch: Double,
  yaw: Double,
  roll: Double,
  eyeAspectRatio: Double,
  faceDetected: Bool,
  confidence: Float,
  gaze: GazeOffset? = nil,
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
  // Eye-gaze-alone check FIRST, before head pitch: this is deliberately
  // the path that catches a phone held below the wheel while the head
  // stays level -- head pitch alone would read `.attentive` here, which
  // is exactly the directive's own worked example of what NOT to miss.
  // `gaze` is nil when pupil landmarks weren't reliably found this frame
  // (confidence-aware: falls back to head pitch only, never fabricates
  // an eye-gaze reading it doesn't have).
  if let gaze = gaze, gaze.vertical < thresholds.gazeVerticalDown {
    return .lookingDown
  }
  if pitch < thresholds.pitchDown {
    return .lookingDown
  }
  if pitch > thresholds.pitchUp {
    return .lookingUp
  }
  if yaw > thresholds.yawTurn {
    // Verified on a physical device across two real driving sessions:
    // Vision's positive yaw corresponds to the driver's LEFT, not their
    // right (the front camera's raw buffer coordinate space is mirrored
    // relative to how the driver perceives their own left/right). Signs
    // below are swapped from the original unverified guess to match.
    return .lookingLeft
  }
  if yaw < -thresholds.yawTurn {
    return .lookingRight
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

/// Combines the DMD-trained attention model's binary
/// looking_road/not_looking_road prediction with Apple Vision's own
/// (richer, directional) gaze classification, per the master directive's
/// Phase 2 requirement that DMD's trained output and Vision work
/// together rather than either one silently replacing the other.
///
/// DMD is coarser than Vision (2 classes vs Vision's 8), so it can never
/// supply a specific direction (left/right/down/up) -- it is used ONLY
/// as a confirming/escalating check: when DMD strongly disagrees with
/// Vision's "attentive" by predicting not-looking-road with high
/// confidence, this escalates to `.uncertain` (never a fabricated
/// specific direction DMD didn't actually predict). It deliberately
/// NEVER clears an existing distraction/eyesClosed/faceNotDetected state
/// that Vision already flagged -- disagreement must not resolve toward
/// the SAFER-looking label, since that is the direction a real miss
/// would be most dangerous in. When the model is unavailable
/// (`dmdNotLookingRoadProb == nil`), this returns `visionState`
/// unchanged -- Vision is the fallback, exactly as the directive asks.
public func combineDMDWithVision(
  visionState: DriverAttentionState,
  dmdNotLookingRoadProb: Double?,
  agreementThreshold: Double = 0.65
) -> DriverAttentionState {
  guard let prob = dmdNotLookingRoadProb else {
    return visionState
  }
  if visionState == .attentive && prob >= agreementThreshold {
    return .uncertain
  }
  return visionState
}
