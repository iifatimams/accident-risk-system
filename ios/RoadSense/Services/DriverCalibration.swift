import Foundation

/// Per-driver calibration: everyone's resting head position and natural
/// eye-gaze center differs (seat position, phone-mount angle, how far
/// forward their head tilts when relaxed). Without this, the fixed
/// thresholds in GazeThresholds are a reasonable AVERAGE, not a personal
/// one -- a driver who naturally sits with their head tilted slightly
/// down, or has an off-center phone mount, could be flagged as
/// distracted for simply looking at the road their own way.
///
/// This records what "attentive" actually looks like for one driver (a
/// short "just look at the road normally" session) and subtracts that
/// personal baseline from every subsequent raw reading before it's
/// compared against the existing (still shared, still unvalidated-
/// against-real-footage) thresholds -- i.e. it re-centers the
/// measurement, it does not change the thresholds themselves.
public struct CalibrationBaseline: Codable, Equatable {
  public var pitch: Double
  public var yaw: Double
  public var roll: Double
  public var gazeHorizontal: Double
  public var gazeVertical: Double
  public var sampleCount: Int
  public var calibratedAt: TimeInterval

  public init(pitch: Double, yaw: Double, roll: Double, gazeHorizontal: Double, gazeVertical: Double, sampleCount: Int, calibratedAt: TimeInterval) {
    self.pitch = pitch
    self.yaw = yaw
    self.roll = roll
    self.gazeHorizontal = gazeHorizontal
    self.gazeVertical = gazeVertical
    self.sampleCount = sampleCount
    self.calibratedAt = calibratedAt
  }
}

/// One instantaneous, uncalibrated reading fed into a calibration
/// session while the driver is asked to just look at the road normally.
/// `gaze` is nil for a frame where pupil landmarks weren't found --
/// those frames are simply excluded from the average, not treated as
/// zero (which would silently bias the baseline toward "centered").
public struct CalibrationSample: Equatable {
  public var pitch: Double
  public var yaw: Double
  public var roll: Double
  public var gaze: GazeOffset?

  public init(pitch: Double, yaw: Double, roll: Double, gaze: GazeOffset?) {
    self.pitch = pitch
    self.yaw = yaw
    self.roll = roll
    self.gaze = gaze
  }
}

/// Accumulates calibration samples and produces a `CalibrationBaseline`
/// once enough have been collected. A pure value accumulator (no
/// Vision/CoreML dependency), so it's independently unit-testable --
/// same split as the rest of this file's pattern (AttentionHysteresis,
/// gazeOffset).
public struct CalibrationSession {
  public private(set) var samples: [CalibrationSample] = []
  /// Minimum samples before `finish()` will produce a baseline --
  /// roughly a few seconds of frames, not one lucky/unlucky frame.
  public let minimumSamples: Int

  public init(minimumSamples: Int = 30) {
    self.minimumSamples = minimumSamples
  }

  public mutating func record(_ sample: CalibrationSample) {
    samples.append(sample)
  }

  public var progress: Double {
    min(1.0, Double(samples.count) / Double(minimumSamples))
  }

  public var isComplete: Bool {
    samples.count >= minimumSamples
  }

  /// nil if not enough samples were collected yet. `now` is injected
  /// (not `Date()` read internally) so this stays a pure function of
  /// its inputs, like the rest of this file.
  public func finish(now: TimeInterval) -> CalibrationBaseline? {
    guard isComplete else {
      return nil
    }
    let meanPitch = samples.map(\.pitch).reduce(0, +) / Double(samples.count)
    let meanYaw = samples.map(\.yaw).reduce(0, +) / Double(samples.count)
    let meanRoll = samples.map(\.roll).reduce(0, +) / Double(samples.count)
    let gazeSamples = samples.compactMap(\.gaze)
    let meanGazeH = gazeSamples.isEmpty ? 0.0 : gazeSamples.map(\.horizontal).reduce(0, +) / Double(gazeSamples.count)
    let meanGazeV = gazeSamples.isEmpty ? 0.0 : gazeSamples.map(\.vertical).reduce(0, +) / Double(gazeSamples.count)
    return CalibrationBaseline(
      pitch: meanPitch, yaw: meanYaw, roll: meanRoll,
      gazeHorizontal: meanGazeH, gazeVertical: meanGazeV,
      sampleCount: samples.count, calibratedAt: now
    )
  }
}

/// Re-centers one raw reading around a personal baseline, so the
/// EXISTING thresholds (tuned around "0 == neutral") still apply
/// meaningfully to a driver whose own neutral isn't exactly 0. When
/// `baseline` is nil (never calibrated), returns the input unchanged --
/// calibration is an enhancement, never a requirement to function.
public func applyCalibration(
  pitch: Double, yaw: Double, roll: Double, gaze: GazeOffset?, baseline: CalibrationBaseline?
) -> (pitch: Double, yaw: Double, roll: Double, gaze: GazeOffset?) {
  guard let baseline = baseline else {
    return (pitch, yaw, roll, gaze)
  }
  let adjustedGaze = gaze.map {
    GazeOffset(horizontal: $0.horizontal - baseline.gazeHorizontal, vertical: $0.vertical - baseline.gazeVertical)
  }
  return (pitch - baseline.pitch, yaw - baseline.yaw, roll - baseline.roll, adjustedGaze)
}
