// RoadSense/Services/CollisionDetector.swift
import Foundation

/// A single detected (and not-suppressed) collision suspicion.
public struct CollisionSuspicion: Codable {
  /// Stable, globally-unique, generated once here. Carried through local
  /// storage, every upload attempt/retry, and the server's own dedup
  /// (client_incident_id) -- the single thread that ties "what the phone
  /// detected" to "what the server recorded" to "what the driver decided",
  /// regardless of how many times or how late it gets uploaded.
  public let id: String
  public let accelMagnitudeMs2: Double
  public let score: Double
  public let onsetEstimate: TimeInterval
  public let modelVersion: String
}

/// Independent, on-device collision detection: a pure threshold check on
/// combined acceleration magnitude (gravity-inclusive, m/s^2). No dependency
/// on RoadSenseOnDeviceRiskEngine, the camera/vision pipeline, or
/// network/server availability -- it must keep working even if those are
/// unavailable, since that is the whole point of detecting on-device.
///
/// The threshold matches accident_risk/processing/collision_detector.py
/// exactly, so the on-device and server-side checks agree on what counts as
/// a suspected impact. Referenced to the VZCrash research (round 2/3): real
/// crash clips averaged ~3.9g peak vs ~1.5g for normal/near-miss driving, so
/// 2.5g is well above ordinary-driving noise but below observed crash
/// peaks -- NOT validated against a real impact on this project's own
/// hardware. Treat this as a first, rule-based pass only.
public final class CollisionDetector {
  public static let gravityMs2 = 9.80665
  public static let thresholdMs2 = 2.5 * gravityMs2
  public static let modelVersion = "kinematic_threshold_v1_unvalidated_ios"

  private let suppressionWindowSeconds: Double
  private var lastTriggerAt: TimeInterval = -.greatestFiniteMagnitude

  /// suppressionWindowSeconds: duplicate suppression for THIS device's own
  /// stream (unlike the server, which tracks many streams and so keys its
  /// own suppression by stream_id -- a single phone only ever has one).
  public init(suppressionWindowSeconds: Double = 120.0) {
    self.suppressionWindowSeconds = suppressionWindowSeconds
  }

  /// Call with every raw combined-acceleration sample (m/s^2) as it
  /// arrives. Returns a suspicion only when the sample is at/above
  /// threshold AND is not a duplicate of a recent trigger.
  public func evaluate(accelMagnitudeMs2: Double, at timestamp: TimeInterval) -> CollisionSuspicion? {
    guard accelMagnitudeMs2 >= Self.thresholdMs2 else {
      return nil
    }
    guard timestamp - lastTriggerAt >= suppressionWindowSeconds else {
      return nil
    }
    lastTriggerAt = timestamp
    return CollisionSuspicion(
      id: UUID().uuidString,
      accelMagnitudeMs2: accelMagnitudeMs2,
      score: accelMagnitudeMs2 / Self.thresholdMs2,
      onsetEstimate: timestamp,
      modelVersion: Self.modelVersion
    )
  }
}
