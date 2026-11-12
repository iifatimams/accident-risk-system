import Foundation

/// Phase 5: lightweight temporal object tracking over whatever per-frame
/// detections the rear camera pipeline currently produces. Deliberately
/// detector-agnostic -- it takes a flat list of (class, box, confidence)
/// per frame and does not care whether those came from the still-crude
/// rectangle heuristic in `BackgroundCaptureService` or a future validated
/// detector (Phase 4.4). Pure Foundation, no Vision/AVFoundation
/// dependency, so it is unit-testable on any platform.
///
/// KNOWN LIMITATION this does NOT fix: object permanence and consistent
/// IDs only help if the underlying per-frame detections are themselves
/// trustworthy. Feeding this a detector that misidentifies what a "car" is
/// will track a consistently-wrong thing consistently -- see
/// ForwardCollisionHazard.swift and BackgroundCaptureService.swift for how
/// this is kept epistemically cautious given that.

/// A box in normalized frame coordinates (0...1), origin and axis
/// convention matching whatever the caller's detection source uses --
/// this type does no coordinate-system interpretation of its own.
public struct NormalizedRect: Equatable {
  public var x: Double
  public var y: Double
  public var width: Double
  public var height: Double

  public init(x: Double, y: Double, width: Double, height: Double) {
    self.x = x
    self.y = y
    self.width = width
    self.height = height
  }

  public var area: Double { max(width, 0) * max(height, 0) }
  public var centerX: Double { x + width / 2 }
  public var centerY: Double { y + height / 2 }

  /// Intersection-over-union, same definition used to grade the BDD100K
  /// detector benchmarks (scripts/benchmark_detector_bdd100k*.py) -- kept
  /// consistent so "how well two boxes overlap" means the same thing
  /// everywhere in this project.
  public static func iou(_ a: NormalizedRect, _ b: NormalizedRect) -> Double {
    let ix1 = max(a.x, b.x)
    let iy1 = max(a.y, b.y)
    let ix2 = min(a.x + a.width, b.x + b.width)
    let iy2 = min(a.y + a.height, b.y + b.height)
    let intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    let union = a.area + b.area - intersection
    return union > 0 ? intersection / union : 0
  }
}

/// One frame's detection, before any track identity is assigned.
public struct Detection: Equatable {
  public var classLabel: String
  public var box: NormalizedRect
  public var confidence: Float

  public init(classLabel: String, box: NormalizedRect, confidence: Float) {
    self.classLabel = classLabel
    self.box = box
    self.confidence = confidence
  }
}

/// One historical box observation for a track, used by
/// ForwardCollisionHazard to measure apparent-size change over time.
public struct BoxObservation: Equatable {
  public var timestamp: TimeInterval
  public var box: NormalizedRect

  public init(timestamp: TimeInterval, box: NormalizedRect) {
    self.timestamp = timestamp
    self.box = box
  }
}

/// A detection with persistent identity across frames.
public struct TrackedObject: Equatable {
  public let id: UUID
  public var classLabel: String
  public var box: NormalizedRect
  public var confidence: Float
  public var firstSeen: TimeInterval
  public var lastSeen: TimeInterval
  /// Consecutive frames since the last matched detection. Reset to 0 on
  /// every match; the track is dropped once this exceeds
  /// ObjectTracker.maxMissedFrames, not on the very first miss -- a single
  /// dropped detection (motion blur, brief occlusion) must not fragment one
  /// real object into two different track ids.
  public var missedFrames: Int
  /// Bounded to ObjectTracker.maxHistory, oldest first.
  public var history: [BoxObservation]
}

/// Greedy per-class IoU matching between existing tracks and this frame's
/// detections -- the same matching strategy already used (and already
/// documented as a deliberate simplification, not full multi-object
/// tracking) in the BDD100K benchmark scripts, applied here across time
/// instead of across one prediction/ground-truth pair.
public struct ObjectTracker {
  public var maxMissedFrames: Int
  public var maxHistory: Int
  public var iouMatchThreshold: Double
  private var tracks: [TrackedObject] = []

  public init(maxMissedFrames: Int = 5, maxHistory: Int = 10, iouMatchThreshold: Double = 0.3) {
    self.maxMissedFrames = maxMissedFrames
    self.maxHistory = maxHistory
    self.iouMatchThreshold = iouMatchThreshold
  }

  public var activeTracks: [TrackedObject] { tracks }

  @discardableResult
  public mutating func update(detections: [Detection], at timestamp: TimeInterval) -> [TrackedObject] {
    var unmatched = Set(detections.indices)
    var next: [TrackedObject] = []

    for track in tracks {
      var bestIoU = 0.0
      var bestIndex: Int?
      for index in unmatched where detections[index].classLabel == track.classLabel {
        let overlap = NormalizedRect.iou(track.box, detections[index].box)
        if overlap > bestIoU {
          bestIoU = overlap
          bestIndex = index
        }
      }
      if let index = bestIndex, bestIoU >= iouMatchThreshold {
        var updated = track
        let detection = detections[index]
        updated.box = detection.box
        updated.confidence = detection.confidence
        updated.lastSeen = timestamp
        updated.missedFrames = 0
        updated.history.append(BoxObservation(timestamp: timestamp, box: detection.box))
        if updated.history.count > maxHistory {
          updated.history.removeFirst(updated.history.count - maxHistory)
        }
        next.append(updated)
        unmatched.remove(index)
      } else {
        var aged = track
        aged.missedFrames += 1
        if aged.missedFrames <= maxMissedFrames {
          next.append(aged)
        }
        // else: dropped -- a genuinely new track forms if the object (or a
        // different one) reappears later, rather than resurrecting a stale
        // identity after a long gap.
      }
    }

    for index in unmatched.sorted() {
      let detection = detections[index]
      next.append(
        TrackedObject(
          id: UUID(), classLabel: detection.classLabel, box: detection.box,
          confidence: detection.confidence, firstSeen: timestamp, lastSeen: timestamp,
          missedFrames: 0, history: [BoxObservation(timestamp: timestamp, box: detection.box)]
        )
      )
    }

    tracks = next
    return tracks
  }
}
