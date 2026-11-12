// RoadSense/Services/CollisionIncidentStore.swift
import Foundation

/// Local-only lifecycle for one suspected collision. Independent of (but
/// eventually linked to, via serverAlertId) the server's own richer
/// lifecycle in accident_risk/api/db.py's INCIDENT_LIFECYCLE.
public enum LocalIncidentStatus: String, Codable {
  case suspected
  case confirmed
  case dismissed
}

public struct StoredIncident: Codable, Identifiable, Equatable {
  public let id: String  // == CollisionSuspicion.id; the one stable thread tying detection, storage, upload, and the server's own record together.
  public var streamId: String
  public var accelMagnitudeMs2: Double
  public var accelX: Double
  public var accelY: Double
  public var accelZ: Double
  public var onsetEstimate: TimeInterval
  public var modelVersion: String
  public var status: LocalIncidentStatus
  public var detectedAt: TimeInterval
  public var confirmedAt: TimeInterval?
  public var dismissedAt: TimeInterval?
  public var serverAlertId: Int?
  public var uploaded: Bool
  public var syncedDecision: Bool
  /// True once the server has definitively REJECTED this local decision
  /// (HTTP 409 -- an invalid lifecycle transition, e.g. someone else
  /// already dismissed/resolved it) rather than merely failed to respond.
  /// Without this, `pendingDecisionSync()` retried a permanently-rejected
  /// sync forever, every 10s, for the rest of the app session -- a real
  /// consequence of checkpoint resume task #4 making the server correctly
  /// reject those transitions. Optional (not a plain Bool) so decoding an
  /// incident saved before this field existed doesn't fail outright and
  /// silently drop every previously stored incident -- Codable does not
  /// apply a stored property's default value when a key is simply absent.
  public var syncRejected: Bool?
}

/// The phone must never depend on the server to preserve an accident event:
/// every suspected incident is written here FIRST, synchronously, before
/// any network attempt. Reuses UserDefaults -- the app's own existing
/// persistence mechanism (see AuthManager's token storage) -- as a single
/// JSON-encoded array. No new local database/framework: at this scale (a
/// handful of incidents per device, not thousands) that would be
/// unnecessary complexity for a university capstone.
public final class CollisionIncidentStore {
  public static let shared = CollisionIncidentStore()
  private let defaultsKey = "roadsense_collision_incidents"
  private let queue = DispatchQueue(label: "ae.aus.roadsense.incidentStore")
  private let defaults: UserDefaults

  public init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
  }

  @discardableResult
  public func record(
    _ suspicion: CollisionSuspicion, streamId: String, accelX: Double, accelY: Double, accelZ: Double
  ) -> StoredIncident {
    let incident = StoredIncident(
      id: suspicion.id, streamId: streamId, accelMagnitudeMs2: suspicion.accelMagnitudeMs2,
      accelX: accelX, accelY: accelY, accelZ: accelZ,
      onsetEstimate: suspicion.onsetEstimate, modelVersion: suspicion.modelVersion,
      status: .suspected, detectedAt: suspicion.onsetEstimate,
      confirmedAt: nil, dismissedAt: nil, serverAlertId: nil,
      uploaded: false, syncedDecision: false, syncRejected: false
    )
    queue.sync {
      var all = loadAllLocked()
      all.append(incident)
      saveAllLocked(all)
    }
    return incident
  }

  public func all() -> [StoredIncident] {
    queue.sync { loadAllLocked() }
  }

  public func incident(id: String) -> StoredIncident? {
    all().first { $0.id == id }
  }

  /// Still-suspected incidents the driver hasn't acted on -- used on launch
  /// to re-present the confirm/dismiss prompt for a crash the app never
  /// finished showing (e.g. the app was killed), WITHOUT creating a new
  /// detection or showing it again after it's actually been addressed.
  public func unresolvedSuspected() -> [StoredIncident] {
    all().filter { $0.status == .suspected }
  }

  @discardableResult
  public func confirm(id: String) -> StoredIncident? {
    mutate(id: id) {
      $0.status = .confirmed
      $0.confirmedAt = Date().timeIntervalSince1970
      $0.syncedDecision = false
    }
  }

  @discardableResult
  public func dismiss(id: String) -> StoredIncident? {
    mutate(id: id) {
      $0.status = .dismissed
      $0.dismissedAt = Date().timeIntervalSince1970
      $0.syncedDecision = false
    }
  }

  public func markUploaded(id: String, serverAlertId: Int) {
    _ = mutate(id: id) {
      $0.uploaded = true
      $0.serverAlertId = serverAlertId
    }
  }

  public func markDecisionSynced(id: String) {
    _ = mutate(id: id) { $0.syncedDecision = true }
  }

  /// The server definitively rejected this decision (HTTP 409 -- an
  /// invalid lifecycle transition). Distinct from "not yet synced":
  /// retrying this is pointless, since the rejection is about the
  /// server's CURRENT state (e.g. another party already
  /// dismissed/resolved it), not a transient failure.
  public func markDecisionRejected(id: String) {
    _ = mutate(id: id) { $0.syncRejected = true }
  }

  /// Detections that haven't reached the server yet -- retry these.
  public func pendingUpload() -> [StoredIncident] {
    all().filter { !$0.uploaded }
  }

  /// Uploaded incidents whose confirm/dismiss decision (made locally,
  /// possibly while offline) hasn't been told to the server yet. Excludes
  /// anything already marked syncRejected -- without that, a permanently
  /// rejected decision (checkpoint resume task #4's lifecycle enforcement
  /// correctly saying no) retried forever, every cycle, for the rest of
  /// the app session.
  public func pendingDecisionSync() -> [StoredIncident] {
    all().filter {
      $0.uploaded && $0.status != .suspected && !$0.syncedDecision && $0.syncRejected != true
    }
  }

  @discardableResult
  private func mutate(id: String, _ change: (inout StoredIncident) -> Void) -> StoredIncident? {
    queue.sync {
      var all = loadAllLocked()
      guard let index = all.firstIndex(where: { $0.id == id }) else {
        return nil
      }
      change(&all[index])
      saveAllLocked(all)
      return all[index]
    }
  }

  private func loadAllLocked() -> [StoredIncident] {
    guard let data = defaults.data(forKey: defaultsKey),
      let decoded = try? JSONDecoder().decode([StoredIncident].self, from: data)
    else {
      return []
    }
    return decoded
  }

  private func saveAllLocked(_ incidents: [StoredIncident]) {
    guard let data = try? JSONEncoder().encode(incidents) else {
      return
    }
    defaults.set(data, forKey: defaultsKey)
  }
}
