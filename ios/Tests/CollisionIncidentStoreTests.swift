import XCTest
@testable import RoadSenseAuth

final class CollisionIncidentStoreTests: XCTestCase {
    func makeStore() -> CollisionIncidentStore {
        CollisionIncidentStore(defaults: UserDefaults(suiteName: "roadsense-incident-test-\(UUID().uuidString)")!)
    }

    func suspicion(id: String = UUID().uuidString) -> CollisionSuspicion {
        CollisionSuspicion(id: id, accelMagnitudeMs2: 30.0, score: 1.2, onsetEstimate: 1000.0, modelVersion: "test")
    }

    func testConfirmedAndUploadedIncidentIsPendingSyncUntilMarkedSynced() {
        let store = makeStore()
        let incident = store.record(suspicion(), streamId: "s1", accelX: 1, accelY: 2, accelZ: 3)
        store.markUploaded(id: incident.id, serverAlertId: 42)
        store.confirm(id: incident.id)
        XCTAssertEqual(store.pendingDecisionSync().map(\.id), [incident.id])
        store.markDecisionSynced(id: incident.id)
        XCTAssertTrue(store.pendingDecisionSync().isEmpty)
    }

    /// The actual bug fixed this session: checkpoint resume task #4 made
    /// the server correctly reject a confirm/dismiss on an incident
    /// already resolved by someone else (HTTP 409) -- without
    /// syncRejected, that rejection would retry every cycle forever.
    func testRejectedDecisionStopsBeingPendingWithoutBeingMarkedSynced() {
        let store = makeStore()
        let incident = store.record(suspicion(), streamId: "s1", accelX: 1, accelY: 2, accelZ: 3)
        store.markUploaded(id: incident.id, serverAlertId: 42)
        store.confirm(id: incident.id)
        XCTAssertEqual(store.pendingDecisionSync().count, 1)

        store.markDecisionRejected(id: incident.id)

        XCTAssertTrue(store.pendingDecisionSync().isEmpty, "a rejected decision must stop retrying")
        let reloaded = store.incident(id: incident.id)
        XCTAssertEqual(reloaded?.syncRejected, true)
        XCTAssertNotEqual(reloaded?.syncedDecision, true, "rejected is NOT the same claim as successfully synced")
    }

    func testIncidentNotYetUploadedIsNeverPendingDecisionSyncEvenIfConfirmed() {
        let store = makeStore()
        let incident = store.record(suspicion(), streamId: "s1", accelX: 1, accelY: 2, accelZ: 3)
        store.confirm(id: incident.id)
        // Confirmed locally, but never uploaded -- there's no serverAlertId
        // to sync a decision against yet; this is pendingUpload's job, not
        // pendingDecisionSync's.
        XCTAssertTrue(store.pendingDecisionSync().isEmpty)
        XCTAssertEqual(store.pendingUpload().map(\.id), [incident.id])
    }

    func testUnresolvedSuspectedOnlyIncludesStillSuspectedIncidents() {
        let store = makeStore()
        let suspected = store.record(suspicion(), streamId: "s1", accelX: 1, accelY: 2, accelZ: 3)
        let dismissed = store.record(suspicion(), streamId: "s1", accelX: 1, accelY: 2, accelZ: 3)
        store.dismiss(id: dismissed.id)
        XCTAssertEqual(store.unresolvedSuspected().map(\.id), [suspected.id])
    }

    func testDecodingAnIncidentSavedBeforeSyncRejectedExistedStillSucceeds() {
        // Simulates an app upgrade: an incident array encoded by an OLDER
        // version of StoredIncident (no syncRejected key at all) must
        // still decode successfully, not silently drop every stored
        // incident -- the real risk of adding a new non-optional field
        // with only a default value (Codable does not honor those).
        let legacyJSON = """
        [{"id":"abc","streamId":"s1","accelMagnitudeMs2":30.0,"accelX":1,"accelY":2,"accelZ":3,
          "onsetEstimate":1000.0,"modelVersion":"test","status":"suspected","detectedAt":1000.0,
          "uploaded":false,"syncedDecision":false}]
        """
        let decoded = try? JSONDecoder().decode([StoredIncident].self, from: Data(legacyJSON.utf8))
        XCTAssertNotNil(decoded)
        XCTAssertEqual(decoded?.count, 1)
        XCTAssertNil(decoded?.first?.syncRejected)
    }
}
