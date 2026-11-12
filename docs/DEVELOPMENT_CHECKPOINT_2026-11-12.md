# RoadSense development checkpoint — 2026-11-12

This is a development checkpoint, not a finished product or validated safety system.
Branch: `final-development-review`; push target: `origin/final-development-review` only.
The pre-checkpoint local and remote HEAD was `ebcb922da92f4370454f818c4cb059905d467a51`.

## Preserved milestones

- Future-risk research rounds 1–3, scripts and the limitations/audit document were
  already committed on the remote branch. See `FUTURE_RISK_RESEARCH_SUMMARY.md`.
  No future-risk research model is deployed by this checkpoint.
- Research recording, consent-gated CSV/JSON export and authentication fixes were
  already committed. Research consent is separate from family contact consent.
- This checkpoint adds independent acceleration-threshold collision detection in
  Python and iOS, local iOS incident storage and confirmation UI, queued upload and
  decision synchronization, server incident states, and synthetic regression tests.
- Family monitoring adds invite/accept/revoke and independent notification/location
  permissions, notification records and acknowledgements. Delivery is mock-only:
  no SMS, push or email provider is connected.

No runtime implementation was changed during checkpoint preparation. The new code
was reviewed as a development snapshot; the defects below remain in that snapshot.
Acceleration is in m/s²; incident timestamps are Unix seconds UTC. The trigger score
is an acceleration-to-threshold ratio, not an accident probability.

## Verification

- `.venv/bin/python -m pytest -q`: **117 passed, 67 warnings in 7.00 seconds**.
  Warnings concern Starlette/AnyIO and SciPy optimizer deprecations.
- Xcode 27.2 (27B52), Debug, generic iOS Simulator, signing disabled: **BUILD SUCCEEDED**.
  Existing VisionManager Sendable/downcast warnings and skipped App Intents metadata
  extraction were reported. This was compilation, not an iOS runtime/UI test.
- Initial sandboxed build failed because Core ML compilation was denied by the
  sandbox; the same build succeeded with authorized access outside the sandbox.
- Build command:

  ```sh
  xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
    -configuration Debug -sdk iphonesimulator \
    -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath /private/tmp/roadsense-checkpoint-20261112-build \
    CODE_SIGNING_ALLOWED=NO build
  ```

- Build log: `/private/tmp/roadsense-checkpoint-20261112-build-unrestricted.log`.
- No Android changes. No dataset downloads or changes to raw data were performed.
- Synthetic tests verify software behavior only, not collision-detection accuracy.

## Exact checkpoint file list

```text
docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md
ios/RoadSense.xcodeproj/project.pbxproj
ios/RoadSense/App/ContentView.swift
ios/RoadSense/Services/BackgroundCaptureService.swift
ios/RoadSense/Services/CollisionDetector.swift
ios/RoadSense/Services/CollisionIncidentStore.swift
src/accident_risk/api/db.py
src/accident_risk/api/main.py
src/accident_risk/api/notifications.py
src/accident_risk/api/recorder.py
src/accident_risk/api/routes_app.py
src/accident_risk/processing/collision_detector.py
src/accident_risk/schemas.py
src/accident_risk/sources/recording.py
tests/test_app_api.py
tests/test_processing.py
```

## Intentionally excluded

- `scripts/download_dmd.sh`: contains signed dataset access URLs; left intact and
  untracked. Do not add it to Git in its current form.
- Ignored `.env`, `.claude/`, Android local settings/build files, caches, local
  recordings, `data/raw/`, generated `data/processed/` directories and `artifacts/`.
- Existing tracked `data/app.sqlite3` and bundled Core ML packages are unchanged;
  no database, participant data, generated dataset or model artifact is added by
  this checkpoint. Their presence in earlier history is not corrected here.

## Exact resume tasks, in priority order

1. Fix `BackgroundCaptureService.syncDecision`: the UI confirm/dismiss callbacks
   can reach `DispatchQueue.main.sync` from the main queue after an upload has
   supplied `serverAlertId`. Use actor-safe asynchronous access and add an iOS
   regression test for both buttons before and after upload, including offline use.
2. Add an idempotent migration for existing SQLite `alerts` tables lacking
   `detail_json`. `CREATE TABLE IF NOT EXISTS` alone does not add the new column.
   Test upgrade from the previous schema with synthetic rows and preservation of
   existing alerts; do not migrate the real database as part of a test.
3. Persist and scope incident deduplication in the database. The current UUID cache
   is process-local and lookup/create is not atomic. Test restart, concurrent retry,
   different drivers and streams, and offline uploads after the original session
   has ended or the server has restarted. Verify ownership/authentication of ingest.
4. Enforce automatic-incident lifecycle rules for confirm/cancel as well as
   dismiss/resolve. Test that resolved/dismissed incidents cannot be reopened and
   distinguish manager confirmation from driver confirmation in notification text.
5. Complete iOS lifecycle tests: local persistence/relaunch, retry timer behavior,
   multiple pending incidents, account changes, missing sensors and sensor cadence.
   A successful build does not establish that these paths work on a device.
6. Complete family monitoring UI and a real notification provider only in a future
   authorized feature session. Add reliable timeout scheduling (currently triggered
   by API reads), atomic notification delivery/retry handling, and acknowledgement
   token delivery and contact-controlled revocation. Current delivery remains mock.
7. Verify research consent/export UI and end-to-end privacy behavior; validate real
   wearable/vehicle integration and collection while parked. Revisit future-risk
   research only after reading its existing audit: event-aware splits, continuous
   replay, distribution shift, false alarms and calibration remain unresolved.
8. After fixes, rerun the full Python suite, iOS build and device/UI regression
   checks. Make no product-completion or model-performance claims from these tests.
