# Class diagram set: package notes and database questions

This note goes with the draw.io files in `docs/diagrams/`:

- `class_00_package_overview.drawio.xml`: the connecting package diagram (consistent with `figure_4_1_architecture.png`, the component diagram, and the node/deployment diagram: same package names, same colour coding, same HTTPS boundaries).
- `class_01_acquisition.drawio.xml`: iOS, Acquisition
- `class_02_perception.drawio.xml`: iOS, Driver and Road Perception
- `class_03_orchestration.drawio.xml`: iOS, Orchestration
- `class_04_risk_collision.drawio.xml`: iOS, Risk Assessment and Collision Safety
- `class_05_app_networking.drawio.xml`: iOS, App and Networking
- `class_06_backend_controller_auth.drawio.xml`: Backend, Controller and Auth
- `class_07_backend_persistence_notifications.drawio.xml`: Backend, Persistence and Notifications

Each class box is kept at planning level (stereotype plus name plus the
handful of attributes/operations that identify the class), not a full
implementation dump, so every diagram prints on one page. Boxes and colours
match the Acquisition, Perception, Orchestration, Risk Assessment and
Collision Safety, App and Networking, and Server swimlanes already used in
the component diagram.

## Fixes applied from the professor's feedback

- **Broken into packages, with proper names**, one diagram per package plus
  one connecting package diagram (`class_00_...`), instead of one dense
  diagram.
- **`api.main` is modeled as a controller**, not "main": it appears as
  `RoadSenseController («controller», api.main)` whose only job is
  `create_app(...)` — composing the FastAPI app, wiring the router, and
  bootstrapping the administrator account. `api.routes_app` is modeled
  separately as `AppRouter («boundary», api.routes_app)` — the REST boundary
  that turns each HTTP request into a call on `AuthService`, `SessionRecorder`,
  `Database`, or `NotificationSender`.
- **Arrow semantics were checked and corrected**:
  - *Composition* (filled diamond) only where the owner creates/destroys the
    part and the part has no life outside it — e.g. `BackgroundCaptureService`
    composes its three BLE clients and `VisionManager`; `AuthManager`
    composes `RoadSenseAPI`.
  - *Aggregation* (hollow diamond) for 0..* collections whose members are
    independently meaningful — `CollisionIncidentStore o-- StoredIncident`,
    `TemporalTracker o-- TrackedObject`, `UserProfile o-- KitDevice`.
  - *Dependency* (dashed, open arrowhead) for "calls/uses, doesn't hold a
    reference" — e.g. `VisionManager --> DMDAttentionModel`,
    `SessionRecorder --> collision_detector.check(frame)`.
  - *Realization* (dashed, hollow triangle) reserved for actual protocol
    conformance — `MockNotificationSender ..|> NotificationSender`, and (in
    the connecting diagram) `BackgroundCaptureService` conforming to
    `ObdDelegate` / `WhoopDelegate` / `AmazfitDelegate`.

## How many databases, how many tables?

**One** database: a single SQLite file, opened and migrated by the
`Database` class in `accident_risk.api.db` (`src/accident_risk/api/db.py`).
There is no second database anywhere in the system — the iOS app has no
local SQL store of its own; `CollisionIncidentStore` and
`DriverCalibrationStore` on-device persist to `UserDefaults`, not a
database.

**14 tables**, all created by the same `Database._initialise()` schema:

`users`, `devices`, `sessions`, `session_points`, `session_gaps`,
`session_samples`, `alerts`, `family_links`, `family_notifications`,
`camera_events`, `settings`, `auth_codes`, `tokens`, `access_log`.

## How does the system decide which table to read or write?

There is no generic/dynamic table dispatcher. The decision is made in two
fixed steps, both resolved at request time, never by a table-name parameter:

1. **FastAPI's router** matches the incoming HTTP method + path to exactly
   one endpoint function inside `AppRouter` / `create_router()`
   (`api/routes_app.py`) — e.g. `POST /driver/session/start`,
   `GET /sessions/{session_id}`.
2. That endpoint function calls **one specific, hard-coded method** on the
   single `Database` instance. Each `Database` method issues one hard-coded
   SQL statement (or a small, fixed join) against one table. Nothing in the
   request body picks the table.

## Which function inserts, which function retrieves? (by table)

| Table | Insert | Retrieve |
|---|---|---|
| `users` | `create_user()` | `get_user()`, `get_user_by_phone()`, `list_users()` |
| `devices` | `create_device()` | `list_devices()`, `kit_for_user()` |
| `sessions` | `start_session()` | `get_session()`, `list_sessions()`, `active_session()` |
| `session_points` | `add_point()` | `session_points()` |
| `session_gaps` | *none* — table exists in the schema and is read, but nothing currently inserts into it (a leftover from an earlier gap-tracking feature) | `session_gaps()` |
| `session_samples` | `add_sample()` | `session_samples()` |
| `alerts` | `create_alert()`, `get_or_create_collision_alert()` | `get_alert_by_client_incident_id()`, `list_alerts()` |
| `family_links` | `request_family_link()` | `list_family_links()`, `accepted_family_members_to_notify()` |
| `family_notifications` | `record_family_notification()` | `list_family_notifications()`, `already_notified()` |
| `settings` | `update_settings()` | `get_settings()` |
| `access_log` | `log_access()` | `recent_access()` |
| `auth_codes` / `tokens` | `AuthService.request_code()` / `verify_code()` — these two tables are the one exception: `AuthService` (`api/auth.py`) runs its own `INSERT`/`DELETE` SQL directly over a `Database.connect()` connection instead of going through a `Database` wrapper method | `AuthService.resolve(token)` (reads `tokens` joined with `users`) |

`SessionRecorder` (`api/recorder.py`) sits in front of several of these
(`observe()`, `check_collision()`, `report_collision()`, `summarise()`,
`finish()`) and calls the named `Database` methods above — it never talks
to SQLite directly. `AuthService` is the only component that bypasses
`Database`'s method wrappers and writes raw SQL itself.
