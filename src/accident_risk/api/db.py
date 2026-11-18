"""SQLite persistence for users, kits, sessions and alerts. Standard library only: no new dependencies.
Every call opens a short-lived connection so the store is safe to use from FastAPI's thread pool.
Timestamps are Unix seconds UTC, matching ``SensorFrame.timestamp``.
"""

from __future__ import annotations
import json
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROLES: tuple[str, ...] = ("driver", "administrator")
# Higher rank inherits every capability of the ranks below it.
ROLE_RANK: dict[str, int] = {"driver": 0, "administrator": 1}
DEVICE_KINDS: tuple[str, ...] = ("adapter", "wearable")
DEVICE_STATUSES: tuple[str, ...] = ("active", "spare", "out_of_service")
ALERT_KINDS: tuple[str, ...] = ("high_risk", "collision")
# "pending"/"confirmed"/"cancelled" remain exactly as before for high_risk
# alerts. An automatically DETECTED collision (collision_detector.py) instead uses the richer
# lifecycle: suspected -> awaiting_confirmation -> {confirmed, dismissed,
# unconfirmed_timeout} -> resolved. Both vocabularies coexist in this same
# table/kind so a dashboard can list all collision-kind alerts together.
ALERT_STATUSES: tuple[str, ...] = (
    "pending", "confirmed", "cancelled",
    "suspected", "awaiting_confirmation", "dismissed", "unconfirmed_timeout", "resolved",
)
INCIDENT_LIFECYCLE: dict[str, tuple[str, ...]] = {
    "suspected": ("awaiting_confirmation",),
    "awaiting_confirmation": ("confirmed", "dismissed", "unconfirmed_timeout"),
    "confirmed": ("resolved",),
    "dismissed": ("resolved",),
    "unconfirmed_timeout": ("resolved",),
    "resolved": (),
}
FAMILY_LINK_STATUSES: tuple[str, ...] = ("requested", "accepted", "declined")

DEFAULT_SETTINGS: dict[str, float] = {
    "risk_medium_threshold": 0.40,
    "risk_high_threshold": 0.70,
    "risk_critical_threshold": 0.85,
    "session_gap_tolerance_seconds": 10.0,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    phone         TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    role          TEXT    NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL,
    identifier    TEXT    NOT NULL UNIQUE,
    status        TEXT    NOT NULL DEFAULT 'spare',
    assigned_to   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    note          TEXT    NOT NULL DEFAULT '',
    created_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id     TEXT    NOT NULL UNIQUE,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at    REAL    NOT NULL,
    ended_at      REAL,
    end_reason    TEXT,
    complete      INTEGER NOT NULL DEFAULT 1,
    summary       TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS session_points (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts            REAL    NOT NULL,
    risk_index    REAL,
    risk_level    TEXT,
    speed_kmh     REAL,
    rpm           REAL,
    engine_load   REAL,
    throttle      REAL,
    heart_rate    REAL,
    latitude      REAL,
    longitude     REAL
);
CREATE INDEX IF NOT EXISTS idx_points_session ON session_points(session_id, ts);

CREATE TABLE IF NOT EXISTS session_gaps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source        TEXT    NOT NULL,
    started_at    REAL    NOT NULL,
    ended_at      REAL
);

-- Full per-frame record, alongside (not replacing) the throttled session_points
-- used by the live map/history UI. Same 1Hz cadence as session_points -- raw
-- per-frame data still does not belong in SQLite. Everything beyond the few
-- queryable columns (ts, device, source) lives in payload_json so adding a new
-- sensor/derived field never requires a migration. Unconditional -- every
-- session gets this, there is no consent gate on it.
CREATE TABLE IF NOT EXISTS session_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts            REAL    NOT NULL,
    device_id     TEXT,
    source        TEXT    NOT NULL,
    payload_json  TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_samples_session ON session_samples(session_id, ts);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind          TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    created_at    REAL    NOT NULL,
    risk_index    REAL,
    latitude      REAL,
    longitude     REAL,
    acknowledged_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at REAL,
    -- Independent collision-detection lifecycle fields (model_version,
    -- onset_estimate, detection_score, source: "automatic_detector").
    -- Kept here, not a separate table, since every
    -- collision incident already IS an alert row with kind='collision';
    -- JSON so new fields never need a migration.
    detail_json   TEXT    NOT NULL DEFAULT '{}',
    -- Globally-unique id the on-device detector generates once per detected
    -- event (see CollisionDetector.swift). The durable dedup key for a
    -- retried/offline-queued re-upload of the SAME event: NULL for every
    -- high_risk alert and manual collision report, which never collide
    -- with each other (SQLite treats every NULL as distinct under UNIQUE).
    client_incident_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
-- idx_alerts_client_incident_id is created in Database._migrate(), not here:
-- it must run AFTER the column-backfill migration below, since a database
-- created before client_incident_id existed would otherwise fail this
-- CREATE INDEX with "no such column" before _migrate ever runs.

-- Mutual, symmetric family linking between two real driver accounts --
-- replaces the earlier one-directional, no-login family_contacts design.
-- Both requester and recipient are users() rows; a link only notifies/
-- shares in a given direction once BOTH that direction's own flag is set
-- AND the link is accepted, so each side controls what THEY share
-- independently of what the other side shares back.
CREATE TABLE IF NOT EXISTS family_links (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    requester_id                INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_id                INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status                      TEXT    NOT NULL DEFAULT 'requested',
    requester_notify_accidents  INTEGER NOT NULL DEFAULT 1,
    requester_share_location    INTEGER NOT NULL DEFAULT 0,
    recipient_notify_accidents  INTEGER NOT NULL DEFAULT 1,
    recipient_share_location    INTEGER NOT NULL DEFAULT 0,
    created_at                  REAL    NOT NULL,
    responded_at                REAL,
    UNIQUE(requester_id, recipient_id)
);
CREATE INDEX IF NOT EXISTS idx_family_links_requester ON family_links(requester_id);
CREATE INDEX IF NOT EXISTS idx_family_links_recipient ON family_links(recipient_id);

-- UNIQUE(alert_id, recipient_user_id) gives duplicate-notification
-- suppression for free: a second attempt to notify the same family member
-- about the same incident is simply ignored at insert time, not re-sent.
CREATE TABLE IF NOT EXISTS family_notifications (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id           INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    recipient_user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sent_at            REAL    NOT NULL,
    delivery_status    TEXT    NOT NULL,
    acknowledge_token  TEXT    NOT NULL UNIQUE,
    acknowledged_at    REAL,
    UNIQUE(alert_id, recipient_user_id)
);

CREATE TABLE IF NOT EXISTS camera_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts            REAL    NOT NULL,
    camera        TEXT    NOT NULL CHECK (camera IN ('front', 'rear')),
    label         TEXT    NOT NULL,
    confidence    REAL,
    note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_camera_events_session ON camera_events(session_id, ts);

CREATE TABLE IF NOT EXISTS settings (
    key           TEXT PRIMARY KEY,
    value         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_codes (
    phone         TEXT PRIMARY KEY,
    code_hash     TEXT NOT NULL,
    expires_at    REAL NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tokens (
    token_hash    TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS access_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER,
    role          TEXT,
    path          TEXT NOT NULL,
    ts            REAL NOT NULL
);
"""


class Database:
    """A SQLite store opened per operation; safe across FastAPI worker threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with foreign keys on, committing on clean exit."""
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        """Create the schema and seed default thresholds exactly once."""
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            self._migrate(connection)
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value)
                )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Idempotent column migrations for databases created before a column
        existed. `CREATE TABLE IF NOT EXISTS` in SCHEMA above does nothing to
        an alerts table that already exists without detail_json -- added
        later, after the independent collision-detector lifecycle needed
        somewhere to store its fields. Safe to run on every startup: a
        database already at the current schema sees no missing columns and
        this is a no-op."""
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(alerts)")
        }
        if "detail_json" not in existing_columns:
            connection.execute(
                "ALTER TABLE alerts ADD COLUMN detail_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "client_incident_id" not in existing_columns:
            connection.execute("ALTER TABLE alerts ADD COLUMN client_incident_id TEXT")
        # Created here (not in SCHEMA) so it always runs AFTER the column
        # exists, whether this is a brand-new database (column just added by
        # CREATE TABLE above) or one migrated by the ALTER TABLE just above.
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_client_incident_id "
            "ON alerts(client_incident_id)"
        )

    # ---------------------------------------------------------------- users

    def create_user(self, phone: str, name: str, role: str) -> dict[str, Any]:
        """Register a user. Raises ValueError on a bad role or duplicate phone."""
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role}")
        phone = normalise_phone(phone)
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO users(phone, name, role, created_at) VALUES (?, ?, ?, ?)",
                    (phone, name.strip(), role, time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A user with that phone number already exists") from exc
            return self._row(connection, "users", int(cursor.lastrowid))

    def get_user_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Look up an active or inactive user by normalised phone number."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE phone = ?", (normalise_phone(phone),)
            ).fetchone()
            return dict(row) if row else None

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Fetch one user by primary key."""
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def list_users(self, role: str | None = None) -> list[dict[str, Any]]:
        """List users, newest first, optionally filtered to one role."""
        query = "SELECT * FROM users"
        params: tuple[Any, ...] = ()
        if role:
            query += " WHERE role = ?"
            params = (role,)
        query += " ORDER BY id DESC"
        with self.connect() as connection:
            return [dict(r) for r in connection.execute(query, params)]

    def update_user(
        self, user_id: int, *, name: str | None = None, role: str | None = None,
        active: bool | None = None,
    ) -> dict[str, Any] | None:
        """Change a user's name, role or active flag. Returns the updated row."""
        if role is not None and role not in ROLES:
            raise ValueError(f"Unknown role: {role}")

        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name.strip()
        if role is not None:
            updates["role"] = role
        if active is not None:
            updates["active"] = int(active)

        if not updates:
            return self.get_user(user_id)

        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE users SET {assignments} WHERE id = ?", (*updates.values(), user_id)
            )
            return self._row(connection, "users", user_id)

    # -------------------------------------------------------------- devices

    def create_device(self, kind: str, identifier: str, note: str = "") -> dict[str, Any]:
        """Add an adapter or wearable to the inventory as a spare."""
        if kind not in DEVICE_KINDS:
            raise ValueError(f"Unknown device kind: {kind}")
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO devices(kind, identifier, note, created_at) VALUES (?, ?, ?, ?)",
                    (kind, identifier.strip(), note.strip(), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A device with that identifier already exists") from exc
            return self._row(connection, "devices", int(cursor.lastrowid))

    def list_devices(self) -> list[dict[str, Any]]:
        """List every device with the name of its assignee, if any."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT d.*, u.name AS assigned_name FROM devices d "
                "LEFT JOIN users u ON u.id = d.assigned_to ORDER BY d.kind, d.identifier"
            )
            return [dict(r) for r in rows]

    def update_device(
        self, device_id: int, *, identifier: str | None = None, status: str | None = None, assigned_to: int | None = None,
        clear_assignment: bool = False,
    ) -> dict[str, Any] | None:
        """Change a device's identifier, status or assignment. Assigning also marks it active."""
        if status is not None and status not in DEVICE_STATUSES:
            raise ValueError(f"Unknown device status: {status}")

        updates: dict[str, Any] = {}
        if identifier is not None:
            updates["identifier"] = identifier.strip()
        if status is not None:
            updates["status"] = status

        if clear_assignment:
            updates["assigned_to"] = None
        elif assigned_to is not None:
            updates["assigned_to"] = assigned_to
            updates.setdefault("status", "active")

        if not updates:
            return self._fetch_one("SELECT * FROM devices WHERE id = ?", (device_id,))

        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as connection:
            try:
                connection.execute(
                    f"UPDATE devices SET {assignments} WHERE id = ?", (*updates.values(), device_id)
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A device with that identifier already exists") from exc
            return self._row(connection, "devices", device_id)

    def kit_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Return the devices currently assigned to one driver."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM devices WHERE assigned_to = ? ORDER BY kind", (user_id,)
            )
            return [dict(r) for r in rows]


    # ------------------------------------------------------------- sessions

    def start_session(self, user_id: int, stream_id: str, started_at: float | None = None) -> dict:
        """Open a session for a driver. One row per stream_id (FR13.2)."""
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO sessions(stream_id, user_id, started_at) VALUES (?, ?, ?)",
                    (stream_id, user_id, started_at if started_at is not None else time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("That stream_id already has a session") from exc
            return self._row(connection, "sessions", int(cursor.lastrowid))

    def active_session(self, user_id: int) -> dict[str, Any] | None:
        """Return the driver's open session, if one is running."""
        return self._fetch_one(
            "SELECT * FROM sessions WHERE user_id = ? AND ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (user_id,),
        )

    def end_session(
        self, session_id: int, reason: str = "driver_stopped", complete: bool = True
    ) -> dict[str, Any] | None:
        """Close a session, recording the end time and why it ended (FR13.6)."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ?, complete = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), reason, int(complete), session_id),
            )
            return self._row(connection, "sessions", session_id)

    def add_point(self, session_id: int, point: dict[str, Any]) -> None:
        """Append one downsampled track point used for history and the live map."""
        columns = (
            "ts", "risk_index", "risk_level", "speed_kmh", "rpm",
            "engine_load", "throttle", "heart_rate", "latitude", "longitude",
        )
        values = [point.get(name) for name in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO session_points(session_id, {', '.join(columns)}) "
                f"VALUES (?, {placeholders})",
                (session_id, *values),
            )

    def session_points(self, session_id: int, limit: int = 5000) -> list[dict[str, Any]]:
        """Return a session's track points in time order."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM session_points WHERE session_id = ? ORDER BY ts LIMIT ?",
                (session_id, limit),
            )
            return [dict(r) for r in rows]

    def session_gaps(self, session_id: int) -> list[dict[str, Any]]:
        """Return every recorded gap for a session."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM session_gaps WHERE session_id = ? ORDER BY started_at",
                (session_id,),
            )
            return [dict(r) for r in rows]

    def add_sample(
        self, session_id: int, ts: float, device_id: str | None, source: str, payload: dict[str, Any]
    ) -> None:
        """Append one full per-frame sample, unconditionally -- every input the
        pipeline has for this frame, not just what feeds the live risk score.

        payload holds every field beyond (session_id, ts, device_id, source) as
        JSON, so new sensor/derived fields never require a schema migration.
        """
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO session_samples(session_id, ts, device_id, source, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, ts, device_id, source, json.dumps(payload, separators=(",", ":"))),
            )

    def session_samples(self, session_id: int, limit: int = 20000) -> list[dict[str, Any]]:
        """Return a session's full per-frame samples in time order, payload inlined."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT ts, device_id, source, payload_json FROM session_samples "
                "WHERE session_id = ? ORDER BY ts LIMIT ?",
                (session_id, limit),
            )
            out = []
            for row in rows:
                record = {"ts": row["ts"], "device_id": row["device_id"], "source": row["source"]}
                record.update(json.loads(row["payload_json"]))
                out.append(record)
            return out

    def save_summary(self, session_id: int, summary: dict[str, Any]) -> None:
        """Store the computed session summary as JSON (FR13.7)."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (json.dumps(summary, separators=(",", ":")), session_id),
            )

    def list_sessions(
        self, *, user_id: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List sessions newest first, scoped to one driver when given (FR13.8)."""
        query = (
            "SELECT s.*, u.name AS driver_name, u.phone AS driver_phone "
            "FROM sessions s JOIN users u ON u.id = s.user_id"
        )
        params: list[Any] = []
        if user_id is not None:
            query += " WHERE s.user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY s.started_at DESC LIMIT ?"
        params.append(limit)
        
        with self.connect() as connection:
            rows = connection.execute(query, params)
            return [self._decode_session(dict(r)) for r in rows]

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        """Fetch one session with its driver's name attached."""
        row = self._fetch_one(
            "SELECT s.*, u.name AS driver_name, u.phone AS driver_phone "
            "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id = ?",
            (session_id,),
        )
        return self._decode_session(row) if row else None

    # --------------------------------------------------------------- alerts

    def get_alert_by_client_incident_id(self, client_incident_id: str) -> dict[str, Any] | None:
        """Read-only lookup for a retry that has no active session to attach
        a NEW incident to (e.g. arriving after the session ended, or after
        this process restarted) -- still must recognize an id it already
        knows about, even though it cannot create one without a session."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM alerts WHERE client_incident_id = ?", (client_incident_id,)
            ).fetchone()
            return self._decode_alert(dict(row)) if row else None

    def create_alert(
        self, user_id: int, kind: str, *, session_id: int | None = None,
        risk_index: float | None = None, latitude: float | None = None,
        longitude: float | None = None, status: str = "pending",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a high-risk or collision alert (FR9, FR11)."""
        if kind not in ALERT_KINDS:
            raise ValueError(f"Unknown alert kind: {kind}")
        if status not in ALERT_STATUSES:
            raise ValueError(f"Unknown alert status: {status}")

        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO alerts(session_id, user_id, kind, status, created_at, risk_index, "
                "latitude, longitude, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, user_id, kind, status, time.time(), risk_index, latitude, longitude,
                    json.dumps(detail or {}, separators=(",", ":")),
                ),
            )
            return self._decode_alert(self._row(connection, "alerts", int(cursor.lastrowid)))

    def get_or_create_collision_alert(
        self, client_incident_id: str, user_id: int, session_id: int | None,
        *, latitude: float | None = None, longitude: float | None = None,
        status: str = "suspected", detail: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically and durably deduplicate a collision incident by its
        client-generated id. Checkpoint resume task #3: the previous
        in-process `_incident_ids_seen` dict in SessionRecorder was lost on
        every server restart, and its separate check-then-create (one lock
        acquisition to check, a later one to record) was not atomic -- two
        concurrent retries of the SAME incident (the immediate report and a
        10s-later retry-timer upload both landing before either response
        returns) could each pass the "not seen" check and create two rows
        for one real event.

        A single INSERT ... ON CONFLICT DO NOTHING against the UNIQUE
        idx_alerts_client_incident_id index is both: SQLite's own writer
        serialization makes the check-and-insert atomic regardless of how
        many concurrent callers race here, and the dedup key lives in the
        database itself, so it survives a server restart -- a retry
        arriving after this process restarted still finds the original row.

        Returns (alert, created): `created` is False when an existing
        incident (from an earlier call, possibly a previous process) was
        returned instead of a new one being made, so callers can decide
        whether to run first-time-only side effects (e.g. the
        suspected -> awaiting_confirmation promotion) only when something
        was actually just created.
        """
        if status not in ALERT_STATUSES:
            raise ValueError(f"Unknown alert status: {status}")
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO alerts(session_id, user_id, kind, status, created_at, "
                "latitude, longitude, detail_json, client_incident_id) "
                "VALUES (?, ?, 'collision', ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_incident_id) DO NOTHING",
                (
                    session_id, user_id, status, time.time(), latitude, longitude,
                    json.dumps(detail or {}, separators=(",", ":")), client_incident_id,
                ),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT * FROM alerts WHERE client_incident_id = ?", (client_incident_id,)
                ).fetchone()
                return self._decode_alert(dict(row)), False
            return self._decode_alert(self._row(connection, "alerts", int(cursor.lastrowid))), True

    def set_alert_status(
        self, alert_id: int, status: str, acknowledged_by: int | None = None,
        *, enforce_lifecycle: bool = False,
    ) -> dict[str, Any] | None:
        """Confirm, cancel, dismiss, time out, or resolve an alert.

        enforce_lifecycle=True rejects a transition not listed in
        INCIDENT_LIFECYCLE for the alert's current status -- used for the
        richer automatic-detector states; the legacy pending/confirmed/
        cancelled path (high_risk alerts, manual collision reports) leaves
        this off, unchanged from before.
        """
        if status not in ALERT_STATUSES:
            raise ValueError(f"Unknown alert status: {status}")
        with self.connect() as connection:
            if enforce_lifecycle:
                current = connection.execute(
                    "SELECT status FROM alerts WHERE id = ?", (alert_id,)
                ).fetchone()
                if current is None:
                    return None
                allowed = INCIDENT_LIFECYCLE.get(current["status"], ())
                if status not in allowed:
                    raise ValueError(f"Cannot move {current['status']!r} -> {status!r}")
            connection.execute(
                "UPDATE alerts SET status = ?, acknowledged_by = ?, acknowledged_at = ? "
                "WHERE id = ?",
                (status, acknowledged_by, time.time() if acknowledged_by else None, alert_id),
            )
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            return self._decode_alert(dict(row)) if row else None

    def timeout_unconfirmed_incidents(self, older_than_seconds: float) -> list[dict[str, Any]]:
        """Move any awaiting_confirmation collision incident older than the
        window to unconfirmed_timeout. Call-on-access, same pattern as
        SessionRecorder.stale_streams -- no background scheduler needed."""
        cutoff = time.time() - older_than_seconds
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM alerts WHERE status = 'awaiting_confirmation' AND created_at < ?",
                (cutoff,),
            ).fetchall()
            timed_out = []
            for row in rows:
                connection.execute(
                    "UPDATE alerts SET status = 'unconfirmed_timeout' WHERE id = ?", (row["id"],)
                )
                updated = connection.execute(
                    "SELECT * FROM alerts WHERE id = ?", (row["id"],)
                ).fetchone()
                timed_out.append(self._decode_alert(dict(updated)))
            return timed_out

    def list_alerts(
        self,
        *,
        user_id: int | None = None,
        session_id: int | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List alerts newest first, scoped to one driver, session, and/or kind when given."""
        query = (
            "SELECT a.*, u.name AS driver_name FROM alerts a JOIN users u ON u.id = a.user_id"
        )
        clauses, params = [], []
        if user_id is not None:
            clauses.append("a.user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("a.session_id = ?")
            params.append(session_id)
        if kind is not None:
            clauses.append("a.kind = ?")
            params.append(kind)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)

        with self.connect() as connection:
            return [self._decode_alert(dict(r)) for r in connection.execute(query, params)]

    # ---------------------------------------------------------- family links

    def request_family_link(self, requester_id: int, recipient_id: int) -> dict[str, Any]:
        """Create a pending mutual-link request. Raises ValueError for a
        self-link or a link that already exists in either direction --
        callers should check list_family_links first if they want a
        friendlier error than a raised exception."""
        if requester_id == recipient_id:
            raise ValueError("Cannot send a family request to yourself")
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM family_links WHERE (requester_id = ? AND recipient_id = ?) "
                "OR (requester_id = ? AND recipient_id = ?)",
                (requester_id, recipient_id, recipient_id, requester_id),
            ).fetchone()
            if existing is not None:
                raise ValueError("A family link already exists between these two accounts")
            cursor = connection.execute(
                "INSERT INTO family_links(requester_id, recipient_id, status, created_at) "
                "VALUES (?, ?, 'requested', ?)",
                (requester_id, recipient_id, time.time()),
            )
            return self._decode_family_link(self._row(connection, "family_links", int(cursor.lastrowid)))

    def respond_to_family_link(
        self, link_id: int, recipient_id: int, accept: bool
    ) -> dict[str, Any] | None:
        """Only the RECIPIENT of the request can accept or decline it --
        checked here, not just at the route layer. Only a pending request
        can be responded to; an already-decided one is inert."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM family_links WHERE id = ? AND recipient_id = ? AND status = 'requested'",
                (link_id, recipient_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE family_links SET status = ?, responded_at = ? WHERE id = ?",
                ("accepted" if accept else "declined", time.time(), link_id),
            )
            return self._decode_family_link(self._row(connection, "family_links", link_id))

    def list_family_links(self, user_id: int) -> list[dict[str, Any]]:
        """Every link involving this user, either direction, newest first,
        with the OTHER party's own name/phone attached so the caller never
        has to make a second lookup."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT fl.*, "
                "CASE WHEN fl.requester_id = ? THEN ru2.name ELSE ru1.name END AS other_name, "
                "CASE WHEN fl.requester_id = ? THEN ru2.phone ELSE ru1.phone END AS other_phone, "
                "CASE WHEN fl.requester_id = ? THEN fl.recipient_id ELSE fl.requester_id END AS other_user_id "
                "FROM family_links fl "
                "JOIN users ru1 ON ru1.id = fl.requester_id "
                "JOIN users ru2 ON ru2.id = fl.recipient_id "
                "WHERE fl.requester_id = ? OR fl.recipient_id = ? "
                "ORDER BY fl.created_at DESC",
                (user_id, user_id, user_id, user_id, user_id),
            )
            return [self._decode_family_link(dict(r), viewer_id=user_id) for r in rows]

    def update_family_link_permissions(
        self, link_id: int, owner_id: int, *,
        notify_accidents: bool | None = None, share_location: bool | None = None,
    ) -> dict[str, Any] | None:
        """Each side configures only THEIR OWN half of the link -- whether
        owner_id is the requester or the recipient decides which pair of
        columns gets updated. Returns None if owner_id is not part of this
        link at all."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT requester_id, recipient_id FROM family_links WHERE id = ?", (link_id,)
            ).fetchone()
            if row is None or owner_id not in (row["requester_id"], row["recipient_id"]):
                return None
            prefix = "requester" if owner_id == row["requester_id"] else "recipient"
            updates: dict[str, int] = {}
            if notify_accidents is not None:
                updates[f"{prefix}_notify_accidents"] = int(notify_accidents)
            if share_location is not None:
                updates[f"{prefix}_share_location"] = int(share_location)
            if updates:
                assignments = ", ".join(f"{key} = ?" for key in updates)
                connection.execute(
                    f"UPDATE family_links SET {assignments} WHERE id = ?",
                    (*updates.values(), link_id),
                )
            return self._decode_family_link(self._row(connection, "family_links", link_id))

    def remove_family_link(self, link_id: int, user_id: int) -> bool:
        """Either party can remove the link at any time -- a pending
        request, or an already-accepted one (leaving a family group).
        Returns False if user_id is not part of this link."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT requester_id, recipient_id FROM family_links WHERE id = ?", (link_id,)
            ).fetchone()
            if row is None or user_id not in (row["requester_id"], row["recipient_id"]):
                return False
            connection.execute("DELETE FROM family_links WHERE id = ?", (link_id,))
            return True

    def accepted_family_members_to_notify(self, user_id: int) -> list[dict[str, Any]]:
        """The other party of every ACCEPTED link involving user_id, where
        THAT OTHER PARTY has opted into being notified about user_id's
        incidents. The two flags are owned by different people, on
        purpose: whether to notify is the RECEIVER's own preference (do
        they want to be told at all), while share_location is user_id's
        OWN choice (do THEY want to disclose their location to this
        specific family member) -- the person being tracked controls
        disclosure, not the person receiving the alert."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM family_links WHERE status = 'accepted' "
                "AND (requester_id = ? OR recipient_id = ?)",
                (user_id, user_id),
            )
            out = []
            for row in rows:
                if row["requester_id"] == user_id:
                    other_id = row["recipient_id"]
                    other_wants_notify = row["recipient_notify_accidents"]
                    my_share_location = row["requester_share_location"]
                else:
                    other_id = row["requester_id"]
                    other_wants_notify = row["requester_notify_accidents"]
                    my_share_location = row["recipient_share_location"]
                if not other_wants_notify:
                    continue
                other = self.get_user(other_id)
                if other is None:
                    continue
                out.append({**other, "link_id": row["id"], "share_location": bool(my_share_location)})
            return out

    def already_notified(self, alert_id: int, recipient_user_id: int) -> bool:
        """Check BEFORE sending, not just before recording -- otherwise a
        real provider would double-send even though the database would
        only ever keep one row (the UNIQUE constraint stops the duplicate
        RECORD, not a duplicate real-world message already sent before
        that record was attempted)."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM family_notifications WHERE alert_id = ? AND recipient_user_id = ?",
                (alert_id, recipient_user_id),
            ).fetchone()
            return row is not None

    def record_family_notification(
        self, alert_id: int, recipient_user_id: int, delivery_status: str
    ) -> dict[str, Any] | None:
        """Returns None if this (alert, recipient) pair was already notified
        -- the UNIQUE constraint makes duplicate suppression automatic, not
        something the caller has to remember to check first."""
        token = secrets.token_urlsafe(24)
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO family_notifications"
                    "(alert_id, recipient_user_id, sent_at, delivery_status, acknowledge_token) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (alert_id, recipient_user_id, time.time(), delivery_status, token),
                )
            except sqlite3.IntegrityError:
                return None
            return self._row(connection, "family_notifications", int(cursor.lastrowid))

    def acknowledge_family_notification(self, acknowledge_token: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM family_notifications WHERE acknowledge_token = ?",
                (acknowledge_token,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE family_notifications SET acknowledged_at = ? WHERE id = ?",
                (time.time(), row["id"]),
            )
            return self._row(connection, "family_notifications", row["id"])

    def list_family_notifications(self, alert_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM family_notifications WHERE alert_id = ? ORDER BY sent_at", (alert_id,)
            )
            return [dict(r) for r in rows]

    @staticmethod
    def _decode_family_link(row: dict[str, Any], viewer_id: int | None = None) -> dict[str, Any]:
        row["requester_notify_accidents"] = bool(row["requester_notify_accidents"])
        row["requester_share_location"] = bool(row["requester_share_location"])
        row["recipient_notify_accidents"] = bool(row["recipient_notify_accidents"])
        row["recipient_share_location"] = bool(row["recipient_share_location"])
        return row

    # ------------------------------------------------------------- settings

    def get_settings(self) -> dict[str, float]:
        """Return every configurable system parameter (FR18)."""
        with self.connect() as connection:
            return {r["key"]: r["value"] for r in connection.execute("SELECT * FROM settings")}

    def update_settings(self, values: dict[str, float]) -> dict[str, float]:
        """Validate and store threshold changes; rejects out-of-order thresholds."""
        current = self.get_settings()
        merged = {**current, **{k: float(v) for k, v in values.items() if k in current}}
        
        unknown = set(values) - set(current)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")

        medium = merged["risk_medium_threshold"]
        high = merged["risk_high_threshold"]
        critical = merged["risk_critical_threshold"]

        if not 0 < medium < high < critical < 1:
            raise ValueError("Thresholds must satisfy 0 < medium < high < critical < 1")
        if merged["session_gap_tolerance_seconds"] <= 0:
            raise ValueError("Gap tolerance must be positive")

        with self.connect() as connection:
            for key, value in merged.items():
                connection.execute("UPDATE settings SET value = ? WHERE key = ?", (value, key))
        return merged

    # ----------------------------------------------------------- access log

    def log_access(self, user_id: int | None, role: str | None, path: str) -> None:
        """Record a data access event (NFR11)."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO access_log(user_id, role, path, ts) VALUES (?, ?, ?, ?)",
                (user_id, role, path, time.time()),
            )

    def recent_access(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent access events for the audit view."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM access_log ORDER BY ts DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in rows]

    # -------------------------------------------------------------- helpers

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        """Run a query expected to return at most one row."""
        with self.connect() as connection:
            row = connection.execute(query, params).fetchone()
            return dict(row) if row else None

    @staticmethod
    def _row(connection: sqlite3.Connection, table: str, row_id: int) -> dict[str, Any]:
        """Read a row back inside an open transaction."""
        row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise LookupError(f"{table} row {row_id} not found")
        return dict(row)

    @staticmethod
    def _decode_session(row: dict[str, Any]) -> dict[str, Any]:
        """Expand the stored summary JSON into a dictionary."""
        try:
            row["summary"] = json.loads(row.get("summary") or "{}")
        except json.JSONDecodeError:
            row["summary"] = {}
        return row

    @staticmethod
    def _decode_alert(row: dict[str, Any]) -> dict[str, Any]:
        """Expand the stored detail JSON (model_version, onset_estimate,
        detection_score, source, ...) into a dictionary."""
        try:
            row["detail"] = json.loads(row.get("detail_json") or "{}")
        except json.JSONDecodeError:
            row["detail"] = {}
        return row


def normalise_phone(phone: str) -> str:
    """Reduce a phone number to digits with an optional leading +, for stable lookup."""
    cleaned = "".join(c for c in phone.strip() if c.isdigit() or c == "+")
    digits = cleaned.lstrip("+")
    if not digits.isdigit() or not 7 <= len(digits) <= 15:
        raise ValueError("Enter a phone number with 7 to 15 digits")
    return f"+{digits}"


def role_allows(role: str, required: str) -> bool:
    """True when `role` sits at or above `required` in the hierarchy (FR2)."""
    if role not in ROLE_RANK or required not in ROLE_RANK:
        return False
    return ROLE_RANK[role] >= ROLE_RANK[required]