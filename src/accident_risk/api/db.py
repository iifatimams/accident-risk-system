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

ROLES: tuple[str, ...] = ("driver", "account_manager", "administrator")
# Higher rank inherits every capability of the ranks below it.
ROLE_RANK: dict[str, int] = {"driver": 0, "account_manager": 1, "administrator": 2}
DEVICE_KINDS: tuple[str, ...] = ("adapter", "wearable")
DEVICE_STATUSES: tuple[str, ...] = ("active", "spare", "out_of_service")
ALERT_KINDS: tuple[str, ...] = ("high_risk", "collision")
# "pending"/"confirmed"/"cancelled" remain exactly as before for high_risk
# alerts and the existing manual /driver/collision report. An automatically
# DETECTED collision (collision_detector.py) instead uses the richer
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
FAMILY_CONTACT_STATUSES: tuple[str, ...] = ("invited", "accepted", "revoked")

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

-- Research-grade record, alongside (not replacing) the throttled session_points
-- used by the live map/history UI. Same 1Hz cadence as session_points -- raw
-- per-frame data still does not belong in SQLite. Everything beyond the few
-- queryable columns (ts, device, source) lives in payload_json so adding a new
-- sensor/derived field never requires a migration.
CREATE TABLE IF NOT EXISTS session_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts            REAL    NOT NULL,
    device_id     TEXT,
    source        TEXT    NOT NULL,
    payload_json  TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_samples_session ON session_samples(session_id, ts);

-- Explicit, self-granted, revocable consent for research recording/export.
-- Independent of family-monitoring consent (not yet built) and of any
-- role/ownership check on export -- consent gates whether research-grade
-- data exists/ships for THIS participant at all, regardless of who is asking.
CREATE TABLE IF NOT EXISTS research_consent (
    user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    granted       INTEGER NOT NULL DEFAULT 0,
    granted_at    REAL,
    revoked_at    REAL
);

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
    -- onset_estimate, detection_score, source: "automatic_detector" or
    -- "manual_report"). Kept here, not a separate table, since every
    -- collision incident already IS an alert row with kind='collision';
    -- JSON so new fields never need a migration.
    detail_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);

-- Consent-based family monitoring. A contact is NOT a users() row -- they
-- never log in; invite_token (shared by the driver out-of-band, there is no
-- bundled SMS/email provider) is their only credential, used once to
-- accept and thereafter to acknowledge notifications. Independent of
-- research_consent above -- accepting a family invite never implies
-- research consent, and vice versa.
CREATE TABLE IF NOT EXISTS family_contacts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name             TEXT    NOT NULL,
    phone            TEXT    NOT NULL,
    relationship     TEXT    NOT NULL DEFAULT '',
    status           TEXT    NOT NULL DEFAULT 'invited',
    notify_accidents INTEGER NOT NULL DEFAULT 1,
    share_location   INTEGER NOT NULL DEFAULT 0,
    invite_token     TEXT    NOT NULL UNIQUE,
    created_at       REAL    NOT NULL,
    responded_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_family_contacts_driver ON family_contacts(driver_user_id);

-- UNIQUE(alert_id, contact_id) gives duplicate-notification suppression for
-- free: a second attempt to notify the same contact about the same incident
-- is simply ignored at insert time, not re-sent.
CREATE TABLE IF NOT EXISTS family_notifications (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id           INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    contact_id         INTEGER NOT NULL REFERENCES family_contacts(id) ON DELETE CASCADE,
    sent_at            REAL    NOT NULL,
    delivery_status    TEXT    NOT NULL,
    acknowledge_token  TEXT    NOT NULL UNIQUE,
    acknowledged_at    REAL,
    UNIQUE(alert_id, contact_id)
);

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
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value)
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

    # ------------------------------------------------------- research consent

    def get_research_consent(self, user_id: int) -> dict[str, Any]:
        """Current consent state; a user who has never decided reads as not granted."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_consent WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return {"user_id": user_id, "granted": False, "granted_at": None, "revoked_at": None}
        return {**dict(row), "granted": bool(row["granted"])}

    def set_research_consent(self, user_id: int, granted: bool) -> dict[str, Any]:
        """Grant or revoke research consent. Revocation stops future research
        recording; it does not retroactively delete samples already written
        while consent was active."""
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO research_consent(user_id, granted, granted_at, revoked_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "granted = excluded.granted, "
                "granted_at = CASE WHEN excluded.granted = 1 THEN excluded.granted_at "
                "ELSE research_consent.granted_at END, "
                "revoked_at = CASE WHEN excluded.granted = 0 THEN excluded.revoked_at "
                "ELSE research_consent.revoked_at END",
                (user_id, int(granted), now if granted else None, None if granted else now),
            )
        return self.get_research_consent(user_id)

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

    def session_by_stream(self, stream_id: str) -> dict[str, Any] | None:
        """Resolve a session from the stream_id the ingestion API receives."""
        return self._fetch_one("SELECT * FROM sessions WHERE stream_id = ?", (stream_id,))

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

    def add_sample(
        self, session_id: int, ts: float, device_id: str | None, source: str, payload: dict[str, Any]
    ) -> None:
        """Append one research-grade sample (NFR: recording/export milestone).

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
        """Return a session's research-grade samples in time order, payload inlined."""
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

    def record_gap(self, session_id: int, source: str, started_at: float) -> int:
        """Open a gap record for a dropped sensor link (FR13.5)."""
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO session_gaps(session_id, source, started_at) VALUES (?, ?, ?)",
                (session_id, source, started_at),
            )
            return int(cursor.lastrowid)

    def close_gap(self, gap_id: int, ended_at: float) -> None:
        """Close an open gap record once the link is restored."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE session_gaps SET ended_at = ? WHERE id = ?", (ended_at, gap_id)
            )

    def session_gaps(self, session_id: int) -> list[dict[str, Any]]:
        """Return every recorded gap for a session."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM session_gaps WHERE session_id = ? ORDER BY started_at",
                (session_id,),
            )
            return [dict(r) for r in rows]

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

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        """Fetch one alert's current (possibly since-updated) state."""
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
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
        self, *, user_id: int | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List alerts newest first, scoped to one driver or kind when given."""
        query = (
            "SELECT a.*, u.name AS driver_name FROM alerts a JOIN users u ON u.id = a.user_id"
        )
        clauses, params = [], []
        if user_id is not None:
            clauses.append("a.user_id = ?")
            params.append(user_id)
        if kind is not None:
            clauses.append("a.kind = ?")
            params.append(kind)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)

        with self.connect() as connection:
            return [self._decode_alert(dict(r)) for r in connection.execute(query, params)]

    # ------------------------------------------------------- family contacts

    def invite_family_contact(
        self, driver_user_id: int, name: str, phone: str, *,
        relationship: str = "", notify_accidents: bool = True, share_location: bool = False,
    ) -> dict[str, Any]:
        """Create a pending invitation. The invite_token is the contact's
        only credential (they never log in) -- the driver must relay it
        out-of-band (text/call them), there is no bundled SMS/email
        provider. Returning it here is handing the CREATOR their own new
        resource's token, not leaking anyone else's credential."""
        token = secrets.token_urlsafe(24)
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO family_contacts(driver_user_id, name, phone, relationship, "
                "status, notify_accidents, share_location, invite_token, created_at) "
                "VALUES (?, ?, ?, ?, 'invited', ?, ?, ?, ?)",
                (
                    driver_user_id, name.strip(), phone.strip(), relationship.strip(),
                    int(notify_accidents), int(share_location), token, time.time(),
                ),
            )
            return self._decode_family_contact(self._row(connection, "family_contacts", int(cursor.lastrowid)))

    def accept_family_contact(self, invite_token: str) -> dict[str, Any] | None:
        """The contact's one consent action. Only a pending invitation can
        be accepted -- an already-revoked or already-accepted token is
        inert, not reusable to re-authorize anything."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM family_contacts WHERE invite_token = ? AND status = 'invited'",
                (invite_token,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE family_contacts SET status = 'accepted', responded_at = ? WHERE id = ?",
                (time.time(), row["id"]),
            )
            return self._decode_family_contact(self._row(connection, "family_contacts", row["id"]))

    def revoke_family_contact(self, contact_id: int, driver_user_id: int) -> dict[str, Any] | None:
        """Only the owning driver can revoke -- checked here, not just at
        the route layer, so a direct call can't bypass ownership."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM family_contacts WHERE id = ? AND driver_user_id = ?",
                (contact_id, driver_user_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE family_contacts SET status = 'revoked', responded_at = ? WHERE id = ?",
                (time.time(), contact_id),
            )
            return self._decode_family_contact(self._row(connection, "family_contacts", contact_id))

    def update_family_contact_permissions(
        self, contact_id: int, driver_user_id: int, *,
        notify_accidents: bool | None = None, share_location: bool | None = None,
    ) -> dict[str, Any] | None:
        """The DRIVER configures these -- independently of each other, and
        independently of whether the contact has accepted yet. Accepting an
        invite is the contact's OWN consent to being contacted at all; it
        never grants location access by itself, since share_location is a
        separate column the driver controls here, defaulting to False."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM family_contacts WHERE id = ? AND driver_user_id = ?",
                (contact_id, driver_user_id),
            ).fetchone()
            if row is None:
                return None
            updates: dict[str, int] = {}
            if notify_accidents is not None:
                updates["notify_accidents"] = int(notify_accidents)
            if share_location is not None:
                updates["share_location"] = int(share_location)
            if updates:
                assignments = ", ".join(f"{key} = ?" for key in updates)
                connection.execute(
                    f"UPDATE family_contacts SET {assignments} WHERE id = ?",
                    (*updates.values(), contact_id),
                )
            return self._decode_family_contact(self._row(connection, "family_contacts", contact_id))

    def list_family_contacts(self, driver_user_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM family_contacts WHERE driver_user_id = ? ORDER BY created_at DESC",
                (driver_user_id,),
            )
            return [self._decode_family_contact(dict(r)) for r in rows]

    def accepted_notifiable_contacts(self, driver_user_id: int) -> list[dict[str, Any]]:
        """Contacts eligible for an accident notification right now:
        accepted (not merely invited, never revoked) AND opted into
        accident notifications specifically -- share_location is a
        SEPARATE permission, checked by the caller when building the
        message, not a gate on whether to notify at all."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM family_contacts WHERE driver_user_id = ? "
                "AND status = 'accepted' AND notify_accidents = 1",
                (driver_user_id,),
            )
            return [self._decode_family_contact(dict(r)) for r in rows]

    def already_notified(self, alert_id: int, contact_id: int) -> bool:
        """Check BEFORE sending, not just before recording -- otherwise a
        real provider would double-send even though the database would
        only ever keep one row (the UNIQUE constraint stops the duplicate
        RECORD, not a duplicate real-world message already sent before
        that record was attempted)."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM family_notifications WHERE alert_id = ? AND contact_id = ?",
                (alert_id, contact_id),
            ).fetchone()
            return row is not None

    def record_family_notification(
        self, alert_id: int, contact_id: int, delivery_status: str
    ) -> dict[str, Any] | None:
        """Returns None if this (alert, contact) pair was already notified
        -- the UNIQUE constraint makes duplicate suppression automatic, not
        something the caller has to remember to check first."""
        token = secrets.token_urlsafe(24)
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO family_notifications"
                    "(alert_id, contact_id, sent_at, delivery_status, acknowledge_token) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (alert_id, contact_id, time.time(), delivery_status, token),
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
    def _decode_family_contact(row: dict[str, Any]) -> dict[str, Any]:
        row["notify_accidents"] = bool(row["notify_accidents"])
        row["share_location"] = bool(row["share_location"])
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