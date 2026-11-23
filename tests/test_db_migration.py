"""Checkpoint resume task #2 (docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md):
CREATE TABLE IF NOT EXISTS alone does not add a column to an alerts table
that already exists without it. Tests the idempotent ALTER TABLE migration
against a synthetic pre-migration database, never the real data/app.sqlite3.
"""

import sqlite3
import time

from accident_risk.api.db import Database


def _create_pre_migration_schema(path):
    """Mimics an alerts table from before detail_json existed, with some
    rows already in it -- this is what Database(path) will see on open."""
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            risk_index REAL,
            latitude REAL,
            longitude REAL,
            acknowledged_by INTEGER,
            acknowledged_at REAL
        )
        """
    )
    connection.execute(
        "INSERT INTO users(id, phone, name, role, created_at) VALUES (1, '+971500000001', 'Pre-migration Driver', 'driver', ?)",
        (time.time(),),
    )
    connection.execute(
        "INSERT INTO alerts(id, user_id, kind, status, created_at, risk_index) "
        "VALUES (1, 1, 'high_risk', 'pending', ?, 0.8)",
        (time.time(),),
    )
    connection.execute(
        "INSERT INTO alerts(id, user_id, kind, status, created_at, risk_index) "
        "VALUES (2, 1, 'collision', 'confirmed', ?, 0.95)",
        (time.time(),),
    )
    connection.commit()
    connection.close()


def test_opening_a_pre_migration_database_adds_detail_json_without_data_loss(tmp_path):
    db_path = tmp_path / "pre_migration.sqlite3"
    _create_pre_migration_schema(db_path)

    # Sanity check on the fixture itself: confirms the synthetic DB really
    # lacks detail_json before Database() ever touches it.
    raw = sqlite3.connect(db_path)
    columns_before = {row[1] for row in raw.execute("PRAGMA table_info(alerts)")}
    assert "detail_json" not in columns_before
    raw.close()

    database = Database(db_path)

    with database.connect() as connection:
        columns_after = {row["name"] for row in connection.execute("PRAGMA table_info(alerts)")}
        assert "detail_json" in columns_after

        # Existing rows survive the migration, and the new column defaults
        # to '{}' rather than NULL or an error, for every pre-existing row.
        rows = connection.execute("SELECT id, kind, status, risk_index, detail_json FROM alerts ORDER BY id").fetchall()
        assert len(rows) == 2
        assert rows[0]["kind"] == "high_risk" and rows[0]["status"] == "pending"
        assert rows[1]["kind"] == "collision" and rows[1]["status"] == "confirmed"
        assert rows[0]["detail_json"] == "{}"
        assert rows[1]["detail_json"] == "{}"

    # The migrated database must still support the normal write path.
    created = database.create_alert(
        user_id=1, kind="collision", status="suspected", risk_index=0.9,
        detail={"model_version": "kinematic_threshold_v1_unvalidated"},
    )
    assert created["detail"]["model_version"] == "kinematic_threshold_v1_unvalidated"


def test_migration_is_idempotent_across_repeated_opens(tmp_path):
    db_path = tmp_path / "idempotent.sqlite3"
    _create_pre_migration_schema(db_path)

    Database(db_path)
    # Re-opening an already-migrated database (detail_json already present)
    # must not raise "duplicate column name" or otherwise fail.
    second = Database(db_path)
    with second.connect() as connection:
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(alerts)")]
        assert columns.count("detail_json") == 1


def test_fresh_database_already_has_detail_json_with_no_migration_needed(tmp_path):
    """A brand-new database goes straight through CREATE TABLE IF NOT EXISTS
    with detail_json already in SCHEMA; _migrate must be a true no-op here,
    not just harmless."""
    database = Database(tmp_path / "fresh.sqlite3")
    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(alerts)")}
        assert "detail_json" in columns


def test_opening_a_pre_evaluation_schema_database_adds_new_tables_without_data_loss(tmp_path):
    """Chapter 6 evaluation support (threshold_history, evaluation_annotations,
    alert_verifications, collision_tests) must appear on an existing database
    that predates them, with every existing session/alert/user row intact."""
    db_path = tmp_path / "pre_evaluation.sqlite3"
    _create_pre_migration_schema(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL UNIQUE,
            user_id INTEGER NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            complete INTEGER NOT NULL DEFAULT 1,
            summary TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    connection.execute(
        "INSERT INTO sessions(id, stream_id, user_id, started_at) VALUES (1, 's1', 1, ?)",
        (time.time(),),
    )
    connection.commit()
    connection.close()

    database = Database(db_path)

    with database.connect() as connection:
        tables = {
            row["name"] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table in (
            "threshold_history", "evaluation_annotations",
            "alert_verifications", "collision_tests",
        ):
            assert table in tables

        # Pre-existing data from before this migration survives untouched.
        assert connection.execute("SELECT count(*) AS n FROM sessions").fetchone()["n"] == 1
        assert connection.execute("SELECT count(*) AS n FROM alerts").fetchone()["n"] == 2
        assert connection.execute("SELECT count(*) AS n FROM users").fetchone()["n"] == 1

        # threshold_history is seeded with one row so an immediate
        # effective_thresholds_at() lookup never comes back empty.
        assert connection.execute("SELECT count(*) AS n FROM threshold_history").fetchone()["n"] == 1

    thresholds = database.effective_thresholds_at(time.time())
    assert thresholds["risk_high_threshold"] == 0.70


def test_two_devices_may_share_an_identifier(tmp_path):
    """Many cheap OBD-II BLE adapters broadcast the same generic name, so
    identifier text is intentionally not unique -- device identity is the
    row's own id."""
    database = Database(tmp_path / "devices.sqlite3")
    first = database.create_device("adapter", "OBDII")
    second = database.create_device("adapter", "OBDII")
    assert first["id"] != second["id"]
    assert first["identifier"] == second["identifier"] == "OBDII"

    renamed = database.update_device(second["id"], identifier="OBDII")
    assert renamed["id"] == second["id"]


def test_opening_a_pre_unique_identifier_database_drops_the_constraint_without_data_loss(tmp_path):
    """A database created before identifier's UNIQUE constraint was removed
    must migrate to allow duplicates, keeping every existing device row
    (including its assignment) intact."""
    db_path = tmp_path / "pre_unique_identifier.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO users(id, phone, name, role, created_at) VALUES (7, '+971500000007', 'Driver Seven', 'driver', ?)",
        (time.time(),),
    )
    connection.execute(
        """
        CREATE TABLE devices (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            kind          TEXT    NOT NULL,
            identifier    TEXT    NOT NULL UNIQUE,
            status        TEXT    NOT NULL DEFAULT 'spare',
            assigned_to   INTEGER,
            note          TEXT    NOT NULL DEFAULT '',
            created_at    REAL    NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO devices(id, kind, identifier, status, assigned_to, note, created_at) "
        "VALUES (1, 'adapter', 'OBDII', 'active', 7, 'first unit', ?)",
        (time.time(),),
    )
    connection.commit()
    connection.close()

    database = Database(db_path)

    with database.connect() as connection:
        unique_indexes = [
            row["name"] for row in connection.execute("PRAGMA index_list(devices)") if row["unique"]
        ]
        assert not unique_indexes

        row = connection.execute("SELECT * FROM devices WHERE id = 1").fetchone()
        assert row["kind"] == "adapter"
        assert row["identifier"] == "OBDII"
        assert row["status"] == "active"
        assert row["assigned_to"] == 7
        assert row["note"] == "first unit"

    second = database.create_device("adapter", "OBDII")
    assert second["id"] != 1


def test_opening_a_pre_rename_family_notifications_database_renames_contact_id(tmp_path):
    """A database created before the family_links redesign still has this
    table's recipient column named contact_id, not recipient_user_id --
    every query against it (send, acknowledge, or a recipient's own inbox)
    was silently broken against such a database until renamed."""
    db_path = tmp_path / "pre_rename_family_notifications.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE family_notifications (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id           INTEGER NOT NULL,
            contact_id         INTEGER NOT NULL,
            sent_at            REAL    NOT NULL,
            delivery_status    TEXT    NOT NULL,
            acknowledge_token  TEXT    NOT NULL UNIQUE,
            acknowledged_at    REAL,
            UNIQUE(alert_id, contact_id)
        )
        """
    )
    connection.execute(
        "INSERT INTO family_notifications"
        "(id, alert_id, contact_id, sent_at, delivery_status, acknowledge_token) "
        "VALUES (1, 1, 7, ?, 'mock_sent', 'tok-abc123')",
        (time.time(),),
    )
    connection.commit()
    connection.close()

    database = Database(db_path)

    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(family_notifications)")}
        assert "recipient_user_id" in columns
        assert "contact_id" not in columns
        row = connection.execute("SELECT * FROM family_notifications WHERE id = 1").fetchone()
        assert row["recipient_user_id"] == 7
        assert row["acknowledge_token"] == "tok-abc123"

    acknowledged = database.acknowledge_family_notification("tok-abc123")
    assert acknowledged is not None and acknowledged["acknowledged_at"] is not None
