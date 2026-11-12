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
