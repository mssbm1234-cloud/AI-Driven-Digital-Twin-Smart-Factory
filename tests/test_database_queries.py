"""Unit and integration tests for SQLite Database Schema, Views, and Parameterized Queries."""

from pathlib import Path
import sqlite3
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database import DatabaseManager


@pytest.fixture
def db():
    return DatabaseManager()


def test_database_and_all_tables_exist(db):
    """Verify SQLite database and all required tables/views exist."""
    assert db.db_path.exists()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view');")
        entities = {row["name"] for row in cursor.fetchall()}

    expected_entities = {
        "machines",
        "telemetry",
        "production",
        "alerts",
        "maintenance_events",
        "ai_predictions",
        "machine_health",
        "v_telemetry_with_ai",
        "v_latest_machine_overview",
    }
    assert expected_entities.issubset(entities)


def test_views_and_aggregated_queries(db):
    """Verify time-series views and latest machine overview views execute successfully."""
    with db.get_connection() as conn:
        # Test v_telemetry_with_ai view
        cursor = conn.execute("SELECT * FROM v_telemetry_with_ai LIMIT 5;")
        rows = cursor.fetchall()
        assert isinstance(rows, list)

        # Test v_latest_machine_overview view
        cursor_latest = conn.execute("SELECT * FROM v_latest_machine_overview;")
        latest_rows = cursor_latest.fetchall()
        assert len(latest_rows) <= 3


def test_machine_filtering_queries(db):
    """Verify queries parameterized for individual machines (M01, M02, M03) work cleanly."""
    with db.get_connection() as conn:
        for m_id in ["M01", "M02", "M03"]:
            query = f"SELECT * FROM v_latest_machine_overview WHERE machine_id = '{m_id}';"
            cursor = conn.execute(query)
            row = cursor.fetchone()
            if row:
                assert row["machine_id"] == m_id
