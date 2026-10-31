"""Unit and integration tests for Phase 5: SQLite Database and Telemetry Storage."""

import os
from pathlib import Path
import sys
import time
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from database.database import DatabaseManager
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database for unit tests."""
    db_file = tmp_path / "test_factory.db"
    db = DatabaseManager(db_path=str(db_file))
    return db


def test_database_init_and_tables(temp_db):
    """Verify all required tables are created and seeded with machine definitions."""
    with temp_db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = {row["name"] for row in cursor.fetchall()}

    expected_tables = {
        "machines",
        "telemetry",
        "production",
        "alerts",
        "maintenance_events",
        "ai_predictions",
    }
    assert expected_tables.issubset(tables)

    # Verify machines seeded
    with temp_db.get_connection() as conn:
        cursor = conn.execute("SELECT machine_id, name FROM machines ORDER BY machine_id;")
        rows = cursor.fetchall()
        machine_ids = [r["machine_id"] for r in rows]
        assert "M01" in machine_ids
        assert "M02" in machine_ids
        assert "M03" in machine_ids


def test_insert_and_query_telemetry(temp_db):
    """Verify inserting and querying machine telemetry."""
    sample_telemetry = {
        "timestamp": "2026-09-30T18:00:00+00:00",
        "machine_id": "M01",
        "machine_name": "CNC Milling",
        "machine_state": "RUNNING",
        "temperature": 66.8,
        "vibration": 2.48,
        "current": 8.7,
        "voltage": 230.2,
        "rpm": 1790.0,
        "pressure": 6.1,
        "power": 3.3,
        "production_count": 5,
        "good_products": 5,
        "defect_count": 0,
        "defect_rate": 0.0,
        "cycle_time": 12.0,
        "downtime_seconds": 0.0,
        "degradation_level": 0.05,
    }
    row_id = temp_db.insert_telemetry(sample_telemetry)
    assert row_id > 0

    records = temp_db.get_latest_telemetry(machine_id="M01", limit=1)
    assert len(records) == 1
    r = records[0]
    assert r["machine_id"] == "M01"
    assert r["temperature"] == 66.8
    assert r["vibration"] == 2.48
    assert r["production_count"] == 5


def test_insert_production_and_alerts(temp_db):
    """Verify inserting production summary and operational alerts."""
    prod_data = {
        "timestamp": "2026-09-30T18:00:00+00:00",
        "factory_name": "Smart Factory Line 1",
        "factory_status": "NORMAL",
        "production_summary": {
            "total_production": 42,
            "total_good": 41,
            "total_defects": 1,
            "defect_rate": 0.0238,
            "active_machines": 3,
            "warning_machines": 0,
            "failed_machines": 0,
            "maintenance_machines": 0,
            "idle_machines": 0,
        },
    }
    prod_id = temp_db.insert_production(prod_data)
    assert prod_id > 0

    latest_prod = temp_db.get_latest_production()
    assert latest_prod is not None
    assert latest_prod["total_production"] == 42
    assert latest_prod["defect_rate"] == 0.0238

    # Insert alert
    alert_id = temp_db.insert_alert({
        "timestamp": "2026-09-30T18:00:00+00:00",
        "machine_id": "M02",
        "alert_type": "HIGH_VIBRATION",
        "severity": "WARNING",
        "message": "Vibration crossed warning threshold",
        "acknowledged": 0,
    })
    assert alert_id > 0


def test_end_to_end_simulator_mqtt_to_sqlite(tmp_path):
    """Verify complete pipeline: Simulator -> MQTT Publisher -> Mosquitto -> Subscriber -> SQLite."""
    test_db_path = tmp_path / "e2e_smart_factory.db"
    db = DatabaseManager(db_path=str(test_db_path))

    sub = FactoryMqttSubscriber()
    pub = FactoryMqttPublisher()

    # Route incoming subscriber messages directly to SQLite DatabaseManager
    sub.register_handler(db.handle_mqtt_message)

    try:
        assert sub.connect(timeout=3.0) is True
        assert pub.connect(timeout=3.0) is True
        time.sleep(0.3)

        # Publish 2 steps (should yield 6 machine telemetry rows + 2 production rows)
        pub.publish_step(dt=1.0)
        pub.publish_step(dt=1.0)

        # Wait up to 2 seconds for subscriber to receive and SQLite to persist
        start = time.time()
        while db.get_telemetry_count() < 6 and (time.time() - start) < 2.5:
            time.sleep(0.05)

        # Verify SQLite received all machine records
        total_telemetry = db.get_telemetry_count()
        assert total_telemetry >= 6, f"Expected at least 6 telemetry rows, got {total_telemetry}"

        # Verify each machine is represented in SQLite
        m01_count = db.get_telemetry_count(machine_id="M01")
        m02_count = db.get_telemetry_count(machine_id="M02")
        m03_count = db.get_telemetry_count(machine_id="M03")
        assert m01_count >= 2
        assert m02_count >= 2
        assert m03_count >= 2

        # Verify production table received entries
        prod = db.get_latest_production()
        assert prod is not None
        assert prod["active_machines"] == 3

    finally:
        pub.disconnect()
        sub.disconnect()
