"""Unit and integration tests for Phase 10: Alerts & Master End-to-End Factory Integration."""

import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from database.database import DatabaseManager
from alerts.alert_manager import AlertManager, AlertSeverity, AlertType
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber
from main_pipeline import SmartFactoryMasterPipeline
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "alerts_test_factory.db"
    return DatabaseManager(db_path=str(db_file))


@pytest.fixture
def alert_manager():
    return AlertManager(cooldown_seconds=2.0)


def test_alert_manager_evaluation_nominal(alert_manager):
    """Verify nominal machine operations generate no alarms."""
    nominal_telemetry = {
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.0,
        "vibration": 2.5,
        "current": 8.5,
        "rpm": 1800.0,
        "pressure": 6.0,
        "power": 3.2,
    }
    ai_pred = {"anomaly_score": 0.02, "is_anomaly": 0, "recommendation": "Nominal"}
    health_eval = {"health_score": 98.0, "condition": "EXCELLENT"}
    risk_eval = {"risk_score": 5.0, "risk_level": "LOW", "primary_driver": "None"}
    maint_diag = {"urgency": "NORMAL", "target_component": "General", "action": "MONITOR"}

    alerts = alert_manager.evaluate_alerts(
        nominal_telemetry,
        ai_pred=ai_pred,
        health_eval=health_eval,
        risk_eval=risk_eval,
        maint_diag=maint_diag,
    )
    assert len(alerts) == 0


def test_alert_manager_evaluation_stress_and_failure(alert_manager):
    """Verify breakdown and elevated stress generate appropriate CRITICAL and WARNING alerts."""
    failure_telemetry = {
        "machine_id": "M01",
        "machine_state": "FAILURE",
        "temperature": 94.5,
        "vibration": 6.8,
        "current": 16.5,
        "rpm": 30.0,
        "pressure": 5.2,
        "power": 0.8,
    }
    ai_pred = {"anomaly_score": 0.95, "is_anomaly": 1, "recommendation": "Bearing fault"}
    health_eval = {"health_score": 10.0, "condition": "CRITICAL"}
    risk_eval = {"risk_score": 92.0, "risk_level": "CRITICAL", "primary_driver": "Mechanical Vibration"}
    maint_diag = {"urgency": "EMERGENCY", "target_component": "Bearing Assembly", "action": "OVERHAUL"}

    alerts = alert_manager.evaluate_alerts(
        failure_telemetry,
        ai_pred=ai_pred,
        health_eval=health_eval,
        risk_eval=risk_eval,
        maint_diag=maint_diag,
    )

    alert_types = [a["alert_type"] for a in alerts]
    assert AlertType.EQUIPMENT_BREAKDOWN in alert_types
    assert AlertType.AI_ANOMALY in alert_types
    assert AlertType.HEALTH_DEGRADATION in alert_types
    assert AlertType.HIGH_OPERATIONAL_RISK in alert_types
    assert AlertType.MAINTENANCE_REQUIRED in alert_types
    assert all(a["severity"] == AlertSeverity.CRITICAL for a in alerts)


def test_alert_cooldown_deduplication(alert_manager):
    """Verify identical alerts are suppressed during the cooldown window."""
    failure_telemetry = {
        "machine_id": "M01",
        "machine_state": "FAILURE",
        "temperature": 92.0,
        "vibration": 6.6,
    }
    ai_pred = {"anomaly_score": 0.88, "is_anomaly": 1, "recommendation": "Fault"}

    # First dispatch - should pass through
    alerts1 = alert_manager.evaluate_and_dispatch(failure_telemetry, ai_pred=ai_pred)
    assert len(alerts1) > 0

    # Immediate second dispatch - should be filtered by cooldown
    alerts2 = alert_manager.evaluate_and_dispatch(failure_telemetry, ai_pred=ai_pred)
    assert len(alerts2) == 0

    # Wait for cooldown to expire
    time.sleep(2.1)
    alerts3 = alert_manager.evaluate_and_dispatch(failure_telemetry, ai_pred=ai_pred)
    assert len(alerts3) > 0


def test_alert_sqlite_persistence_and_acknowledgment(temp_db, alert_manager):
    """Verify alerts are persisted in SQLite, queryable, and acknowledgeable."""
    telemetry = {
        "machine_id": "M02",
        "machine_state": "FAILURE",
        "temperature": 96.0,
        "vibration": 7.8,
    }
    dispatched = alert_manager.evaluate_and_dispatch(telemetry, db=temp_db)
    assert len(dispatched) > 0

    # Query from DB
    db_alerts = temp_db.get_alerts(machine_id="M02")
    assert len(db_alerts) >= 1
    target_alert = db_alerts[0]
    assert target_alert["acknowledged"] == 0

    # Acknowledge
    assert temp_db.acknowledge_alert(target_alert["id"]) is True

    # Re-query
    unacked = temp_db.get_alerts(machine_id="M02", unacknowledged_only=True)
    assert not any(a["id"] == target_alert["id"] for a in unacked)


def test_fastapi_alerts_endpoints():
    """Verify FastAPI /api/alerts and /api/alerts/{id}/acknowledge endpoints."""
    client = TestClient(app)

    # 1. Get alerts
    res = client.get("/api/alerts")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 2. Test 404 for non-existent alert acknowledgment
    res_ack = client.post("/api/alerts/999999/acknowledge")
    assert res_ack.status_code == 404


def test_master_pipeline_end_to_end():
    """Verify the Master Pipeline executes multi-stage simulation with AI, Health, Alerts & Twin."""
    pipeline = SmartFactoryMasterPipeline()

    assert pipeline.start(timeout=3.0) is True
    try:
        # Step 1: Normal
        snap1 = pipeline.step(dt=1.0)
        assert snap1["factory_status"] in ("NORMAL", "WARNING", "CRITICAL")
        assert "M01" in pipeline.twin.machines
        assert pipeline.twin.machines["M01"].health_score > 0.0

        # Step 2: Inject Degradation
        pipeline.publisher.simulator.inject_degradation("M01", rate=0.40)
        pipeline.step(dt=1.5)

        # Step 3: Inject Failure
        pipeline.publisher.simulator.inject_failure("M01")
        pipeline.step(dt=1.0)

        # Allow async MQTT subscriber loop to process message
        t0 = time.time()
        while pipeline.twin.machines["M01"].state != "FAILURE" and (time.time() - t0) < 1.5:
            time.sleep(0.05)

        # Verify Twin and Alerts
        m01_twin = pipeline.twin.machines["M01"]
        assert m01_twin.state == "FAILURE"
        assert m01_twin.risk_level == "CRITICAL"

        # Verify SQLite has recent records
        recent_telemetry = pipeline.db.get_latest_telemetry(machine_id="M01", limit=1)
        assert len(recent_telemetry) > 0
        assert recent_telemetry[0]["machine_id"] == "M01"

    finally:
        pipeline.stop()
