"""Unit and integration tests for Phase 7: Machine Health and Operational Risk Analysis."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database import DatabaseManager
from analytics.health import HealthEvaluator, HealthCondition
from analytics.risk import OperationalRiskEngine, RiskLevel
from analytics.maintenance import PredictiveMaintenanceEngine, MaintenanceUrgency
from analytics.oee import OEEEngine


@pytest.fixture
def health_evaluator():
    return HealthEvaluator()


@pytest.fixture
def risk_engine():
    return OperationalRiskEngine()


@pytest.fixture
def maint_engine():
    return PredictiveMaintenanceEngine()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "analytics_test_factory.db"
    return DatabaseManager(db_path=str(db_file))


def test_health_evaluator_nominal(health_evaluator):
    """Verify nominal machine operating values result in EXCELLENT health."""
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
    res = health_evaluator.evaluate_health(nominal_telemetry, ai_anomaly_score=0.02)
    assert res["health_score"] >= 90.0
    assert res["condition"] == HealthCondition.EXCELLENT
    assert res["rule_health"] >= 95.0


def test_health_evaluator_degraded(health_evaluator):
    """Verify elevated temperature and vibration lower health condition."""
    degraded_telemetry = {
        "machine_id": "M01",
        "machine_state": "WARNING",
        "temperature": 82.0,
        "vibration": 5.2,
        "current": 13.5,
        "rpm": 1400.0,
        "pressure": 5.8,
        "power": 4.1,
    }
    res = health_evaluator.evaluate_health(degraded_telemetry, ai_anomaly_score=0.65)
    assert res["health_score"] < 60.0
    assert res["condition"] in (HealthCondition.POOR, HealthCondition.CRITICAL)


def test_operational_risk_nominal_vs_critical(risk_engine, health_evaluator):
    """Verify risk engine distinguishes low risk normal ops from high/critical risk."""
    # 1. Normal Machine
    norm_telemetry = {
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.2,
        "vibration": 2.48,
        "defect_rate": 0.0,
        "downtime_seconds": 0.0,
    }
    norm_health = health_evaluator.evaluate_health(norm_telemetry, ai_anomaly_score=0.02)
    norm_risk = risk_engine.evaluate_risk(norm_telemetry, norm_health, ai_anomaly_score=0.02)

    assert norm_risk["risk_score"] < 30.0
    assert norm_risk["risk_level"] == RiskLevel.LOW

    # 2. Critical Machine
    crit_telemetry = {
        "machine_id": "M01",
        "machine_state": "FAILURE",
        "temperature": 94.0,
        "vibration": 6.8,
        "defect_rate": 0.25,
        "downtime_seconds": 120.0,
    }
    crit_health = health_evaluator.evaluate_health(crit_telemetry, ai_anomaly_score=0.95)
    crit_risk = risk_engine.evaluate_risk(crit_telemetry, crit_health, ai_anomaly_score=0.95)

    assert crit_risk["risk_score"] >= 85.0
    assert crit_risk["risk_level"] == RiskLevel.CRITICAL


def test_predictive_maintenance_engine(maint_engine, health_evaluator, risk_engine):
    """Verify predictive maintenance pinpoints bearing degradation and creates prescriptive action."""
    vibration_problem_telemetry = {
        "timestamp": "2026-09-30T19:30:00+00:00",
        "machine_id": "M02",
        "machine_state": "WARNING",
        "temperature": 73.0,
        "vibration": 6.0,  # Highly elevated vibration
        "current": 12.5,
        "rpm": 1150.0,
        "pressure": 8.4,
        "power": 5.5,
    }
    health = health_evaluator.evaluate_health(vibration_problem_telemetry, ai_anomaly_score=0.72)
    risk = risk_engine.evaluate_risk(vibration_problem_telemetry, health, ai_anomaly_score=0.72)

    rec = maint_engine.diagnose_and_recommend(vibration_problem_telemetry, health, risk)

    assert rec["urgency"] in (MaintenanceUrgency.PREVENTIVE_WARNING, MaintenanceUrgency.IMMEDIATE_ACTION)
    assert "Bearings" in rec["target_component"]
    assert "vibration" in rec["recommendation"].lower() or "bearings" in rec["recommendation"].lower()


def test_maintenance_logging_to_sqlite(maint_engine, temp_db):
    """Verify actionable maintenance recommendations log to SQLite maintenance_events table."""
    rec = {
        "timestamp": "2026-09-30T19:35:00+00:00",
        "machine_id": "M01",
        "urgency": MaintenanceUrgency.PREVENTIVE_WARNING,
        "target_component": "Bearings & Drive Shaft",
        "recommendation": "PREVENTIVE WARNING: Inspect bearings within 24h.",
    }
    event_id = maint_engine.log_recommendation_to_db(rec, temp_db)
    assert event_id is not None
    assert event_id > 0

    with temp_db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM maintenance_events WHERE id = ?;", (event_id,))
        row = cursor.fetchone()
        assert row["machine_id"] == "M01"
        assert row["event_type"] == MaintenanceUrgency.PREVENTIVE_WARNING


def test_machine_health_logging_to_sqlite(health_evaluator, risk_engine, temp_db):
    """Verify machine health and operational risk records persist into machine_health SQLite table."""
    telemetry = {
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.0,
        "vibration": 2.5,
        "current": 8.5,
        "rpm": 1800.0,
        "pressure": 6.0,
        "power": 3.2,
    }
    health = health_evaluator.evaluate_health(telemetry, ai_anomaly_score=0.03)
    risk = risk_engine.evaluate_risk(telemetry, health, ai_anomaly_score=0.03)

    h_id = temp_db.insert_machine_health(
        health_data=health,
        risk_data=risk,
        anomaly_score=0.03,
        timestamp="2026-09-30T19:35:00+00:00",
    )
    assert h_id is not None
    assert h_id > 0

    records = temp_db.get_latest_machine_health(machine_id="M01", limit=1)
    assert len(records) == 1
    assert records[0]["machine_id"] == "M01"
    assert records[0]["health_score"] == health["health_score"]
    assert records[0]["risk_level"] == "LOW"
    assert records[0]["health_condition"] == "EXCELLENT"


def test_oee_calculation():
    """Verify OEE calculation."""
    telemetry = {
        "downtime_seconds": 360.0,      # 10% downtime of 3600s
        "production_count": 280,
        "good_products": 273,          # 97.5% quality
        "defect_count": 7,
    }
    oee_res = OEEEngine.calculate_machine_oee(
        telemetry, nominal_cycle_time=12.0, planned_operating_seconds=3600.0
    )
    assert oee_res["availability"] == 90.0
    assert oee_res["quality"] == 97.5
    assert oee_res["oee"] > 0.0
