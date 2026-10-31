"""Unit and integration tests for Phase 8: Digital Twin and Real-Time Visualization."""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from digital_twin.twin import DigitalTwin, MachineTwin, TwinStatusIndicator
from api.main import app, twin


@pytest.fixture
def test_twin():
    return DigitalTwin()


@pytest.fixture
def api_client():
    return TestClient(app)


def test_machine_twin_initialization(test_twin):
    """Verify MachineTwin initializes all attributes and exports serializable dict."""
    m = test_twin.machines["M01"]
    assert m.machine_id == "M01"
    assert m.name == "CNC Milling Machine"
    assert m.health_score == 100.0

    d = m.to_dict()
    assert "telemetry" in d
    assert "status_indicator" in d
    assert "oee_metrics" in d


def test_digital_twin_updates_and_status_indicators(test_twin):
    """Verify live updates change status indicators between NORMAL, WARNING, and CRITICAL."""
    # 1. Normal Update
    norm_telemetry = {
        "timestamp": "2026-09-30T20:00:00+00:00",
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.2,
        "vibration": 2.5,
        "current": 8.5,
        "rpm": 1800.0,
        "pressure": 6.0,
        "power": 3.2,
    }
    test_twin.update_machine(
        "M01",
        telemetry=norm_telemetry,
        ai_pred={"anomaly_score": 0.02},
        health_eval={"health_score": 98.5, "condition": "EXCELLENT"},
        risk_eval={"risk_score": 5.0, "risk_level": "LOW", "primary_driver": "None"},
    )
    m01 = test_twin.machines["M01"]
    assert m01.status_indicator == TwinStatusIndicator.NORMAL
    assert test_twin.factory_status == TwinStatusIndicator.NORMAL

    # 2. Warning Update
    warn_telemetry = {
        "timestamp": "2026-09-30T20:01:00+00:00",
        "machine_id": "M01",
        "machine_state": "WARNING",
        "temperature": 82.0,
        "vibration": 5.0,
        "current": 14.0,
        "rpm": 1400.0,
    }
    test_twin.update_machine(
        "M01",
        telemetry=warn_telemetry,
        ai_pred={"anomaly_score": 0.58},
        health_eval={"health_score": 45.0, "condition": "POOR"},
        risk_eval={"risk_score": 68.0, "risk_level": "HIGH", "primary_driver": "Vibration"},
    )
    assert m01.status_indicator == TwinStatusIndicator.WARNING
    assert test_twin.factory_status == TwinStatusIndicator.WARNING

    # 3. Critical / Failure Update
    fail_telemetry = {
        "timestamp": "2026-09-30T20:02:00+00:00",
        "machine_id": "M01",
        "machine_state": "FAILURE",
        "temperature": 95.0,
        "vibration": 7.2,
        "current": 20.0,
        "rpm": 50.0,
    }
    test_twin.update_machine(
        "M01",
        telemetry=fail_telemetry,
        ai_pred={"anomaly_score": 0.98},
        health_eval={"health_score": 10.0, "condition": "CRITICAL"},
        risk_eval={"risk_score": 95.0, "risk_level": "CRITICAL", "primary_driver": "Breakdown"},
    )
    assert m01.status_indicator == TwinStatusIndicator.CRITICAL
    assert test_twin.factory_status == TwinStatusIndicator.CRITICAL


def test_digital_twin_ascii_render(test_twin):
    """Verify ASCII Digital Twin view renders readable representation."""
    rendered = test_twin.render_ascii_view()
    assert "DIGITAL TWIN:" in rendered
    assert "M01:" in rendered
    assert "M02:" in rendered
    assert "M03:" in rendered


def test_fastapi_rest_endpoints(api_client):
    """Verify FastAPI endpoints return valid JSON and HTML."""
    # Health endpoint
    r_health = api_client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "online"

    # Twin snapshot endpoint
    r_twin = api_client.get("/api/twin/state")
    assert r_twin.status_code == 200
    twin_data = r_twin.json()
    assert "machines" in twin_data
    assert "layout_stations" in twin_data

    # Single machine endpoint
    r_m01 = api_client.get("/api/twin/machines/M01")
    assert r_m01.status_code == 200
    assert r_m01.json()["machine_id"] == "M01"

    # Web Dashboard HTML
    r_ui = api_client.get("/")
    assert r_ui.status_code == 200
    assert "text/html" in r_ui.headers["content-type"]
    assert "Smart Factory Digital Twin" in r_ui.text
