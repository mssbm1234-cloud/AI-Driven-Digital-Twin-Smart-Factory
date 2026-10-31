"""Unit and integration tests for Phase 6: AI/ML Anomaly Detection using PyTorch."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database import DatabaseManager
from ai.preprocessing import TelemetryPreprocessor
from ai.anomaly_model import TelemetryAutoencoder
from ai.inference import AnomalyDetector


@pytest.fixture
def detector():
    return AnomalyDetector()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "ai_test_factory.db"
    return DatabaseManager(db_path=str(db_file))


def test_preprocessor_features():
    """Verify preprocessor extracts ordered features and scales correctly."""
    prep = TelemetryPreprocessor()
    telemetry = {
        "temperature": 65.0,
        "vibration": 2.5,
        "current": 8.5,
        "rpm": 1800.0,
        "pressure": 6.0,
        "power": 3.2,
    }
    features = prep.extract_features(telemetry)
    assert features.shape == (6,)
    assert features.dtype.name == "float32"

    normalized = prep.transform(features)
    assert normalized.shape == (6,)

    reconstructed = prep.inverse_transform(normalized)
    assert pytest.approx(reconstructed[0], abs=1e-3) == 65.0
    assert pytest.approx(reconstructed[3], abs=1e-3) == 1800.0


def test_autoencoder_architecture():
    """Verify PyTorch autoencoder forward pass and reconstruction loss calculation."""
    model = TelemetryAutoencoder(input_dim=6, latent_dim=4)
    x = torch.randn(8, 6)
    out = model(x)
    assert out.shape == (8, 6)

    err = model.compute_reconstruction_error(x)
    assert err.shape == (8,)
    assert (err >= 0.0).all()


def test_normal_telemetry_inference(detector):
    """Verify healthy baseline telemetry produces high health score and LOW risk."""
    normal_telemetry = {
        "timestamp": "2026-09-30T19:00:00+00:00",
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.1,
        "vibration": 2.51,
        "current": 8.52,
        "voltage": 230.1,
        "rpm": 1798.0,
        "pressure": 6.01,
        "power": 3.21,
    }

    pred = detector.predict(normal_telemetry)
    assert pred["machine_id"] == "M01"
    assert pred["anomaly_score"] < 0.35
    assert pred["health_score"] >= 80.0
    assert pred["risk_level"] == "LOW"
    assert pred["is_anomaly"] == 0
    assert "operating normally" in pred["recommendation"]


def test_degraded_telemetry_inference(detector):
    """Verify degraded telemetry with elevated heat and vibration triggers HIGH or CRITICAL risk."""
    degraded_telemetry = {
        "timestamp": "2026-09-30T19:05:00+00:00",
        "machine_id": "M01",
        "machine_state": "WARNING",
        "temperature": 86.5,     # High temperature (nominal 65.0)
        "vibration": 5.8,       # High vibration (nominal 2.5)
        "current": 13.8,        # High current (nominal 8.5)
        "voltage": 228.0,
        "rpm": 1350.0,          # Reduced RPM (nominal 1800.0)
        "pressure": 5.7,
        "power": 4.5,
    }

    pred = detector.predict(degraded_telemetry)
    assert pred["machine_id"] == "M01"
    assert pred["anomaly_score"] > 0.40
    assert pred["health_score"] < 60.0
    assert pred["risk_level"] in ("HIGH", "CRITICAL")
    assert pred["is_anomaly"] == 1
    assert "vibration" in pred["recommendation"].lower() or "temperature" in pred["recommendation"].lower()


def test_failure_telemetry_inference(detector):
    """Verify catastrophic machine failure drops health to critical minimum."""
    failure_telemetry = {
        "timestamp": "2026-09-30T19:10:00+00:00",
        "machine_id": "M02",
        "machine_state": "FAILURE",
        "temperature": 98.0,
        "vibration": 7.5,
        "current": 22.0,
        "voltage": 220.0,
        "rpm": 50.0,
        "pressure": 3.0,
        "power": 0.8,
    }

    pred = detector.predict(failure_telemetry)
    assert pred["machine_id"] == "M02"
    assert pred["health_score"] <= 10.0
    assert pred["risk_level"] == "CRITICAL"
    assert pred["is_anomaly"] == 1


def test_predict_and_store_in_sqlite(detector, temp_db):
    """Verify AI predictions are inserted and queryable from SQLite ai_predictions table."""
    telemetry = {
        "timestamp": "2026-09-30T19:15:00+00:00",
        "machine_id": "M03",
        "machine_state": "RUNNING",
        "temperature": 75.2,
        "vibration": 1.82,
        "current": 15.1,
        "voltage": 230.0,
        "rpm": 948.0,
        "pressure": 5.02,
        "power": 4.82,
    }

    pred = detector.predict_and_store(telemetry, temp_db)
    assert "health_score" in pred

    with temp_db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM ai_predictions WHERE machine_id = 'M03';")
        rows = cursor.fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["machine_id"] == "M03"
        assert r["risk_level"] == pred["risk_level"]
        assert pytest.approx(r["health_score"], abs=0.1) == pred["health_score"]
        assert pytest.approx(r["anomaly_score"], abs=0.01) == pred["anomaly_score"]
