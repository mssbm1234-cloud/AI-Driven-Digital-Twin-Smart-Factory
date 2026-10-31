"""Inference engine combining PyTorch Deep Autoencoder and Rule-Based Industrial Health Scoring."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from config import load_config
from database.database import DatabaseManager
from ai.preprocessing import TelemetryPreprocessor
from ai.anomaly_model import TelemetryAutoencoder
from ai.train import MODEL_PATH, PREPROCESSOR_PATH, train_anomaly_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AIInference")


class AnomalyDetector:
    """Predicts machine anomaly scores, health scores, and operational risk levels."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize detector, loading trained PyTorch model and weights.

        If model files do not exist, automatically trains and persists them.
        """
        self.config: Dict[str, Any] = config or load_config()
        self.model_path = model_path or MODEL_PATH
        self.preprocessor_path = preprocessor_path or PREPROCESSOR_PATH

        # Ensure model exists
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            logger.info("Model checkpoint not found. Triggering automated model training...")
            self.model, self.preprocessor, self.threshold = train_anomaly_model()
        else:
            self._load_artifacts()

        # Health weights from config
        self.weights = self.config.get("health_weights", {
            "temperature": 0.20,
            "vibration": 0.25,
            "current": 0.15,
            "rpm": 0.15,
            "pressure": 0.10,
            "power": 0.15,
        })

    def _load_artifacts(self):
        """Load trained weights, threshold, and preprocessor from disk."""
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"), weights_only=True)
        self.threshold = float(checkpoint.get("threshold", 0.025))
        input_dim = int(checkpoint.get("input_dim", 6))
        latent_dim = int(checkpoint.get("latent_dim", 4))

        self.model = TelemetryAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.preprocessor = TelemetryPreprocessor.load(self.preprocessor_path)
        logger.info("Loaded PyTorch Anomaly Detector (Threshold: %.6f)", self.threshold)

    def calculate_rule_based_health(
        self, telemetry: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate baseline health score (0-100) using physical sensor thresholds.

        Returns:
            Tuple of (overall_rule_health, individual_sensor_health_scores).
        """
        machine_id = telemetry.get("machine_id", "M01")
        # Find machine config
        machines = self.config.get("factory", {}).get("machines", [])
        m_cfg = next((m for m in machines if m["id"] == machine_id), {})
        nom = m_cfg.get("nominal", {})
        thresh = m_cfg.get("thresholds", {})

        sub_scores = {}

        # 1. Temperature score
        cur_temp = telemetry["temperature"]
        warn_temp = thresh.get("temperature", {}).get("warning", 80.0)
        fail_temp = thresh.get("temperature", {}).get("failure", 95.0)
        if cur_temp <= nom.get("temperature", 65.0):
            sub_scores["temperature"] = 100.0
        elif cur_temp >= fail_temp:
            sub_scores["temperature"] = 0.0
        else:
            sub_scores["temperature"] = max(0.0, 100.0 - ((cur_temp - nom.get("temperature", 65.0)) / (fail_temp - nom.get("temperature", 65.0))) * 100.0)

        # 2. Vibration score
        cur_vib = telemetry["vibration"]
        warn_vib = thresh.get("vibration", {}).get("warning", 4.2)
        fail_vib = thresh.get("vibration", {}).get("failure", 6.5)
        if cur_vib <= nom.get("vibration", 2.5):
            sub_scores["vibration"] = 100.0
        elif cur_vib >= fail_vib:
            sub_scores["vibration"] = 0.0
        else:
            sub_scores["vibration"] = max(0.0, 100.0 - ((cur_vib - nom.get("vibration", 2.5)) / (fail_vib - nom.get("vibration", 2.5))) * 100.0)

        # 3. Current score
        cur_curr = telemetry["current"]
        warn_curr = thresh.get("current", {}).get("warning", 14.0)
        fail_curr = thresh.get("current", {}).get("failure", 18.0)
        if cur_curr <= nom.get("current", 8.5):
            sub_scores["current"] = 100.0
        elif cur_curr >= fail_curr:
            sub_scores["current"] = 0.0
        else:
            sub_scores["current"] = max(0.0, 100.0 - ((cur_curr - nom.get("current", 8.5)) / (fail_curr - nom.get("current", 8.5))) * 100.0)

        # 4. RPM score
        cur_rpm = telemetry["rpm"]
        nom_rpm = nom.get("rpm", 1800.0)
        warn_rpm = thresh.get("rpm_min", {}).get("warning", 1500.0)
        fail_rpm = thresh.get("rpm_min", {}).get("failure", 1000.0)
        if cur_rpm >= nom_rpm:
            sub_scores["rpm"] = 100.0
        elif cur_rpm <= fail_rpm:
            sub_scores["rpm"] = 0.0
        else:
            sub_scores["rpm"] = max(0.0, ((cur_rpm - fail_rpm) / (nom_rpm - fail_rpm)) * 100.0)

        # 5. Pressure score
        nom_press = nom.get("pressure", 6.0)
        press_dev = abs(telemetry["pressure"] - nom_press)
        sub_scores["pressure"] = max(0.0, 100.0 - (press_dev / (nom_press * 0.40)) * 100.0)

        # 6. Power score
        nom_power = nom.get("power", 3.2)
        power_dev = abs(telemetry["power"] - nom_power)
        sub_scores["power"] = max(0.0, 100.0 - (power_dev / (nom_power * 0.50)) * 100.0)

        # Weighted combination
        rule_health = sum(sub_scores[k] * self.weights.get(k, 0.15) for k in sub_scores)
        return float(np.clip(rule_health, 0.0, 100.0)), sub_scores

    def predict(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive AI anomaly detection and health risk evaluation.

        Args:
            telemetry: Raw machine telemetry dictionary.

        Returns:
            Dictionary containing anomaly_score, health_score, risk_level, and recommendation.
        """
        # 1. Feature extraction & normalization
        raw_features = self.preprocessor.extract_features(telemetry)
        norm_features = self.preprocessor.transform(raw_features)

        # 2. PyTorch Autoencoder forward pass
        tensor_in = torch.tensor(norm_features, dtype=torch.float32)
        with torch.no_grad():
            raw_mse = float(self.model.compute_reconstruction_error(tensor_in).item())

        # Scale reconstruction error to normalized anomaly score [0.0, 1.0]
        # Scaling calibrated such that threshold is ~0.40 anomaly score
        anomaly_score = float(np.clip(raw_mse / (self.threshold * 2.5), 0.0, 1.0))

        # 3. Rule-based physical health computation
        rule_health, sub_scores = self.calculate_rule_based_health(telemetry)

        # 4. Fusion of Rule Health (50%) + AI Reconstruction Health (50%)
        ai_health = (1.0 - anomaly_score) * 100.0
        composite_health = 0.50 * rule_health + 0.50 * ai_health

        # Severe state overrides
        machine_state = telemetry.get("machine_state", "RUNNING")
        if machine_state == "FAILURE":
            composite_health = min(composite_health, 10.0)
            anomaly_score = max(anomaly_score, 0.95)
        elif machine_state == "WARNING":
            composite_health = min(composite_health, 55.0)
            anomaly_score = max(anomaly_score, 0.55)

        # 5. Risk classification
        if composite_health >= 80.0 and anomaly_score < 0.35:
            risk_level = "LOW"
            is_anomaly = False
        elif composite_health >= 60.0 and anomaly_score < 0.60:
            risk_level = "MEDIUM"
            is_anomaly = False
        elif composite_health >= 35.0:
            risk_level = "HIGH"
            is_anomaly = True
        else:
            risk_level = "CRITICAL"
            is_anomaly = True

        # 6. Prescriptive Maintenance Recommendation
        recommendation = self._generate_recommendation(telemetry, sub_scores, risk_level)

        return {
            "timestamp": telemetry.get("timestamp"),
            "machine_id": telemetry.get("machine_id", "M01"),
            "anomaly_score": round(anomaly_score, 4),
            "health_score": round(composite_health, 1),
            "risk_level": risk_level,
            "is_anomaly": int(is_anomaly),
            "recommendation": recommendation,
        }

    def _generate_recommendation(
        self, telemetry: Dict[str, Any], sub_scores: Dict[str, float], risk_level: str
    ) -> str:
        """Generate actionable maintenance guidance based on lowest scoring physical subsystem."""
        if risk_level == "LOW":
            return "Equipment operating normally within standard physical tolerances."

        # Find worst performing sensor
        worst_metric = min(sub_scores, key=sub_scores.get)

        advice_map = {
            "vibration": "Elevated vibration detected. Inspect mechanical couplings, bearings, and shaft alignment.",
            "temperature": "Excessive operating temperature. Check cooling coolant lines, ventilation, and lubrication.",
            "current": "Abnormal motor current draw. Verify electrical windings and mechanical load resistance.",
            "rpm": "Significant RPM drop. Inspect drive belts, tool feed resistance, and spindle drive.",
            "pressure": "Hydraulic/pneumatic pressure deviation. Check for line leaks or regulator valve wear.",
            "power": "Power consumption anomaly. Inspect motor drive efficiency and electrical supply.",
        }

        specific_advice = advice_map.get(worst_metric, "Perform routine diagnostic inspection.")
        if risk_level == "CRITICAL":
            return f"CRITICAL HAZARD: Immediate machine shutdown and overhaul recommended. {specific_advice}"
        elif risk_level == "HIGH":
            return f"HIGH RISK: Schedule preventive maintenance within 24h. {specific_advice}"
        else:
            return f"ATTENTION: Early degradation symptoms detected. {specific_advice}"

    def predict_and_store(
        self, telemetry: Dict[str, Any], db: DatabaseManager
    ) -> Dict[str, Any]:
        """Infer anomaly/health and persist to SQLite ai_predictions table."""
        pred = self.predict(telemetry)
        db.insert_ai_prediction(pred)
        return pred
