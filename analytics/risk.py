"""Operational risk analysis engine for smart factory equipment."""

from typing import Any, Dict, List, Optional
import numpy as np
from config import load_config


class RiskLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OperationalRiskEngine:
    """Evaluates multi-dimensional operational risk based on health, anomalies, and production impact."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk engine with configuration thresholds."""
        self.config = config or load_config()
        self.risk_cfg = self.config.get("risk_thresholds", {
            "low_max": 0.30,
            "medium_max": 0.60,
            "high_max": 0.85,
        })
        self.machines_cfg = {
            m["id"]: m for m in self.config.get("factory", {}).get("machines", [])
        }

    def evaluate_risk(
        self,
        telemetry: Dict[str, Any],
        health_eval: Dict[str, Any],
        ai_anomaly_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute operational risk score, risk level, and primary risk driver.

        Args:
            telemetry: Raw telemetry dictionary.
            health_eval: Health dictionary from HealthEvaluator.
            ai_anomaly_score: Float anomaly score (0.0 to 1.0) from PyTorch model.

        Returns:
            Dictionary containing risk_score (0-100), risk_level, and driver breakdown.
        """
        m_id = telemetry.get("machine_id", "M01")
        m_cfg = self.machines_cfg.get(m_id, {})
        thresh = m_cfg.get("thresholds", {})
        nom = m_cfg.get("nominal", {})

        # 1. Health Deficit Risk (0 - 100)
        health_score = float(health_eval.get("health_score", 100.0))
        health_risk = 100.0 - health_score

        # 2. AI Anomaly Risk (0 - 100)
        anomaly_risk = float(np.clip(ai_anomaly_score * 100.0, 0.0, 100.0))

        # 3. Extreme Physical Stress Risk (0 - 100)
        cur_temp = float(telemetry.get("temperature", 65.0))
        cur_vib = float(telemetry.get("vibration", 2.5))
        fail_temp = thresh.get("temperature", {}).get("failure", 90.0)
        fail_vib = thresh.get("vibration", {}).get("failure", 6.5)

        temp_stress = max(0.0, min(100.0, (cur_temp - nom.get("temperature", 65.0)) / (fail_temp - nom.get("temperature", 65.0)) * 100.0))
        vib_stress = max(0.0, min(100.0, (cur_vib - nom.get("vibration", 2.5)) / (fail_vib - nom.get("vibration", 2.5)) * 100.0))
        stress_risk = max(temp_stress, vib_stress)

        # 4. Production Defect / Quality Risk (0 - 100)
        defect_rate = float(telemetry.get("defect_rate", 0.0))
        quality_risk = min(100.0, defect_rate * 300.0)

        # 5. Downtime Risk (0 - 100)
        downtime = float(telemetry.get("downtime_seconds", 0.0))
        downtime_risk = min(100.0, downtime * 5.0)

        # Multi-factor weighted operational risk score
        risk_score = (
            0.35 * health_risk
            + 0.25 * anomaly_risk
            + 0.20 * stress_risk
            + 0.10 * quality_risk
            + 0.10 * downtime_risk
        )

        # State overrides
        machine_state = telemetry.get("machine_state", "RUNNING")
        if machine_state == "FAILURE":
            risk_score = max(risk_score, 92.0)
        elif machine_state == "WARNING":
            risk_score = max(risk_score, 62.0)

        risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 1)

        # Classify Level
        if risk_score < (self.risk_cfg["low_max"] * 100):
            risk_level = RiskLevel.LOW
        elif risk_score < (self.risk_cfg["medium_max"] * 100):
            risk_level = RiskLevel.MEDIUM
        elif risk_score < (self.risk_cfg["high_max"] * 100):
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # Identify Primary Risk Driver
        driver_candidates = {
            "Mechanical Vibration Stress": vib_stress,
            "Thermal Overheat Stress": temp_stress,
            "AI Detected Anomaly": anomaly_risk,
            "Equipment Health Degradation": health_risk,
            "Defect Rate Quality Loss": quality_risk,
            "Equipment Stoppage / Downtime": downtime_risk,
        }
        primary_driver = max(driver_candidates, key=driver_candidates.get)

        return {
            "machine_id": m_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "primary_driver": primary_driver if risk_score >= 30.0 else "None (Normal Operation)",
            "risk_breakdown": {
                "health_risk": round(health_risk, 1),
                "anomaly_risk": round(anomaly_risk, 1),
                "stress_risk": round(stress_risk, 1),
                "quality_risk": round(quality_risk, 1),
                "downtime_risk": round(downtime_risk, 1),
            },
        }
