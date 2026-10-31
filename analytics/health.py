"""Machine health evaluation engine for industrial telemetry."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from config import load_config


class HealthCondition:
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


class HealthEvaluator:
    """Calculates comprehensive multi-factor machine health scores and conditions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize health evaluator with configuration weights."""
        self.config = config or load_config()
        self.weights = self.config.get("health_weights", {
            "temperature": 0.20,
            "vibration": 0.25,
            "current": 0.15,
            "rpm": 0.15,
            "pressure": 0.10,
            "power": 0.15,
        })
        self.machines_cfg = {
            m["id"]: m for m in self.config.get("factory", {}).get("machines", [])
        }

    def evaluate_sensor_subscores(
        self, telemetry: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute 0-100 health subscores for individual sensor channels."""
        m_id = telemetry.get("machine_id", "M01")
        m_cfg = self.machines_cfg.get(m_id, {})
        nom = m_cfg.get("nominal", {})
        thresh = m_cfg.get("thresholds", {})

        scores = {}

        # 1. Temperature Subscore
        cur_temp = float(telemetry.get("temperature", 65.0))
        nom_temp = nom.get("temperature", 65.0)
        warn_temp = thresh.get("temperature", {}).get("warning", 80.0)
        fail_temp = thresh.get("temperature", {}).get("failure", 95.0)
        if cur_temp <= nom_temp:
            scores["temperature"] = 100.0
        elif cur_temp >= fail_temp:
            scores["temperature"] = 0.0
        else:
            scores["temperature"] = max(0.0, 100.0 - ((cur_temp - nom_temp) / (fail_temp - nom_temp)) * 100.0)

        # 2. Vibration Subscore
        cur_vib = float(telemetry.get("vibration", 2.5))
        nom_vib = nom.get("vibration", 2.5)
        warn_vib = thresh.get("vibration", {}).get("warning", 4.2)
        fail_vib = thresh.get("vibration", {}).get("failure", 6.5)
        if cur_vib <= nom_vib:
            scores["vibration"] = 100.0
        elif cur_vib >= fail_vib:
            scores["vibration"] = 0.0
        else:
            scores["vibration"] = max(0.0, 100.0 - ((cur_vib - nom_vib) / (fail_vib - nom_vib)) * 100.0)

        # 3. Current Subscore
        cur_curr = float(telemetry.get("current", 8.5))
        nom_curr = nom.get("current", 8.5)
        warn_curr = thresh.get("current", {}).get("warning", 14.0)
        fail_curr = thresh.get("current", {}).get("failure", 18.0)
        if cur_curr <= nom_curr:
            scores["current"] = 100.0
        elif cur_curr >= fail_curr:
            scores["current"] = 0.0
        else:
            scores["current"] = max(0.0, 100.0 - ((cur_curr - nom_curr) / (fail_curr - nom_curr)) * 100.0)

        # 4. RPM Subscore
        cur_rpm = float(telemetry.get("rpm", 1800.0))
        nom_rpm = nom.get("rpm", 1800.0)
        fail_rpm = thresh.get("rpm_min", {}).get("failure", 1000.0)
        if cur_rpm >= nom_rpm:
            scores["rpm"] = 100.0
        elif cur_rpm <= fail_rpm:
            scores["rpm"] = 0.0
        else:
            scores["rpm"] = max(0.0, ((cur_rpm - fail_rpm) / (nom_rpm - fail_rpm)) * 100.0)

        # 5. Pressure Subscore
        cur_press = float(telemetry.get("pressure", 6.0))
        nom_press = nom.get("pressure", 6.0)
        press_diff = abs(cur_press - nom_press)
        scores["pressure"] = max(0.0, 100.0 - (press_diff / (nom_press * 0.40)) * 100.0)

        # 6. Power Subscore
        cur_power = float(telemetry.get("power", 3.2))
        nom_power = nom.get("power", 3.2)
        power_diff = abs(cur_power - nom_power)
        scores["power"] = max(0.0, 100.0 - (power_diff / (nom_power * 0.50)) * 100.0)

        return {k: round(v, 2) for k, v in scores.items()}

    def evaluate_health(
        self,
        telemetry: Dict[str, Any],
        ai_anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute composite machine health score and condition assessment.

        Args:
            telemetry: Machine telemetry dictionary.
            ai_anomaly_score: Optional anomaly score (0.0 to 1.0) from PyTorch model.

        Returns:
            Dictionary containing health_score, condition, rule_health, ai_health, and subscores.
        """
        subscores = self.evaluate_sensor_subscores(telemetry)

        # 1. Rule-based weighted sum
        rule_health = sum(subscores[k] * self.weights.get(k, 0.15) for k in subscores)
        rule_health = float(np.clip(rule_health, 0.0, 100.0))

        # 2. AI Anomaly Health Contribution
        if ai_anomaly_score is not None:
            ai_health = float(np.clip((1.0 - ai_anomaly_score) * 100.0, 0.0, 100.0))
            composite_health = 0.50 * rule_health + 0.50 * ai_health
        else:
            ai_health = None
            composite_health = rule_health

        # 3. State-based adjustments
        machine_state = telemetry.get("machine_state", "RUNNING")
        if machine_state == "FAILURE":
            composite_health = min(composite_health, 10.0)
        elif machine_state == "WARNING":
            composite_health = min(composite_health, 58.0)
        elif machine_state == "MAINTENANCE":
            composite_health = max(composite_health, 70.0)

        composite_health = round(float(composite_health), 1)

        # 4. Condition classification
        if composite_health >= 90.0:
            condition = HealthCondition.EXCELLENT
        elif composite_health >= 75.0:
            condition = HealthCondition.GOOD
        elif composite_health >= 55.0:
            condition = HealthCondition.FAIR
        elif composite_health >= 35.0:
            condition = HealthCondition.POOR
        else:
            condition = HealthCondition.CRITICAL

        return {
            "machine_id": telemetry.get("machine_id", "M01"),
            "health_score": composite_health,
            "condition": condition,
            "rule_health": round(rule_health, 1),
            "ai_health": round(ai_health, 1) if ai_health is not None else None,
            "sensor_subscores": subscores,
            "degradation_level": float(telemetry.get("degradation_level", 0.0)),
        }
