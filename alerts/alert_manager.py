"""Industrial Alert Engine for Smart Factory Digital Twin.

Evaluates multi-sensor telemetry, PyTorch AI anomaly detection, health scoring,
and operational risk matrices to generate, deduplicate, rate-limit (cooldown),
and dispatch actionable industrial alerts via MQTT and SQLite.
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from config import load_config
from database.database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AlertEngine")


class AlertSeverity:
    """Standard industrial alert severity classifications."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType:
    """Categorized industrial alert types."""
    AI_ANOMALY = "AI_ANOMALY"
    HEALTH_DEGRADATION = "HEALTH_DEGRADATION"
    HIGH_OPERATIONAL_RISK = "HIGH_OPERATIONAL_RISK"
    SENSOR_THRESHOLD_EXCEEDED = "SENSOR_THRESHOLD_EXCEEDED"
    EQUIPMENT_BREAKDOWN = "EQUIPMENT_BREAKDOWN"
    MAINTENANCE_REQUIRED = "MAINTENANCE_REQUIRED"


class AlertManager:
    """Manages real-time alert generation, deduplication, and multi-channel dispatch."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cooldown_seconds: Optional[float] = None,
    ):
        """Initialize AlertManager with configuration thresholds and cooldown parameters.

        Args:
            config: Optional system configuration dictionary.
            cooldown_seconds: Time in seconds before re-triggering the same alert type per machine.
        """
        self.config: Dict[str, Any] = config or load_config()

        alerts_cfg = self.config.get("alerts", {})
        if cooldown_seconds is not None:
            self.cooldown_seconds = float(cooldown_seconds)
        else:
            self.cooldown_seconds = float(alerts_cfg.get("cooldown_seconds", 10.0))

        self.machines_cfg = {
            m["id"]: m for m in self.config.get("factory", {}).get("machines", [])
        }

        # Cooldown cache: (machine_id, alert_type, severity) -> last_dispatched_timestamp
        self._cooldown_cache: Dict[Tuple[str, str, str], float] = {}

        # Metric counters
        self.total_generated: int = 0
        self.total_dispatched: int = 0
        self.total_suppressed: int = 0

    def evaluate_alerts(
        self,
        telemetry: Dict[str, Any],
        ai_pred: Optional[Dict[str, Any]] = None,
        health_eval: Optional[Dict[str, Any]] = None,
        risk_eval: Optional[Dict[str, Any]] = None,
        maint_diag: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate incoming machine telemetry and analytics for alert conditions.

        Reuses existing analytics computations without duplicating calculations.

        Args:
            telemetry: Raw telemetry dictionary.
            ai_pred: Output from PyTorch AnomalyDetector.
            health_eval: Output from HealthEvaluator.
            risk_eval: Output from OperationalRiskEngine.
            maint_diag: Output from PredictiveMaintenanceEngine.

        Returns:
            List of raw un-filtered alert candidate dictionaries.
        """
        candidates: List[Dict[str, Any]] = []
        machine_id = telemetry.get("machine_id", "M01")
        timestamp = telemetry.get("timestamp", datetime.now(timezone.utc).isoformat())
        machine_state = telemetry.get("machine_state", "RUNNING")
        m_cfg = self.machines_cfg.get(machine_id, {})
        thresh = m_cfg.get("thresholds", {})

        # 1. Equipment Breakdown / Severe State Alerts
        if machine_state == "FAILURE":
            candidates.append({
                "timestamp": timestamp,
                "machine_id": machine_id,
                "alert_type": AlertType.EQUIPMENT_BREAKDOWN,
                "severity": AlertSeverity.CRITICAL,
                "message": f"CRITICAL: Machine {machine_id} has failed (Temp: {telemetry.get('temperature', 0):.1f}°C, Vib: {telemetry.get('vibration', 0):.2f}mm/s). Immediate intervention required.",
                "acknowledged": 0,
            })
        elif machine_state == "WARNING":
            candidates.append({
                "timestamp": timestamp,
                "machine_id": machine_id,
                "alert_type": AlertType.SENSOR_THRESHOLD_EXCEEDED,
                "severity": AlertSeverity.WARNING,
                "message": f"WARNING: Machine {machine_id} physical parameters operating in warning envelope.",
                "acknowledged": 0,
            })

        # 2. PyTorch AI Anomaly Detection Alerts
        if ai_pred:
            anomaly_score = float(ai_pred.get("anomaly_score", 0.0))
            is_anomaly = bool(ai_pred.get("is_anomaly", False))
            if is_anomaly or anomaly_score >= 0.40:
                severity = AlertSeverity.CRITICAL if anomaly_score >= 0.70 else AlertSeverity.WARNING
                rec = ai_pred.get("recommendation", "AI detected abnormal sensor patterns.")
                candidates.append({
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "alert_type": AlertType.AI_ANOMALY,
                    "severity": severity,
                    "message": f"AI Anomaly Detected (Score: {anomaly_score:.3f}): {rec}",
                    "acknowledged": 0,
                })

        # 3. Equipment Health Degradation Alerts
        if health_eval:
            health_score = float(health_eval.get("health_score", 100.0))
            condition = health_eval.get("condition", "EXCELLENT")
            if health_score < 60.0:
                severity = AlertSeverity.CRITICAL if health_score < 35.0 else AlertSeverity.WARNING
                candidates.append({
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "alert_type": AlertType.HEALTH_DEGRADATION,
                    "severity": severity,
                    "message": f"Equipment health degraded to {health_score:.1f}% ({condition}). Accelerated wear observed.",
                    "acknowledged": 0,
                })

        # 4. High Operational Risk Alerts
        if risk_eval:
            risk_level = risk_eval.get("risk_level", "LOW")
            risk_score = float(risk_eval.get("risk_score", 0.0))
            primary_driver = risk_eval.get("primary_driver", "None")
            if risk_level in ("HIGH", "CRITICAL"):
                severity = AlertSeverity.CRITICAL if risk_level == "CRITICAL" else AlertSeverity.WARNING
                candidates.append({
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "alert_type": AlertType.HIGH_OPERATIONAL_RISK,
                    "severity": severity,
                    "message": f"Elevated Operational Risk ({risk_score:.1f}/100 - {risk_level}). Primary Factor: {primary_driver}.",
                    "acknowledged": 0,
                })

        # 5. Prescriptive Maintenance Urgency Alerts
        if maint_diag:
            urgency = maint_diag.get("urgency", "NORMAL")
            component = maint_diag.get("target_component", "General System")
            action = maint_diag.get("action", "MONITOR")
            if urgency in ("HIGH", "CRITICAL", "EMERGENCY"):
                severity = AlertSeverity.CRITICAL if urgency in ("CRITICAL", "EMERGENCY") else AlertSeverity.WARNING
                candidates.append({
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "alert_type": AlertType.MAINTENANCE_REQUIRED,
                    "severity": severity,
                    "message": f"Maintenance Action Needed: {action} on {component} (Urgency: {urgency}).",
                    "acknowledged": 0,
                })

        # 6. Physical Threshold Stress Checks (Specific channel spikes)
        temp = float(telemetry.get("temperature", 0.0))
        vib = float(telemetry.get("vibration", 0.0))
        fail_temp = thresh.get("temperature", {}).get("failure", 90.0)
        fail_vib = thresh.get("vibration", {}).get("failure", 6.5)

        if temp >= fail_temp:
            candidates.append({
                "timestamp": timestamp,
                "machine_id": machine_id,
                "alert_type": AlertType.SENSOR_THRESHOLD_EXCEEDED,
                "severity": AlertSeverity.CRITICAL,
                "message": f"Thermal Limit Exceeded: Temperature reached {temp:.1f}°C (Limit: {fail_temp}°C).",
                "acknowledged": 0,
            })
        if vib >= fail_vib:
            candidates.append({
                "timestamp": timestamp,
                "machine_id": machine_id,
                "alert_type": AlertType.SENSOR_THRESHOLD_EXCEEDED,
                "severity": AlertSeverity.CRITICAL,
                "message": f"Vibration Limit Exceeded: Vibration reached {vib:.2f}mm/s (Limit: {fail_vib}mm/s).",
                "acknowledged": 0,
            })

        self.total_generated += len(candidates)
        return candidates

    def filter_with_cooldown(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out duplicate alerts that occurred within the cooldown window.

        Args:
            alerts: List of raw candidate alerts.

        Returns:
            Filtered list of deduplicated alerts eligible for immediate dispatch.
        """
        now = time.time()
        dispatched_batch: List[Dict[str, Any]] = []

        for alert in alerts:
            key = (alert["machine_id"], alert["alert_type"], alert["severity"])
            last_sent = self._cooldown_cache.get(key, 0.0)

            if (now - last_sent) >= self.cooldown_seconds:
                self._cooldown_cache[key] = now
                dispatched_batch.append(alert)
                self.total_dispatched += 1
            else:
                self.total_suppressed += 1
                logger.debug(
                    "Alert suppressed due to cooldown (%.1fs remaining): %s",
                    self.cooldown_seconds - (now - last_sent),
                    key,
                )

        return dispatched_batch

    def dispatch_alerts(
        self,
        alerts: List[Dict[str, Any]],
        db: Optional[DatabaseManager] = None,
        publisher: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Persist alerts to SQLite and publish to MQTT.

        Args:
            alerts: List of deduplicated alerts.
            db: Optional DatabaseManager instance.
            publisher: Optional FactoryMqttPublisher instance.

        Returns:
            List of successfully dispatched alert records with generated DB IDs.
        """
        dispatched_records = []
        for alert in alerts:
            alert_copy = dict(alert)

            # Persist to SQLite
            if db:
                try:
                    db_id = db.insert_alert(alert_copy)
                    alert_copy["id"] = db_id
                except Exception as e:
                    logger.error("Failed to persist alert to SQLite: %s", e)

            # Publish via MQTT
            if publisher and getattr(publisher, "is_connected", False):
                try:
                    topic = "factory/alerts"
                    publisher.client.publish(
                        topic,
                        payload=json.dumps(alert_copy),
                        qos=1,
                    )
                except Exception as e:
                    logger.error("Failed to publish alert to MQTT topic %s: %s", topic, e)

            logger.info(
                "[%s ALERT] Machine: %s | Type: %s | %s",
                alert_copy.get("severity", "INFO"),
                alert_copy.get("machine_id", "N/A"),
                alert_copy.get("alert_type", "GENERIC"),
                alert_copy.get("message", ""),
            )
            dispatched_records.append(alert_copy)

        return dispatched_records

    def evaluate_and_dispatch(
        self,
        telemetry: Dict[str, Any],
        ai_pred: Optional[Dict[str, Any]] = None,
        health_eval: Optional[Dict[str, Any]] = None,
        risk_eval: Optional[Dict[str, Any]] = None,
        maint_diag: Optional[Dict[str, Any]] = None,
        db: Optional[DatabaseManager] = None,
        publisher: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """End-to-end pipeline method: evaluate, filter by cooldown, persist to SQLite, and publish to MQTT."""
        raw_alerts = self.evaluate_alerts(
            telemetry=telemetry,
            ai_pred=ai_pred,
            health_eval=health_eval,
            risk_eval=risk_eval,
            maint_diag=maint_diag,
        )
        filtered = self.filter_with_cooldown(raw_alerts)
        return self.dispatch_alerts(filtered, db=db, publisher=publisher)

    def reset_cooldown(self, machine_id: Optional[str] = None):
        """Reset the cooldown cache (useful for testing or manual operator override)."""
        if machine_id:
            keys_to_delete = [k for k in self._cooldown_cache if k[0] == machine_id]
            for k in keys_to_delete:
                del self._cooldown_cache[k]
        else:
            self._cooldown_cache.clear()
