"""Predictive maintenance and prescriptive recommendation engine."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from database.database import DatabaseManager


class MaintenanceUrgency:
    NORMAL = "NORMAL"
    SCHEDULED = "SCHEDULED"
    PREVENTIVE_WARNING = "PREVENTIVE_WARNING"
    IMMEDIATE_ACTION = "IMMEDIATE_ACTION"
    EMERGENCY_OVERHAUL = "EMERGENCY_OVERHAUL"


class PredictiveMaintenanceEngine:
    """Diagnoses degrading industrial subsystems and generates prescriptive maintenance recommendations."""

    def diagnose_and_recommend(
        self,
        telemetry: Dict[str, Any],
        health_eval: Dict[str, Any],
        risk_eval: Dict[str, Any],
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate targeted prescriptive maintenance guidance based on sensor metrics and trends.

        Args:
            telemetry: Current machine telemetry dictionary.
            health_eval: Result dictionary from HealthEvaluator.
            risk_eval: Result dictionary from OperationalRiskEngine.
            telemetry_history: Optional historical telemetry list to detect degradation slopes.

        Returns:
            Dictionary containing urgency, target_component, recommendation, and trend metrics.
        """
        m_id = telemetry.get("machine_id", "M01")
        machine_state = telemetry.get("machine_state", "RUNNING")
        health_score = float(health_eval.get("health_score", 100.0))
        risk_level = risk_eval.get("risk_level", "LOW")
        subscores = health_eval.get("sensor_subscores", {})

        # 1. Evaluate historical trend slope if available
        trend_signals = []
        if telemetry_history and len(telemetry_history) >= 3:
            temps = [float(h["temperature"]) for h in telemetry_history]
            vibs = [float(h["vibration"]) for h in telemetry_history]
            currs = [float(h["current"]) for h in telemetry_history]
            rpms = [float(h["rpm"]) for h in telemetry_history]

            if temps[-1] - temps[0] > 4.0:
                trend_signals.append("Temperature trending upward (thermal accumulation)")
            if vibs[-1] - vibs[0] > 0.8:
                trend_signals.append("Vibration trending upward (bearing wear)")
            if currs[-1] - currs[0] > 1.5:
                trend_signals.append("Motor current trending upward (mechanical binding)")
            if rpms[0] - rpms[-1] > 100.0:
                trend_signals.append("RPM trending downward (feed slowdown)")

        # 2. Identify target failing component
        worst_sensor = min(subscores, key=subscores.get) if subscores else "general"

        component_map = {
            "vibration": ("Bearings & Drive Shaft", "Mechanical imbalance or race fatigue"),
            "temperature": ("Cooling & Lubrication System", "Heat dissipation deficit or lubricant breakdown"),
            "current": ("Motor Stator / Windings", "Winding degradation or high torque drag"),
            "rpm": ("Transmission / Spindle Drive", "Belt slip or spindle friction overload"),
            "pressure": ("Hydraulic / Pneumatic Circuit", "Pressure regulator seal wear or minor leak"),
            "power": ("Power Inverter / Drive Module", "Electrical conversion loss"),
        }

        target_component, root_cause = component_map.get(
            worst_sensor, ("General Mechanical System", "Multi-parameter wear")
        )

        # 3. Urgency and Prescriptive Action
        if machine_state == "FAILURE" or health_score < 25.0:
            urgency = MaintenanceUrgency.EMERGENCY_OVERHAUL
            action = f"EMERGENCY OVERHAUL: Machine {m_id} breakdown. Lockout/Tagout equipment immediately. Replace {target_component}. Root cause: {root_cause}."
        elif risk_level == "CRITICAL" or health_score < 40.0:
            urgency = MaintenanceUrgency.IMMEDIATE_ACTION
            action = f"IMMEDIATE ACTION: Halt machine {m_id} within 1 hour. Perform inspection on {target_component}. Root cause: {root_cause}."
        elif risk_level == "HIGH" or health_score < 60.0:
            urgency = MaintenanceUrgency.PREVENTIVE_WARNING
            action = f"PREVENTIVE WARNING: Machine {m_id} degradation detected. Schedule maintenance inspection on {target_component} within 24 hours."
        elif risk_level == "MEDIUM" or health_score < 80.0:
            urgency = MaintenanceUrgency.SCHEDULED
            action = f"SCHEDULED MAINTENANCE: Minor wear on {m_id}. Inspect {target_component} during next planned shift turnaround."
        else:
            urgency = MaintenanceUrgency.NORMAL
            action = f"Machine {m_id} is healthy. Standard operating parameters observed."

        return {
            "timestamp": telemetry.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "machine_id": m_id,
            "urgency": urgency,
            "target_component": target_component,
            "root_cause_hypothesis": root_cause,
            "recommendation": action,
            "trend_signals": trend_signals,
            "health_score": health_score,
            "risk_level": risk_level,
        }

    def log_recommendation_to_db(
        self, rec: Dict[str, Any], db: DatabaseManager
    ) -> Optional[int]:
        """Log maintenance action into SQLite maintenance_events table if actionable."""
        if rec["urgency"] == MaintenanceUrgency.NORMAL:
            return None

        event_payload = {
            "timestamp": rec["timestamp"],
            "machine_id": rec["machine_id"],
            "event_type": rec["urgency"],
            "description": f"[{rec['target_component']}] {rec['recommendation']}",
            "downtime_minutes": 0.0,
            "technician": "AI Predictive System",
        }
        return db.insert_maintenance_event(event_payload)
