"""Digital Twin state synchronization engine for Smart Factory operations.

Maintains in-memory real-time virtual representations of the physical factory line,
machines, conveyor routing, health indicators, risk levels, and production metrics.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
from config import load_config


class TwinStatusIndicator:
    """Visual status badges for the Digital Twin."""
    NORMAL = "🟢 NORMAL"
    WARNING = "🟡 WARNING"
    CRITICAL = "🔴 CRITICAL"
    OFFLINE = "⚫ OFFLINE"
    MAINTENANCE = "🔵 MAINTENANCE"


class MachineTwin:
    """Virtual representation of a single industrial machine."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize machine twin model."""
        self.machine_id: str = config.get("id", "M00")
        self.name: str = config.get("name", "Machine")
        self.machine_type: str = config.get("type", "generic")
        self.nominal: Dict[str, float] = config.get("nominal", {})

        # Operational State
        self.state: str = "OFFLINE"
        self.status_indicator: str = TwinStatusIndicator.OFFLINE
        self.last_updated: str = datetime.now(timezone.utc).isoformat()
        self.is_online: bool = False

        # Live Sensor Telemetry
        self.telemetry: Dict[str, Any] = {
            "temperature": self.nominal.get("temperature", 65.0),
            "vibration": self.nominal.get("vibration", 2.5),
            "current": self.nominal.get("current", 8.5),
            "voltage": self.nominal.get("voltage", 230.0),
            "rpm": self.nominal.get("rpm", 1800.0),
            "pressure": self.nominal.get("pressure", 6.0),
            "power": self.nominal.get("power", 3.2),
            "cycle_time": self.nominal.get("cycle_time", 12.0),
            "production_count": 0,
            "good_products": 0,
            "defect_count": 0,
            "defect_rate": 0.0,
            "downtime_seconds": 0.0,
            "degradation_level": 0.0,
        }

        # Analytics & Health Twin State
        self.health_score: float = 100.0
        self.condition: str = "EXCELLENT"
        self.anomaly_score: float = 0.0
        self.risk_score: float = 0.0
        self.risk_level: str = "LOW"
        self.primary_risk_driver: str = "None (Normal Operation)"
        self.maintenance_urgency: str = "NORMAL"
        self.maintenance_recommendation: str = "Operating within normal tolerances."
        self.target_component: str = "General System"
        self.oee_metrics: Dict[str, float] = {
            "availability": 100.0,
            "performance": 100.0,
            "quality": 100.0,
            "oee": 100.0,
        }

    def update(
        self,
        telemetry: Dict[str, Any],
        ai_pred: Optional[Dict[str, Any]] = None,
        health_eval: Optional[Dict[str, Any]] = None,
        risk_eval: Optional[Dict[str, Any]] = None,
        maint_diag: Optional[Dict[str, Any]] = None,
        oee_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update machine virtual twin with live telemetry and analytics."""
        self.is_online = True
        self.last_updated = telemetry.get("timestamp", datetime.now(timezone.utc).isoformat())
        self.state = telemetry.get("machine_state", "RUNNING")
        self.telemetry.update(telemetry)

        # Update AI Anomaly predictions
        if ai_pred:
            self.anomaly_score = float(ai_pred.get("anomaly_score", 0.0))

        # Update Health evaluations
        if health_eval:
            self.health_score = float(health_eval.get("health_score", 100.0))
            self.condition = health_eval.get("condition", "EXCELLENT")

        # Update Operational Risk
        if risk_eval:
            self.risk_score = float(risk_eval.get("risk_score", 0.0))
            self.risk_level = risk_eval.get("risk_level", "LOW")
            self.primary_risk_driver = risk_eval.get("primary_driver", "None")

        # Update Predictive Maintenance
        if maint_diag:
            self.maintenance_urgency = maint_diag.get("urgency", "NORMAL")
            self.maintenance_recommendation = maint_diag.get("recommendation", "")
            self.target_component = maint_diag.get("target_component", "")

        # Update OEE
        if oee_metrics:
            self.oee_metrics.update(oee_metrics)

        # Compute Status Indicator Badge
        if self.state == "FAILURE" or self.risk_level == "CRITICAL":
            self.status_indicator = TwinStatusIndicator.CRITICAL
        elif self.state == "WARNING" or self.risk_level == "HIGH":
            self.status_indicator = TwinStatusIndicator.WARNING
        elif self.state == "MAINTENANCE":
            self.status_indicator = TwinStatusIndicator.MAINTENANCE
        else:
            self.status_indicator = TwinStatusIndicator.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize machine twin to dictionary."""
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "type": self.machine_type,
            "state": self.state,
            "status_indicator": self.status_indicator,
            "is_online": self.is_online,
            "last_updated": self.last_updated,
            "telemetry": self.telemetry,
            "health_score": self.health_score,
            "condition": self.condition,
            "anomaly_score": self.anomaly_score,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "primary_risk_driver": self.primary_risk_driver,
            "maintenance_urgency": self.maintenance_urgency,
            "maintenance_recommendation": self.maintenance_recommendation,
            "target_component": self.target_component,
            "oee_metrics": self.oee_metrics,
        }


class DigitalTwin:
    """Virtual representation of the entire smart factory line and equipment."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Digital Twin with factory topology."""
        self.config = config or load_config()
        self.factory_name: str = self.config.get("factory", {}).get(
            "name", "Smart Factory Virtual Line"
        )

        # Factory layout stations definition
        self.layout_stations: List[Dict[str, Any]] = [
            {"id": "ST01", "name": "Raw Material Input", "type": "infeed"},
            {"id": "ST02", "name": "Infeed Conveyor", "type": "conveyor"},
            {"id": "M01", "name": "CNC Milling Machine", "type": "machine"},
            {"id": "ST03", "name": "Intermediate Conveyor", "type": "conveyor"},
            {"id": "M02", "name": "Hydraulic Stamping Press", "type": "machine"},
            {"id": "M03", "name": "Robotic Welding Station", "type": "machine"},
            {"id": "ST04", "name": "Assembly Station", "type": "assembly"},
            {"id": "ST05", "name": "Quality Inspection", "type": "inspection"},
            {"id": "ST06", "name": "Finished Goods Output", "type": "outfeed"},
        ]

        # Initialize Machine Twins
        self.machines: Dict[str, MachineTwin] = {}
        for m_cfg in self.config.get("factory", {}).get("machines", []):
            m_id = m_cfg["id"]
            self.machines[m_id] = MachineTwin(m_cfg)

        # Factory Level State
        self.factory_status: str = TwinStatusIndicator.NORMAL
        self.last_synced: str = datetime.now(timezone.utc).isoformat()
        self.production_summary: Dict[str, Any] = {
            "total_production": 0,
            "total_good": 0,
            "total_defects": 0,
            "defect_rate": 0.0,
            "active_machines": 0,
            "warning_machines": 0,
            "failed_machines": 0,
            "maintenance_machines": 0,
        }

    def update_machine(
        self,
        machine_id: str,
        telemetry: Dict[str, Any],
        ai_pred: Optional[Dict[str, Any]] = None,
        health_eval: Optional[Dict[str, Any]] = None,
        risk_eval: Optional[Dict[str, Any]] = None,
        maint_diag: Optional[Dict[str, Any]] = None,
        oee_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update a specific machine twin."""
        if machine_id in self.machines:
            self.machines[machine_id].update(
                telemetry, ai_pred, health_eval, risk_eval, maint_diag, oee_metrics
            )
            self._recalculate_factory_state()

    def update_production_summary(self, prod_data: Dict[str, Any]):
        """Update factory line production metrics."""
        summary = prod_data.get("production_summary", prod_data)
        self.production_summary.update(summary)
        self._recalculate_factory_state()

    def _recalculate_factory_state(self):
        """Recompute factory-level status based on child machine twin states."""
        self.last_synced = datetime.now(timezone.utc).isoformat()

        failed_count = sum(1 for m in self.machines.values() if m.state == "FAILURE" or m.risk_level == "CRITICAL")
        warning_count = sum(1 for m in self.machines.values() if m.state == "WARNING" or m.risk_level == "HIGH")

        if failed_count > 0:
            self.factory_status = TwinStatusIndicator.CRITICAL
        elif warning_count > 0:
            self.factory_status = TwinStatusIndicator.WARNING
        else:
            self.factory_status = TwinStatusIndicator.NORMAL

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a complete, serializable snapshot of the Digital Twin."""
        return {
            "factory_name": self.factory_name,
            "factory_status": self.factory_status,
            "last_synced": self.last_synced,
            "layout_stations": self.layout_stations,
            "machines": {m_id: m.to_dict() for m_id, m in self.machines.items()},
            "production_summary": self.production_summary,
        }

    def render_ascii_view(self) -> str:
        """Render a clean ASCII visual representation of the factory Digital Twin."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"  🏭 DIGITAL TWIN: {self.factory_name.upper()}  [{self.factory_status}]")
        lines.append(f"  Synced: {self.last_synced}")
        lines.append("=" * 80)
        lines.append("  [Raw Material Infeed] ──▶ [Conveyor 1]")
        lines.append("                                │")

        for m_id, m in self.machines.items():
            badge = m.status_indicator
            h_bar = "█" * int(m.health_score / 10) + "░" * (10 - int(m.health_score / 10))
            lines.append(f"  ┌── {m_id}: {m.name} [{badge}]")
            lines.append(f"  │   Health: [{h_bar}] {m.health_score:5.1f}% ({m.condition}) | Risk: {m.risk_score:4.1f} ({m.risk_level})")
            lines.append(f"  │   Sensors: Temp: {m.telemetry['temperature']:4.1f}°C | Vib: {m.telemetry['vibration']:4.2f} mm/s | RPM: {m.telemetry['rpm']:5.0f} | Curr: {m.telemetry['current']:4.1f}A")
            lines.append(f"  │   Diagnostics: {m.maintenance_recommendation}")
            lines.append("  ▼")

        lines.append("  [Assembly Station] ──▶ [Quality Inspection] ──▶ [Finished Goods Output]")
        lines.append("=" * 80)
        lines.append(
            f"  Production: Total: {self.production_summary.get('total_production', 0)} | "
            f"Good: {self.production_summary.get('total_good', 0)} | "
            f"Defects: {self.production_summary.get('total_defects', 0)} ({self.production_summary.get('defect_rate', 0.0)*100:.2f}%)"
        )
        lines.append("=" * 80)
        return "\n".join(lines)
