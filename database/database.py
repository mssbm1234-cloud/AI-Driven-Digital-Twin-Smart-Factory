"""SQLite Database Manager for Smart Factory operations.

Provides persistent storage, historical time-series queries, and automatic
ingestion of MQTT telemetry, production metrics, alerts, and AI predictions.
"""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DatabaseManager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "smart_factory.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseManager:
    """Manages SQLite schema creation, connections, inserts, and analytics queries."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to data/smart_factory.db.
            config: Optional configuration dictionary.
        """
        self.config = config or load_config()

        if db_path:
            self.db_path = Path(db_path)
        else:
            # Check environment / config
            env_db = os.environ.get("DATABASE_PATH")
            if env_db:
                self.db_path = PROJECT_ROOT / env_db
            else:
                self.db_path = DEFAULT_DB_PATH

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema and seed machines
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional database connection with row factory."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency WAL mode
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Database transaction rolled back due to error: %s", e)
            raise
        finally:
            conn.close()

    def init_db(self):
        """Execute schema DDL and synchronize machine configurations."""
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found at: {SCHEMA_PATH}")

        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)

        logger.info("Database schema initialized at %s", self.db_path)
        self.sync_machines_from_config()

    def sync_machines_from_config(self):
        """Upsert configured machines into the machines metadata table."""
        machines = self.config.get("factory", {}).get("machines", [])
        if not machines:
            return

        with self.get_connection() as conn:
            for m in machines:
                nom = m.get("nominal", {})
                conn.execute(
                    """
                    INSERT INTO machines (
                        machine_id, name, type,
                        nominal_temperature, nominal_vibration,
                        nominal_current, nominal_rpm,
                        nominal_pressure, nominal_power,
                        nominal_cycle_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(machine_id) DO UPDATE SET
                        name=excluded.name,
                        type=excluded.type,
                        nominal_temperature=excluded.nominal_temperature,
                        nominal_vibration=excluded.nominal_vibration,
                        nominal_current=excluded.nominal_current,
                        nominal_rpm=excluded.nominal_rpm,
                        nominal_pressure=excluded.nominal_pressure,
                        nominal_power=excluded.nominal_power,
                        nominal_cycle_time=excluded.nominal_cycle_time;
                    """,
                    (
                        m["id"],
                        m.get("name", m["id"]),
                        m.get("type", "generic"),
                        nom.get("temperature", 65.0),
                        nom.get("vibration", 2.5),
                        nom.get("current", 8.5),
                        nom.get("rpm", 1800.0),
                        nom.get("pressure", 6.0),
                        nom.get("power", 3.2),
                        nom.get("cycle_time", 12.0),
                    ),
                )
        logger.info("Synchronized %d machine configurations to SQLite.", len(machines))

    def insert_telemetry(self, t: Dict[str, Any]) -> int:
        """Insert a single machine telemetry record into the database.

        Args:
            t: Telemetry dictionary.

        Returns:
            The inserted row ID.
        """
        sql = """
        INSERT INTO telemetry (
            timestamp, machine_id, machine_state,
            temperature, vibration, current, voltage,
            rpm, pressure, power,
            production_count, good_products, defect_count,
            defect_rate, cycle_time, downtime_seconds, degradation_level
        ) VALUES (
            :timestamp, :machine_id, :machine_state,
            :temperature, :vibration, :current, :voltage,
            :rpm, :pressure, :power,
            :production_count, :good_products, :defect_count,
            :defect_rate, :cycle_time, :downtime_seconds, :degradation_level
        );
        """
        params = {
            "timestamp": t["timestamp"],
            "machine_id": t["machine_id"],
            "machine_state": t["machine_state"],
            "temperature": float(t["temperature"]),
            "vibration": float(t["vibration"]),
            "current": float(t["current"]),
            "voltage": float(t["voltage"]),
            "rpm": float(t["rpm"]),
            "pressure": float(t["pressure"]),
            "power": float(t["power"]),
            "production_count": int(t.get("production_count", 0)),
            "good_products": int(t.get("good_products", 0)),
            "defect_count": int(t.get("defect_count", 0)),
            "defect_rate": float(t.get("defect_rate", 0.0)),
            "cycle_time": float(t.get("cycle_time", 0.0)),
            "downtime_seconds": float(t.get("downtime_seconds", 0.0)),
            "degradation_level": float(t.get("degradation_level", 0.0)),
        }

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.lastrowid

    def insert_production(self, p: Dict[str, Any]) -> int:
        """Insert a factory production summary record."""
        summary = p.get("production_summary", {})
        sql = """
        INSERT INTO production (
            timestamp, factory_name, factory_status,
            total_production, total_good, total_defects, defect_rate,
            active_machines, warning_machines, failed_machines,
            maintenance_machines, idle_machines
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    p["timestamp"],
                    p.get("factory_name", "Smart Factory"),
                    p.get("factory_status", "NORMAL"),
                    int(summary.get("total_production", 0)),
                    int(summary.get("total_good", 0)),
                    int(summary.get("total_defects", 0)),
                    float(summary.get("defect_rate", 0.0)),
                    int(summary.get("active_machines", 0)),
                    int(summary.get("warning_machines", 0)),
                    int(summary.get("failed_machines", 0)),
                    int(summary.get("maintenance_machines", 0)),
                    int(summary.get("idle_machines", 0)),
                ),
            )
            return cursor.lastrowid

    def insert_alert(self, a: Dict[str, Any]) -> int:
        """Insert an operational alert record."""
        sql = """
        INSERT INTO alerts (timestamp, machine_id, alert_type, severity, message, acknowledged)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    a["timestamp"],
                    a.get("machine_id"),
                    a.get("alert_type", "GENERIC_ALERT"),
                    a.get("severity", "WARNING"),
                    a.get("message", ""),
                    int(a.get("acknowledged", 0)),
                ),
            )
            return cursor.lastrowid

    def insert_maintenance_event(self, m: Dict[str, Any]) -> int:
        """Insert a machine maintenance service event."""
        sql = """
        INSERT INTO maintenance_events (timestamp, machine_id, event_type, description, downtime_minutes, technician)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    m["timestamp"],
                    m["machine_id"],
                    m.get("event_type", "SCHEDULED"),
                    m.get("description", ""),
                    float(m.get("downtime_minutes", 0.0)),
                    m.get("technician", "Service Tech"),
                ),
            )
            return cursor.lastrowid

    def insert_ai_prediction(self, pred: Dict[str, Any]) -> int:
        """Insert an AI/ML inference evaluation record."""
        sql = """
        INSERT INTO ai_predictions (
            timestamp, machine_id, anomaly_score, health_score,
            risk_level, is_anomaly, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    pred["timestamp"],
                    pred["machine_id"],
                    float(pred["anomaly_score"]),
                    float(pred["health_score"]),
                    pred["risk_level"],
                    int(pred.get("is_anomaly", 0)),
                    pred.get("recommendation", ""),
                ),
            )
            return cursor.lastrowid

    def insert_machine_health(
        self,
        health_data: Dict[str, Any],
        risk_data: Optional[Dict[str, Any]] = None,
        anomaly_score: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert a machine health and operational risk record into machine_health table.

        Args:
            health_data: Dictionary from HealthEvaluator containing health_score, condition, subscores.
            risk_data: Optional dictionary from OperationalRiskEngine containing risk_score, risk_level, primary_driver.
            anomaly_score: PyTorch Autoencoder anomaly score.
            timestamp: ISO8601 timestamp string.

        Returns:
            The inserted row ID.
        """
        ts = timestamp or health_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        m_id = health_data.get("machine_id", "M01")
        h_score = float(health_data.get("health_score", 100.0))
        condition = health_data.get("condition", "EXCELLENT")
        rule_h = float(health_data.get("rule_health", h_score))
        ai_h = float(health_data.get("ai_health") if health_data.get("ai_health") is not None else h_score)

        r_score = float(risk_data.get("risk_score", 0.0)) if risk_data else round(100.0 - h_score, 1)
        r_level = risk_data.get("risk_level", "LOW") if risk_data else ("LOW" if h_score >= 80 else ("MEDIUM" if h_score >= 60 else "CRITICAL"))
        driver = risk_data.get("primary_driver", "None") if risk_data else "None"

        sql = """
        INSERT INTO machine_health (
            timestamp, machine_id, health_score, health_condition,
            rule_health, ai_health, anomaly_score, risk_score,
            risk_level, primary_risk_driver
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    ts,
                    m_id,
                    h_score,
                    condition,
                    rule_h,
                    ai_h,
                    float(anomaly_score),
                    r_score,
                    r_level,
                    driver,
                ),
            )
            return cursor.lastrowid

    def get_latest_machine_health(
        self, machine_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent machine health and operational risk records."""
        with self.get_connection() as conn:
            if machine_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM machine_health
                    WHERE machine_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (machine_id, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM machine_health ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def handle_mqtt_message(self, topic: str, data: Dict[str, Any]):
        """Handler method to automatically persist incoming MQTT messages.

        Can be registered directly into FactoryMqttSubscriber.register_handler().
        """
        try:
            if topic == "factory/production/telemetry":
                self.insert_production(data)
            elif "/telemetry" in topic and "machine_id" in data:
                self.insert_telemetry(data)
            elif "alerts" in topic:
                self.insert_alert(data)
        except Exception as e:
            logger.error("Error storing MQTT message from %s into SQLite: %s", topic, e)

    def get_latest_telemetry(
        self, machine_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent telemetry records."""
        with self.get_connection() as conn:
            if machine_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM telemetry
                    WHERE machine_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (machine_id, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_production(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent factory production summary."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM production ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_alerts(
        self,
        machine_id: Optional[str] = None,
        severity: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical or active alerts with optional filtering.

        Args:
            machine_id: Optional machine filter ('M01', 'M02', 'M03').
            severity: Optional severity filter ('INFO', 'WARNING', 'CRITICAL').
            unacknowledged_only: If True, only return unacknowledged alerts.
            limit: Maximum records to return.

        Returns:
            List of alert dictionaries ordered newest first.
        """
        clauses = []
        params = []

        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if unacknowledged_only:
            clauses.append("acknowledged = 0")

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM alerts {where_str} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark a specific alert as acknowledged in the database.

        Args:
            alert_id: Primary key ID of the alert to acknowledge.

        Returns:
            True if row was updated, False if alert ID was not found.
        """
        sql = "UPDATE alerts SET acknowledged = 1 WHERE id = ?;"
        with self.get_connection() as conn:
            cursor = conn.execute(sql, (alert_id,))
            return cursor.rowcount > 0

    def get_telemetry_count(self, machine_id: Optional[str] = None) -> int:
        """Count total telemetry rows in the database."""
        with self.get_connection() as conn:
            if machine_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM telemetry WHERE machine_id = ?",
                    (machine_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM telemetry")
            return cursor.fetchone()[0]

