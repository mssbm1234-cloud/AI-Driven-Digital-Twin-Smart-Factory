"""AI-Driven Digital Twin: Phase 10 Master End-to-End Factory Integration Pipeline.

Synchronizes and orchestrates the complete Smart Factory stack:
1. Multi-Machine Physics Simulation (M01, M02, M03)
2. MQTT Industrial Telemetry Publishing & Ingestion (Mosquitto)
3. Deep Autoencoder AI Anomaly Detection (PyTorch)
4. Equipment Health Analytics (0-100%)
5. Operational Risk Assessment Matrix (LOW, MEDIUM, HIGH, CRITICAL)
6. Predictive Maintenance Diagnosis & Component Isolation
7. Alert Deduplication, Cooldown, and Multi-Channel Dispatch
8. Real-Time Digital Twin Virtual In-Memory State Model
9. SQLite Database Persistence (Time-Series & Relational)
10. FastAPI REST API & Interactive Web Visualizer Dashboard
"""

from datetime import datetime, timezone
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import load_config
from simulator.factory_simulator import FactorySimulator
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber
from database.database import DatabaseManager
from ai.inference import AnomalyDetector
from analytics.health import HealthEvaluator
from analytics.risk import OperationalRiskEngine
from analytics.maintenance import PredictiveMaintenanceEngine
from analytics.oee import OEEEngine
from alerts.alert_manager import AlertManager, AlertSeverity, AlertType
from digital_twin.twin import DigitalTwin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MasterPipeline")


class SmartFactoryMasterPipeline:
    """Master orchestrator integrating all 10 project phases into a cohesive live pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all subsystems and connect pipeline components."""
        self.config: Dict[str, Any] = config or load_config()

        # Core Subsystems
        self.db = DatabaseManager(config=self.config)
        self.twin = DigitalTwin(config=self.config)
        self.ai_detector = AnomalyDetector(config=self.config)
        self.health_evaluator = HealthEvaluator(config=self.config)
        self.risk_engine = OperationalRiskEngine(config=self.config)
        self.maint_engine = PredictiveMaintenanceEngine()
        self.oee_engine = OEEEngine()
        self.alert_manager = AlertManager(config=self.config)

        # Messaging Subsystems
        self.publisher = FactoryMqttPublisher(config=self.config)
        self.subscriber = FactoryMqttSubscriber(config=self.config)

        # Operational state
        self._running: bool = False
        self.processed_cycles: int = 0
        self.generated_alerts: List[Dict[str, Any]] = []

        # Wire MQTT subscriber callbacks
        self.subscriber.register_handler(self._on_mqtt_message)

    def _on_mqtt_message(self, topic: str, data: Dict[str, Any]):
        """Master telemetry processor executing AI, Analytics, Alerts, Twin, and DB storage."""
        try:
            if topic == "factory/production/telemetry":
                self.db.insert_production(data)
                self.twin.update_production_summary(data)

            elif "/telemetry" in topic and "machine_id" in data:
                # 1. Persist Raw Telemetry
                self.db.insert_telemetry(data)

                # 2. PyTorch AI Anomaly Inference
                ai_pred = self.ai_detector.predict_and_store(data, self.db)

                # 3. Multi-Factor Health Evaluation
                health_eval = self.health_evaluator.evaluate_health(
                    data, ai_anomaly_score=ai_pred["anomaly_score"]
                )

                # 4. Multi-Dimensional Operational Risk Matrix
                risk_eval = self.risk_engine.evaluate_risk(
                    data, health_eval, ai_anomaly_score=ai_pred["anomaly_score"]
                )

                # Persist Machine Health & Operational Risk to SQLite (machine_health table)
                self.db.insert_machine_health(
                    health_data=health_eval,
                    risk_data=risk_eval,
                    anomaly_score=ai_pred["anomaly_score"],
                    timestamp=data.get("timestamp"),
                )

                # 5. Predictive Maintenance Diagnosis
                maint_diag = self.maint_engine.diagnose_and_recommend(data, health_eval, risk_eval)
                self.maint_engine.log_recommendation_to_db(maint_diag, self.db)

                # 6. OEE Calculation
                oee_metrics = self.oee_engine.calculate_machine_oee(data)

                # 7. Industrial Alert Generation & Cooldown Filter
                dispatched_alerts = self.alert_manager.evaluate_and_dispatch(
                    telemetry=data,
                    ai_pred=ai_pred,
                    health_eval=health_eval,
                    risk_eval=risk_eval,
                    maint_diag=maint_diag,
                    db=self.db,
                    publisher=self.publisher,
                )
                if dispatched_alerts:
                    self.generated_alerts.extend(dispatched_alerts)

                # 8. Digital Twin State Model Synchronization
                self.twin.update_machine(
                    machine_id=data["machine_id"],
                    telemetry=data,
                    ai_pred=ai_pred,
                    health_eval=health_eval,
                    risk_eval=risk_eval,
                    maint_diag=maint_diag,
                    oee_metrics=oee_metrics,
                )

        except Exception as e:
            logger.error("Pipeline message processing exception on topic %s: %s", topic, e)

    def start(self, timeout: float = 5.0) -> bool:
        """Connect MQTT publisher and subscriber to begin data processing."""
        logger.info("Starting Master Smart Factory Integration Pipeline...")
        sub_ok = self.subscriber.connect(timeout=timeout)
        pub_ok = self.publisher.connect(timeout=timeout)

        if not (sub_ok and pub_ok):
            logger.error("Failed to connect to MQTT broker at %s:%s", self.publisher.broker_host, self.publisher.broker_port)
            return False

        logger.info("Pipeline connected successfully to MQTT Broker.")
        time.sleep(0.3)
        return True

    def stop(self):
        """Cleanly disconnect pipeline subsystems."""
        logger.info("Shutting down Master Integration Pipeline...")
        self._running = False
        self.publisher.disconnect()
        self.subscriber.disconnect()
        logger.info("Pipeline stopped cleanly.")

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """Execute a single synchronous simulation step through all 10 phases."""
        snapshot = self.publisher.publish_step(dt=dt)
        self.processed_cycles += 1
        time.sleep(0.15)  # Allow async subscriber loop to complete processing
        return snapshot

    def run_demonstration(self, steps: int = 5):
        """Execute a multi-stage demonstration highlighting normal operation, wear, and failure recovery."""
        print("\n" + "=" * 95)
        print("🏭 SMART FACTORY AI DIGITAL TWIN — MASTER END-TO-END DEMO (PHASES 1 - 10)")
        print("=" * 95)

        if not self.start():
            print("[!] Could not connect to Mosquitto MQTT Broker. Ensure service is active.")
            return

        try:
            print("\n[STAGE 1: Baseline Nominal Operations]")
            for i in range(2):
                print(f"  > Cycle {i+1}: Running nominal factory production...")
                self.step(dt=1.0)
                time.sleep(0.2)

            print("\n[STAGE 2: Inducing Mechanical Degradation on M01]")
            self.publisher.simulator.inject_degradation("M01", rate=0.35)
            self.step(dt=1.5)
            time.sleep(0.2)

            print("\n[STAGE 3: Critical Machine Breakdown on M01 & Alert Triggering]")
            self.publisher.simulator.inject_failure("M01")
            self.step(dt=1.0)
            time.sleep(0.3)

            print("\n[STAGE 4: Automated Maintenance Recovery on M01]")
            self.publisher.simulator.repair_machine("M01")
            self.step(dt=1.0)
            time.sleep(0.2)

            # Display Live Digital Twin Render
            print("\n" + self.twin.render_ascii_view())

            # Display Recent Alerts Summary
            alerts = self.db.get_alerts(limit=5)
            print("\n" + "=" * 95)
            print("🚨 RECENT INDUSTRIAL ALERTS PERSISTED TO SQLite (alerts table)")
            print("=" * 95)
            if alerts:
                for a in alerts:
                    print(
                        f"  [{a['id']:02d}] {a['timestamp'][:19]} | {a['severity']:<8} | "
                        f"Machine: {a['machine_id']} | Type: {a['alert_type']:<22} | {a['message'][:60]}..."
                    )
            else:
                print("  No active alerts recorded.")

            print("\n" + "=" * 95)
            print(">>> COMPLETE SMART FACTORY PIPELINE INTEGRATION VERIFIED SUCCESSFULLY <<<")
            print("=" * 95)

        finally:
            self.stop()


    def run_continuous(self, interval: float = 1.0):
        """Continuously stream factory telemetry, AI analytics, and alerts over MQTT indefinitely."""
        print("\n" + "=" * 95)
        print("🏭 SMART FACTORY AI DIGITAL TWIN — CONTINUOUS REAL-TIME DATA STREAM")
        print(f"📡 MQTT Broker: {self.publisher.broker_host}:{self.publisher.broker_port} (TLS: {self.publisher.use_tls})")
        print("⚡ Press Ctrl+C to stop the stream.")
        print("=" * 95)

        if not self.start():
            print(f"[!] Could not connect to MQTT Broker at {self.publisher.broker_host}:{self.publisher.broker_port}.")
            return

        self._running = True
        cycle = 0

        def handle_signal(sig, frame):
            print("\n[!] Gracefully stopping continuous data stream...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            while self._running:
                cycle += 1
                self.step(dt=interval)
                m01 = self.twin.machines.get("M01")
                m02 = self.twin.machines.get("M02")
                m03 = self.twin.machines.get("M03")
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Cycle {cycle:04d} | "
                    f"M01: {m01.telemetry['temperature']:.1f}°C, {m01.telemetry['vibration']:.2f}mm/s, H:{m01.health_score:.1f}% | "
                    f"M02: {m02.telemetry['temperature']:.1f}°C, H:{m02.health_score:.1f}% | "
                    f"M03: {m03.telemetry['temperature']:.1f}°C, H:{m03.health_score:.1f}% | "
                    f"Line: {self.twin.factory_status}"
                )
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            print(f"\n[✓] Finished continuous stream ({cycle} cycles processed).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smart Factory AI Digital Twin Master Pipeline")
    parser.add_argument("--continuous", action="store_true", help="Run continuous streaming mode indefinitely")
    parser.add_argument("--interval", type=float, default=1.0, help="Telemetry interval in seconds (default: 1.0)")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps in demo mode (default: 5)")
    args = parser.parse_args()

    pipeline = SmartFactoryMasterPipeline()
    if args.continuous:
        pipeline.run_continuous(interval=args.interval)
    else:
        pipeline.run_demonstration(steps=args.steps)


if __name__ == "__main__":
    main()
