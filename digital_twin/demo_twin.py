"""Demonstration runner for Phase 8: Digital Twin State Synchronization & Visualization.

Streams live MQTT telemetry through the full analytical stack, dynamically updating
the Digital Twin virtual representation across normal, warning, failure, and recovery states.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from database.database import DatabaseManager
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber
from ai.inference import AnomalyDetector
from analytics.health import HealthEvaluator
from analytics.risk import OperationalRiskEngine
from analytics.maintenance import PredictiveMaintenanceEngine
from analytics.oee import OEEEngine
from digital_twin.twin import DigitalTwin


def main():
    print("=" * 80)
    print("AI-Driven Digital Twin - Phase 8 Virtual Smart Factory Demonstration")
    print("=" * 80)

    twin = DigitalTwin()
    db = DatabaseManager()
    ai_detector = AnomalyDetector()
    health_evaluator = HealthEvaluator()
    risk_engine = OperationalRiskEngine()
    maint_engine = PredictiveMaintenanceEngine()

    sub = FactoryMqttSubscriber()
    pub = FactoryMqttPublisher()

    # Subscriber pipeline updating Digital Twin in real-time
    def on_telemetry(topic: str, data: dict):
        if topic == "factory/production/telemetry":
            twin.update_production_summary(data)
            db.insert_production(data)
        elif "/telemetry" in topic:
            m_id = data.get("machine_id", "M01")

            # 1. AI Anomaly Detection
            ai_pred = ai_detector.predict_and_store(data, db)
            anomaly_score = ai_pred["anomaly_score"]

            # 2. Health & Risk Analysis
            health_eval = health_evaluator.evaluate_health(data, ai_anomaly_score=anomaly_score)
            risk_eval = risk_engine.evaluate_risk(data, health_eval, ai_anomaly_score=anomaly_score)
            maint_diag = maint_engine.diagnose_and_recommend(data, health_eval, risk_eval)
            oee_metrics = OEEEngine.calculate_machine_oee(data)

            # 3. Synchronize Digital Twin State
            twin.update_machine(
                machine_id=m_id,
                telemetry=data,
                ai_pred=ai_pred,
                health_eval=health_eval,
                risk_eval=risk_eval,
                maint_diag=maint_diag,
                oee_metrics=oee_metrics,
            )
            db.insert_telemetry(data)

    sub.register_handler(on_telemetry)

    print("Connecting Subscriber and Publisher to Mosquitto (localhost:1883)...")
    if not sub.connect(timeout=3.0) or not pub.connect(timeout=3.0):
        print("[!] Connection to Mosquitto broker failed.")
        sys.exit(1)

    time.sleep(0.3)

    stages = [
        ("STAGE 1: Normal Baseline Factory Operations", lambda: None),
        ("STAGE 2: Inducing Mechanical Wear on M01 (Degradation)", lambda: pub.simulator.inject_degradation("M01", rate=0.25)),
        ("STAGE 3: Vibration & Thermal Overload on M01 (Warning Trigger)", lambda: None),
        ("STAGE 4: Breakdown Failure on M01 (Critical Trigger)", lambda: pub.simulator.inject_failure("M01")),
        ("STAGE 5: Maintenance Overhaul & Recovery on M01", lambda: pub.simulator.repair_machine("M01")),
    ]

    for title, action in stages:
        print("\n" + "#" * 80)
        print(f"  {title}")
        print("#" * 80)
        action()
        pub.publish_step(dt=1.5)
        time.sleep(0.5)

        # Print current Digital Twin ASCII Virtual Map
        print(twin.render_ascii_view())

    pub.disconnect()
    sub.disconnect()

    print("\n" + "=" * 80)
    print(">>> PHASE 8 DEMONSTRATION COMPLETE: Digital Twin State Synchronized! <<<")
    print("Tip: You can launch the interactive Web Digital Twin using:")
    print("     python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    print("     and navigate to http://127.0.0.1:8000 in your browser.")
    print("=" * 80)


if __name__ == "__main__":
    main()
