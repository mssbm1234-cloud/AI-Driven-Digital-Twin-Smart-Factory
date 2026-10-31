"""Demonstration runner for Phase 7: Machine Health, Operational Risk & Predictive Maintenance."""

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


def main():
    print("=" * 105)
    print("AI-Driven Digital Twin - Phase 7 Machine Health, Operational Risk & Maintenance Analytics Demo")
    print("=" * 105)

    db = DatabaseManager()
    ai_detector = AnomalyDetector()
    health_evaluator = HealthEvaluator()
    risk_engine = OperationalRiskEngine()
    maint_engine = PredictiveMaintenanceEngine()

    sub = FactoryMqttSubscriber()
    pub = FactoryMqttPublisher()

    analysis_results = []

    def on_telemetry(topic: str, data: dict):
        if "/telemetry" in topic and topic != "factory/production/telemetry":
            # 1. Store telemetry
            db.insert_telemetry(data)

            # 2. PyTorch AI Anomaly Detection
            ai_pred = ai_detector.predict_and_store(data, db)
            anomaly_score = ai_pred["anomaly_score"]

            # 3. Comprehensive Machine Health Assessment
            health_eval = health_evaluator.evaluate_health(data, ai_anomaly_score=anomaly_score)

            # 4. Operational Risk Engine
            risk_eval = risk_engine.evaluate_risk(data, health_eval, ai_anomaly_score=anomaly_score)

            # 5. Prescriptive Maintenance Diagnostics
            maint_diag = maint_engine.diagnose_and_recommend(data, health_eval, risk_eval)
            maint_engine.log_recommendation_to_db(maint_diag, db)

            analysis_results.append((data, health_eval, risk_eval, maint_diag))

    sub.register_handler(on_telemetry)

    print("Connecting Subscriber and Publisher to Mosquitto (localhost:1883)...")
    if not sub.connect(timeout=3.0) or not pub.connect(timeout=3.0):
        print("[!] Broker connection failed.")
        sys.exit(1)

    time.sleep(0.3)

    print("\n--- Executing 4 Lifecycle Stages across Factory Machines ---")
    print("Stage 1: Normal steady-state operation...")
    pub.publish_step(dt=1.0)
    time.sleep(0.4)

    print("Stage 2: Inducing mechanical degradation on M01...")
    pub.simulator.inject_degradation("M01", rate=0.25)
    pub.publish_step(dt=1.0)
    time.sleep(0.4)

    print("Stage 3: Thermal & vibration escalation (Warning State)...")
    pub.publish_step(dt=2.0)
    time.sleep(0.4)

    print("Stage 4: Breakdown failure on M01...")
    pub.simulator.inject_failure("M01")
    pub.publish_step(dt=1.0)
    time.sleep(0.5)

    pub.disconnect()
    sub.disconnect()

    # Display Analytics Dashboard Table
    print("\n" + "=" * 105)
    print(f"{'Machine':<7} | {'State':<8} | {'Health':<6} | {'Condition':<10} | {'Risk':<5} | {'Risk Level':<10} | {'Primary Driver':<24} | {'Urgency':<18}")
    print("-" * 105)

    for data, health, risk, maint in analysis_results:
        print(
            f"{data['machine_id']:<7} | {data['machine_state']:<8} | "
            f"{health['health_score']:<6.1f} | {health['condition']:<10} | "
            f"{risk['risk_score']:<5.1f} | {risk['risk_level']:<10} | "
            f"{risk['primary_driver']:<24} | {maint['urgency']:<18}"
        )

    print("=" * 105)

    # Show logged maintenance recommendations
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM maintenance_events ORDER BY id DESC LIMIT 2;")
        rows = cursor.fetchall()
        if rows:
            print("\nLatest Actionable Maintenance Events Stored in SQLite:")
            for r in rows:
                print(f"[{r['event_type']}] Machine: {r['machine_id']} | Details: {r['description']}")

    print("\n>>> PHASE 7 DEMONSTRATION COMPLETE: Health, Risk & Maintenance Analyzed! <<<")
    print("=" * 105)


if __name__ == "__main__":
    main()
