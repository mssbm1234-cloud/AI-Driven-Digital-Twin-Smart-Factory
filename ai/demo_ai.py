"""Interactive demonstration runner for Phase 6: PyTorch AI Anomaly Detection Pipeline.

Runs end-to-end:
Factory Simulator -> MQTT Publisher -> Mosquitto -> Subscriber -> PyTorch Inference -> SQLite (ai_predictions)
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


def main():
    print("=" * 90)
    print("AI-Driven Digital Twin - Phase 6 PyTorch Anomaly Detection & Health Scoring Demo")
    print("=" * 90)

    db = DatabaseManager()
    detector = AnomalyDetector()
    sub = FactoryMqttSubscriber()
    pub = FactoryMqttPublisher()

    ai_results = []

    # Telemetry and AI processing callback
    def on_telemetry(topic: str, data: dict):
        if "/telemetry" in topic and topic != "factory/production/telemetry":
            # 1. Save telemetry to SQLite
            db.insert_telemetry(data)
            # 2. Run PyTorch AI inference and store in ai_predictions
            pred = detector.predict_and_store(data, db)
            ai_results.append((data, pred))

    sub.register_handler(on_telemetry)

    print("Connecting Subscriber and Publisher to Mosquitto (localhost:1883)...")
    if not sub.connect(timeout=3.0) or not pub.connect(timeout=3.0):
        print("[!] Connection to broker failed.")
        sys.exit(1)

    time.sleep(0.3)

    print("\n--- Simulating 5 Lifecycle Stages with Live PyTorch Analysis ---")

    # Step 1: Normal
    print("Stage 1: Normal steady-state operation...")
    pub.publish_step(dt=1.0)
    time.sleep(0.4)

    # Step 2: Induce wear on M01
    print("Stage 2: Inducing mechanical wear on M01...")
    pub.simulator.inject_degradation("M01", rate=0.25)
    pub.publish_step(dt=1.0)
    time.sleep(0.4)

    # Step 3: Continued degradation
    print("Stage 3: Thermal and vibration escalation on M01...")
    pub.publish_step(dt=1.5)
    time.sleep(0.4)

    # Step 4: Severe warning threshold
    print("Stage 4: Threshold crossing into WARNING...")
    pub.publish_step(dt=2.0)
    time.sleep(0.4)

    # Step 5: Machine breakdown
    print("Stage 5: Tripping M01 into FAILURE...")
    pub.simulator.inject_failure("M01")
    pub.publish_step(dt=1.0)
    time.sleep(0.5)

    pub.disconnect()
    sub.disconnect()

    # Display AI analysis output table
    print("\n" + "=" * 90)
    print(f"{'Machine':<7} | {'State':<8} | {'Temp(C)':<7} | {'Vib':<5} | {'RPM':<6} | {'Anomaly':<7} | {'Health%':<7} | {'Risk':<8} | {'Recommendation'}")
    print("-" * 90)

    for telemetry, pred in ai_results:
        # Display M01 progression and selected other machines
        rec = pred["recommendation"]
        if len(rec) > 36:
            rec = rec[:33] + "..."
        print(
            f"{telemetry['machine_id']:<7} | {telemetry['machine_state']:<8} | "
            f"{telemetry['temperature']:<7.1f} | {telemetry['vibration']:<5.2f} | "
            f"{telemetry['rpm']:<6.0f} | {pred['anomaly_score']:<7.3f} | "
            f"{pred['health_score']:<7.1f} | {pred['risk_level']:<8} | {rec}"
        )

    print("=" * 90)

    # Query SQLite ai_predictions table count
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM ai_predictions")
        stored_preds = cursor.fetchone()[0]

    print(f"Total AI Predictions Persisted in SQLite 'ai_predictions' table: {stored_preds}")
    print(">>> PHASE 6 DEMONSTRATION COMPLETE: PyTorch Anomaly Detection Verified! <<<")


if __name__ == "__main__":
    main()
