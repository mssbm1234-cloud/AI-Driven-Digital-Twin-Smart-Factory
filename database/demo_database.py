"""Demonstration runner for Phase 5: SQLite Database & Telemetry Storage.

Connects the MQTT subscriber directly to the SQLite DatabaseManager,
publishes live telemetry cycles via Mosquitto, and queries the persisted records.
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


def main():
    print("=" * 80)
    print("AI-Driven Digital Twin - Phase 5 SQLite Database Pipeline Demo")
    print("=" * 80)

    # 1. Initialize SQLite Database
    db = DatabaseManager()
    print(f"Database initialized: {db.db_path}")

    # 2. Hook Subscriber to Database
    sub = FactoryMqttSubscriber()
    sub.register_handler(db.handle_mqtt_message)

    pub = FactoryMqttPublisher()

    print("Connecting Subscriber and Publisher to Mosquitto (localhost:1883)...")
    if not sub.connect(timeout=3.0) or not pub.connect(timeout=3.0):
        print("[!] Failed to connect to Mosquitto broker.")
        sys.exit(1)

    time.sleep(0.3)

    # 3. Stream 3 cycles across all machines
    initial_count = db.get_telemetry_count()
    print(f"Initial Telemetry Rows in SQLite: {initial_count}")
    print("\nStreaming 3 live simulation cycles: Simulator -> Publisher -> Broker -> Subscriber -> SQLite...")

    for step in range(1, 4):
        pub.publish_step(dt=1.0)
        time.sleep(0.4)

    time.sleep(0.6)
    pub.disconnect()
    sub.disconnect()

    # 4. Query SQLite
    final_count = db.get_telemetry_count()
    print("\n" + "=" * 80)
    print(f"SQLite Telemetry Rows Stored: {final_count} (+{final_count - initial_count} new rows)")
    print("=" * 80)

    # Display latest records from telemetry table
    print("Latest Telemetry Records from SQLite Table 'telemetry':")
    print(f"{'ID':<4} | {'Timestamp':<25} | {'Machine':<7} | {'State':<8} | {'Temp(C)':<7} | {'Vib':<5} | {'Current':<7} | {'RPM':<6}")
    print("-" * 80)

    latest_rows = db.get_latest_telemetry(limit=6)
    for r in reversed(latest_rows):
        print(
            f"{r['id']:<4} | {r['timestamp'][:23]:<25} | {r['machine_id']:<7} | "
            f"{r['machine_state']:<8} | {r['temperature']:<7.2f} | {r['vibration']:<5.2f} | "
            f"{r['current']:<7.2f} | {r['rpm']:<6.1f}"
        )

    # Display latest production summary
    print("\nLatest Production Snapshot from SQLite Table 'production':")
    prod = db.get_latest_production()
    if prod:
        print(
            f"Factory: {prod['factory_name']} | Status: {prod['factory_status']} | "
            f"Active: {prod['active_machines']} | Total Produced: {prod['total_production']} | "
            f"Defects: {prod['total_defects']} ({prod['defect_rate']*100:.2f}%)"
        )
    print("=" * 80)
    print(">>> PHASE 5 DEMONSTRATION COMPLETE: Telemetry successfully persisted in SQLite! <<<")


if __name__ == "__main__":
    main()
