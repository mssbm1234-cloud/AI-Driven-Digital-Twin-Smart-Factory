"""Demonstration runner for Phase 4: MQTT Publisher-to-Subscriber Telemetry Pipeline.

Connects Subscriber and Publisher to Mosquitto, executes 3 simulation steps,
and displays received, validated JSON packets in real-time.
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

from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber


def main():
    print("=" * 75)
    print("AI-Driven Digital Twin - Phase 4 MQTT Publisher -> Subscriber Pipeline")
    print("=" * 75)

    sub = FactoryMqttSubscriber()
    pub = FactoryMqttPublisher()

    # Register live telemetry print callback
    def on_telemetry(topic: str, data: dict):
        if "production" in topic:
            prod = data.get("production_summary", {})
            print(
                f"[RX] {topic:<32} | Status: {data.get('factory_status'):<8} | "
                f"Total: {prod.get('total_production')} | Active: {prod.get('active_machines')}"
            )
        elif "/telemetry" in topic:
            print(
                f"[RX] {topic:<32} | {data.get('machine_id')} ({data.get('machine_state'):<7}) | "
                f"T: {data.get('temperature'):5.1f}C | V: {data.get('vibration'):4.2f} mm/s | "
                f"I: {data.get('current'):4.1f}A | RPM: {data.get('rpm'):6.1f}"
            )

    sub.register_handler(on_telemetry)

    # Connect both
    print("Connecting Subscriber to Mosquitto (localhost:1883)...")
    if not sub.connect(timeout=3.0):
        print("[!] Could not connect subscriber. Ensure Mosquitto service is running.")
        sys.exit(1)

    print("Connecting Publisher to Mosquitto (localhost:1883)...")
    if not pub.connect(timeout=3.0):
        print("[!] Could not connect publisher.")
        sub.disconnect()
        sys.exit(1)

    time.sleep(0.3)
    print("\n--- Streaming 3 Live Simulation Cycles via MQTT ---")

    for step in range(1, 4):
        print(f"\n>> Publishing Simulation Step {step}...")
        pub.publish_step(dt=1.0)
        time.sleep(0.5)

    time.sleep(0.5)
    pub.disconnect()
    sub.disconnect()

    print("\n" + "=" * 75)
    print(f"Total MQTT Messages Received: {sub.total_received}")
    print(f"Valid Validated Payloads:     {sub.valid_messages}")
    print(f"Invalid / Malformed Payloads: {sub.invalid_messages}")
    print("Messages By Topic:")
    for topic, count in sub.messages_by_topic.items():
        print(f"  - {topic:<32}: {count} messages")
    print(">>> PHASE 4 DEMONSTRATION COMPLETE <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()
