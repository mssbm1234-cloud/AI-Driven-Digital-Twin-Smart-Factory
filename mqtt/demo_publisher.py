"""Demonstration runner for Phase 3: MQTT Publisher.

Publishes 5 consecutive telemetry steps over the Mosquitto MQTT broker,
displaying topics and JSON payloads.
"""

import sys
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


def main():
    print("=" * 70)
    print("AI-Driven Digital Twin - Phase 3 MQTT Publisher Demonstration")
    print("=" * 70)

    publisher = FactoryMqttPublisher()
    connected = publisher.connect(timeout=3.0)

    if not connected:
        print("[!] Could not connect to Mosquitto MQTT broker at localhost:1883.")
        print("    Ensure the Mosquitto service is running: Start-Service mosquitto")
        sys.exit(1)

    print(f"Connected to Mosquitto at {publisher.broker_host}:{publisher.broker_port}")
    print("Publishing 5 consecutive telemetry iterations...")
    print("=" * 70)

    publisher.start_publishing(max_iterations=5)

    print("=" * 70)
    print(f"Published {publisher.published_message_count} total MQTT packets.")
    print(">>> PHASE 3 DEMONSTRATION COMPLETE <<<")


if __name__ == "__main__":
    main()
