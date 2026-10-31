"""MQTT Publisher for Smart Factory telemetry and production metrics.

Bridges the FactorySimulator engine with the Mosquitto MQTT broker, publishing
real-time JSON telemetry for each machine and aggregate line production.
"""

from datetime import datetime, timezone
import json
import logging
import signal
import sys
import time
import ssl
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
from config import load_config
from simulator.factory_simulator import FactorySimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MqttPublisher")


class FactoryMqttPublisher:
    """Publishes factory simulation telemetry over MQTT."""

    def __init__(
        self,
        simulator: Optional[FactorySimulator] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the publisher with config and simulator instance.

        Args:
            simulator: FactorySimulator instance. If None, instantiates a new one.
            config: Configuration dictionary. If None, loaded via load_config().
        """
        self.config: Dict[str, Any] = config or load_config()
        self.simulator: FactorySimulator = simulator or FactorySimulator(self.config)

        # MQTT connection parameters
        mqtt_cfg = self.config.get("mqtt", {})
        self.broker_host: str = mqtt_cfg.get("broker", "localhost")
        self.broker_port: int = int(mqtt_cfg.get("port", 1883))
        self.username: Optional[str] = mqtt_cfg.get("username") or None
        self.password: Optional[str] = mqtt_cfg.get("password") or None
        self.use_tls: bool = bool(mqtt_cfg.get("tls", False)) or (self.broker_port == 8883)
        self.keepalive: int = int(mqtt_cfg.get("keepalive", 60))
        self.publish_interval: float = float(
            mqtt_cfg.get("publish_interval_seconds", 1.0)
        )

        # Topic mappings
        topics_cfg = mqtt_cfg.get("topics", {})
        self.machine_telemetry_topics: Dict[str, str] = topics_cfg.get(
            "machine_telemetry",
            {
                "M01": "factory/machine1/telemetry",
                "M02": "factory/machine2/telemetry",
                "M03": "factory/machine3/telemetry",
            },
        )
        self.machine_status_topics: Dict[str, str] = topics_cfg.get(
            "machine_status",
            {
                "M01": "factory/machine1/status",
                "M02": "factory/machine2/status",
                "M03": "factory/machine3/status",
            },
        )
        self.production_topic: str = topics_cfg.get(
            "production", "factory/production/telemetry"
        )

        # MQTT Client initialization with paho-mqtt v2 Callback API
        self.client_id: str = f"factory_publisher_{int(time.time())}"
        self.client: mqtt.Client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )

        # Configure Credentials (if specified)
        if self.username and self.password:
            self.client.username_pw_set(username=self.username, password=self.password)

        # Configure TLS / SSL (if enabled or port 8883)
        if self.use_tls:
            self.client.tls_set(
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )

        # Attach event callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self.is_connected: bool = False
        self._running: bool = False
        self.published_message_count: int = 0

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback triggered upon establishing broker connection."""
        if rc == 0:
            self.is_connected = True
            logger.info("Connected successfully to MQTT broker at %s:%d", self.broker_host, self.broker_port)
        else:
            self.is_connected = False
            logger.error("Failed to connect to MQTT broker. Return code: %s", rc)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        """Callback triggered when disconnected from broker."""
        self.is_connected = False
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT broker (rc=%s). Auto-reconnecting...", rc)
        else:
            logger.info("Gracefully disconnected from MQTT broker.")

    def _on_publish(self, client, userdata, mid, reason_codes=None, properties=None):
        """Callback triggered upon successful message dispatch."""
        self.published_message_count += 1

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the Mosquitto MQTT broker and start background network loop.

        Args:
            timeout: Maximum seconds to wait for connection confirmation.

        Returns:
            True if successfully connected, False otherwise.
        """
        try:
            logger.info(
                "Connecting to MQTT broker %s:%d (TLS: %s, Auth: %s)...",
                self.broker_host,
                self.broker_port,
                self.use_tls,
                "Enabled" if self.username else "Disabled",
            )
            self.client.connect(
                host=self.broker_host,
                port=self.broker_port,
                keepalive=self.keepalive,
            )
            self.client.loop_start()

            # Wait for on_connect callback
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.05)

            if not self.is_connected:
                logger.warning("Connection timeout waiting for broker ACK at %s:%d", self.broker_host, self.broker_port)
                return False

            return True
        except Exception as e:
            logger.error("Error connecting to MQTT broker: %s", e)
            self.is_connected = False
            return False

    def disconnect(self):
        """Gracefully disconnect from broker and stop network loop."""
        self._running = False
        try:
            self.client.loop_stop()
            if self.is_connected:
                self.client.disconnect()
            self.is_connected = False
            logger.info("MQTT Publisher stopped cleanly.")
        except Exception as e:
            logger.error("Error during MQTT disconnect: %s", e)

    def publish_step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Step the factory simulator once and publish all generated payloads to MQTT.

        Args:
            dt: Elapsed time delta for the simulator step.

        Returns:
            The factory snapshot containing all published payloads.
        """
        snapshot = self.simulator.step(dt=dt)

        # 1. Publish telemetry for each machine
        for m_id, telemetry in snapshot["machines"].items():
            topic = self.machine_telemetry_topics.get(
                m_id, f"factory/{m_id.lower()}/telemetry"
            )
            payload_str = json.dumps(telemetry)

            try:
                msg_info = self.client.publish(topic, payload=payload_str, qos=1)
                if msg_info.rc != mqtt.MQTT_ERR_SUCCESS:
                    logger.warning("Failed to publish to topic %s (rc=%d)", topic, msg_info.rc)
            except Exception as e:
                logger.error("Publish exception for topic %s: %s", topic, e)

            # Publish status message to status topic
            status_topic = self.machine_status_topics.get(
                m_id, f"factory/{m_id.lower()}/status"
            )
            status_payload = json.dumps({
                "timestamp": telemetry["timestamp"],
                "machine_id": m_id,
                "machine_state": telemetry["machine_state"],
                "downtime_seconds": telemetry["downtime_seconds"],
            })
            try:
                self.client.publish(status_topic, payload=status_payload, qos=1)
            except Exception as e:
                logger.error("Publish exception for status topic %s: %s", status_topic, e)

        # 2. Publish aggregate line production telemetry
        production_payload = {
            "timestamp": snapshot["timestamp"],
            "factory_name": snapshot["factory_name"],
            "factory_status": snapshot["factory_status"],
            "production_summary": snapshot["production_summary"],
        }
        try:
            self.client.publish(
                self.production_topic,
                payload=json.dumps(production_payload),
                qos=1,
            )
        except Exception as e:
            logger.error("Publish exception for production topic %s: %s", self.production_topic, e)

        return snapshot

    def start_publishing(self, max_iterations: Optional[int] = None):
        """Run continuous publishing loop until interrupted or max_iterations reached.

        Args:
            max_iterations: Optional iteration limit (useful for testing and demos).
        """
        if not self.is_connected:
            connected = self.connect()
            if not connected:
                logger.error("Cannot start publishing: unable to connect to broker.")
                return

        self._running = True

        # Signal handlers for graceful shutdown on Windows & Linux
        def handle_signal(sig, frame):
            print("\n[!] Shutdown signal received. Stopping MQTT publisher...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info("Starting MQTT publication loop (Interval: %.1fs)...", self.publish_interval)
        iteration = 0

        try:
            while self._running:
                iteration += 1
                snapshot = self.publish_step(dt=self.publish_interval)

                summary = snapshot["production_summary"]
                logger.info(
                    "Iteration %03d | Active: %d | Warning: %d | Failed: %d | Total Prod: %d",
                    iteration,
                    summary["active_machines"],
                    summary["warning_machines"],
                    summary["failed_machines"],
                    summary["total_production"],
                )

                if max_iterations and iteration >= max_iterations:
                    logger.info("Reached maximum iterations (%d). Stopping loop.", max_iterations)
                    break

                time.sleep(self.publish_interval)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.disconnect()


if __name__ == "__main__":
    # If run directly as a script, execute live publisher
    pub = FactoryMqttPublisher()
    pub.start_publishing()
