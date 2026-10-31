"""MQTT Subscriber and Telemetry Receiver for Smart Factory operations.

Connects to Mosquitto MQTT broker, subscribes to factory telemetry topics,
parses, validates, and dispatches incoming JSON payloads to registered handlers.
"""

from datetime import datetime, timezone
import json
import logging
import signal
import sys
import time
import ssl
from typing import Any, Callable, Dict, List, Optional

import paho.mqtt.client as mqtt
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MqttSubscriber")

# Required attributes for incoming telemetry verification
REQUIRED_MACHINE_FIELDS = [
    "timestamp",
    "machine_id",
    "machine_state",
    "temperature",
    "vibration",
    "current",
    "voltage",
    "rpm",
    "pressure",
    "power",
    "production_count",
    "good_products",
    "defect_count",
    "cycle_time",
]

REQUIRED_PRODUCTION_FIELDS = [
    "timestamp",
    "factory_name",
    "factory_status",
    "production_summary",
]


class FactoryMqttSubscriber:
    """Subscribes to factory MQTT topics, validates payloads, and routes telemetry."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize subscriber with broker and topic configuration.

        Args:
            config: Optional config dict. If None, loaded via load_config().
        """
        self.config: Dict[str, Any] = config or load_config()

        mqtt_cfg = self.config.get("mqtt", {})
        self.broker_host: str = mqtt_cfg.get("broker", "localhost")
        self.broker_port: int = int(mqtt_cfg.get("port", 1883))
        self.username: Optional[str] = mqtt_cfg.get("username") or None
        self.password: Optional[str] = mqtt_cfg.get("password") or None
        self.use_tls: bool = bool(mqtt_cfg.get("tls", False)) or (self.broker_port == 8883)
        self.keepalive: int = int(mqtt_cfg.get("keepalive", 60))

        # Default subscription topics
        self.topics: List[str] = [
            "factory/+/telemetry",
            "factory/+/status",
            "factory/alerts",
        ]

        # MQTT Client initialization with paho-mqtt v2 Callback API
        self.client_id: str = f"factory_subscriber_{int(time.time())}"
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

        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # State tracking
        self.is_connected: bool = False
        self._running: bool = False

        # Metrics and reception cache
        self.total_received: int = 0
        self.valid_messages: int = 0
        self.invalid_messages: int = 0
        self.messages_by_topic: Dict[str, int] = {}
        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self.latest_production: Optional[Dict[str, Any]] = None

        # Custom handler hooks for downstream pipelines (SQLite, AI, Digital Twin)
        self._handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def register_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for processed and validated telemetry payloads."""
        self._handlers.append(handler)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle connection and subscribe to factory topics."""
        if rc == 0:
            self.is_connected = True
            logger.info("Connected to MQTT broker at %s:%d", self.broker_host, self.broker_port)
            # Subscribe to all configured topics
            for topic in self.topics:
                client.subscribe(topic, qos=1)
                logger.info("Subscribed to MQTT topic: %s", topic)
        else:
            self.is_connected = False
            logger.error("Failed to connect to MQTT broker with rc: %s", rc)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        """Handle disconnection from broker."""
        self.is_connected = False
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT broker (rc=%s). Auto-reconnecting...", rc)
        else:
            logger.info("Disconnected from MQTT broker.")

    def _on_message(self, client, userdata, message):
        """Process, decode, parse, and validate incoming MQTT message."""
        self.total_received += 1
        topic = message.topic
        self.messages_by_topic[topic] = self.messages_by_topic.get(topic, 0) + 1

        # 1. Decode UTF-8 string
        try:
            payload_str = message.payload.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error("Failed to decode UTF-8 payload on %s: %s", topic, e)
            self.invalid_messages += 1
            return

        # 2. Parse JSON
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError as e:
            logger.error("Malformed JSON received on %s: %s", topic, e)
            self.invalid_messages += 1
            return

        if not isinstance(data, dict):
            logger.error("Expected JSON object but received %s on %s", type(data).__name__, topic)
            self.invalid_messages += 1
            return

        # 3. Validate Schema
        is_valid = self.validate_payload(topic, data)
        if not is_valid:
            self.invalid_messages += 1
            return

        self.valid_messages += 1

        # 4. Cache state
        if topic == "factory/production/telemetry":
            self.latest_production = data
        elif "telemetry" in topic:
            machine_id = data.get("machine_id", "unknown")
            self.latest_telemetry[machine_id] = data

        # 5. Dispatch to registered downstream handlers
        for handler in self._handlers:
            try:
                handler(topic, data)
            except Exception as e:
                logger.error("Handler error for topic %s: %s", topic, e)

    def validate_payload(self, topic: str, data: Dict[str, Any]) -> bool:
        """Validate payload attributes against required schema definitions.

        Args:
            topic: The MQTT topic the message was published to.
            data: Parsed JSON dictionary.

        Returns:
            True if schema is valid, False otherwise.
        """
        if topic == "factory/production/telemetry":
            missing = [k for k in REQUIRED_PRODUCTION_FIELDS if k not in data]
            if missing:
                logger.warning("Production payload missing fields: %s", missing)
                return False
            return True

        if "/telemetry" in topic:
            missing = [k for k in REQUIRED_MACHINE_FIELDS if k not in data]
            if missing:
                logger.warning("Machine telemetry missing fields on %s: %s", topic, missing)
                return False
            # Verify numeric types
            for numeric_field in ["temperature", "vibration", "current", "rpm"]:
                if not isinstance(data.get(numeric_field), (int, float)):
                    logger.warning("Field '%s' must be numeric, got %s", numeric_field, type(data.get(numeric_field)))
                    return False
            return True

        # Status and alert topics
        return True

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to MQTT broker and start network loop."""
        try:
            logger.info(
                "Subscriber connecting to %s:%d (TLS: %s, Auth: %s)...",
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

            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.05)

            return self.is_connected
        except Exception as e:
            logger.error("Subscriber connection exception: %s", e)
            return False

    def disconnect(self):
        """Disconnect subscriber and stop background thread."""
        self._running = False
        try:
            self.client.loop_stop()
            if self.is_connected:
                self.client.disconnect()
            self.is_connected = False
            logger.info("Subscriber disconnected cleanly.")
        except Exception as e:
            logger.error("Subscriber disconnect error: %s", e)

    def start_listening(self, duration_seconds: Optional[float] = None):
        """Start listening for incoming telemetry loop.

        Args:
            duration_seconds: Optional duration to listen. If None, runs indefinitely.
        """
        if not self.is_connected:
            if not self.connect():
                logger.error("Cannot listen: connection failed.")
                return

        self._running = True

        def handle_signal(sig, frame):
            print("\n[!] Subscriber shutdown signal received...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info("Subscriber listening on topics: %s", self.topics)
        start_time = time.time()

        try:
            while self._running:
                time.sleep(0.5)
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info("Listening duration (%.1fs) reached.", duration_seconds)
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.disconnect()


if __name__ == "__main__":
    # If run directly as a script, listen for live telemetry
    sub = FactoryMqttSubscriber()
    sub.start_listening()
