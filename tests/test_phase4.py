"""Unit and integration tests for Phase 4: MQTT Subscriber / Telemetry Receiver."""

import json
import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from mqtt.subscriber import FactoryMqttSubscriber
from mqtt.publisher import FactoryMqttPublisher


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def subscriber(config):
    sub = FactoryMqttSubscriber(config=config)
    yield sub
    sub.disconnect()


def test_subscriber_initialization(config):
    """Verify subscriber initializes with topics, broker settings, and empty caches."""
    sub = FactoryMqttSubscriber(config=config)
    assert sub.broker_host == "localhost"
    assert sub.broker_port == 1883
    assert "factory/+/telemetry" in sub.topics
    assert sub.total_received == 0
    assert sub.valid_messages == 0


def test_payload_validation_machine_telemetry(subscriber):
    """Verify machine telemetry schema validation detects valid and malformed payloads."""
    valid_machine_payload = {
        "timestamp": "2026-09-30T17:00:00+00:00",
        "machine_id": "M01",
        "machine_state": "RUNNING",
        "temperature": 65.4,
        "vibration": 2.45,
        "current": 8.5,
        "voltage": 230.0,
        "rpm": 1795.0,
        "pressure": 6.0,
        "power": 3.2,
        "production_count": 10,
        "good_products": 10,
        "defect_count": 0,
        "cycle_time": 12.0,
    }
    assert subscriber.validate_payload("factory/machine1/telemetry", valid_machine_payload) is True

    # Missing temperature
    invalid_missing_temp = valid_machine_payload.copy()
    del invalid_missing_temp["temperature"]
    assert subscriber.validate_payload("factory/machine1/telemetry", invalid_missing_temp) is False

    # Invalid type for vibration (string instead of float)
    invalid_type = valid_machine_payload.copy()
    invalid_type["vibration"] = "not-a-number"
    assert subscriber.validate_payload("factory/machine1/telemetry", invalid_type) is False


def test_payload_validation_production_telemetry(subscriber):
    """Verify factory production schema validation."""
    valid_prod = {
        "timestamp": "2026-09-30T17:00:00+00:00",
        "factory_name": "Smart Factory Virtual Line 1",
        "factory_status": "NORMAL",
        "production_summary": {"total_production": 50, "defect_rate": 0.01},
    }
    assert subscriber.validate_payload("factory/production/telemetry", valid_prod) is True

    invalid_prod = {"timestamp": "2026-09-30T17:00:00+00:00"}
    assert subscriber.validate_payload("factory/production/telemetry", invalid_prod) is False


def test_end_to_end_subscriber_reception(config):
    """Verify subscriber receives live MQTT messages from publisher, validates and routes them."""
    sub = FactoryMqttSubscriber(config=config)
    pub = FactoryMqttPublisher(config=config)

    handled_telemetries = []

    def telemetry_callback(topic, data):
        handled_telemetries.append((topic, data))

    sub.register_handler(telemetry_callback)

    try:
        assert sub.connect(timeout=3.0) is True
        assert pub.connect(timeout=3.0) is True

        # Allow subscription to register
        time.sleep(0.3)

        # Publish 2 steps
        pub.publish_step(dt=1.0)
        pub.publish_step(dt=1.0)

        # Wait up to 3 seconds for all 8 messages (6 machine + 2 production) to arrive
        start_wait = time.time()
        while (len(handled_telemetries) < 8 or sub.latest_production is None) and (time.time() - start_wait) < 3.0:
            time.sleep(0.05)

        # Check subscriber metrics
        assert sub.total_received >= 6
        assert sub.valid_messages >= 6
        assert sub.invalid_messages == 0

        # Check machine caches
        assert "M01" in sub.latest_telemetry
        assert "M02" in sub.latest_telemetry
        assert "M03" in sub.latest_telemetry
        assert sub.latest_production is not None

        # Verify handled records
        machine_ids = [
            data["machine_id"]
            for topic, data in handled_telemetries
            if "/telemetry" in topic and "machine_id" in data
        ]
        assert "M01" in machine_ids
        assert "M02" in machine_ids
        assert "M03" in machine_ids

    finally:
        pub.disconnect()
        sub.disconnect()
