"""Unit and integration tests for Phase 3: MQTT Publisher."""

import json
import sys
import time
from pathlib import Path
import pytest
import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from mqtt.publisher import FactoryMqttPublisher
from simulator.factory_simulator import FactorySimulator


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def publisher(config):
    pub = FactoryMqttPublisher(config=config)
    yield pub
    pub.disconnect()


def test_publisher_initialization(config):
    """Verify publisher initializes topics, broker configuration, and simulator."""
    pub = FactoryMqttPublisher(config=config)
    assert pub.broker_host == "localhost"
    assert pub.broker_port == 1883
    assert pub.machine_telemetry_topics["M01"] == "factory/machine1/telemetry"
    assert pub.machine_telemetry_topics["M02"] == "factory/machine2/telemetry"
    assert pub.machine_telemetry_topics["M03"] == "factory/machine3/telemetry"
    assert pub.production_topic == "factory/production/telemetry"
    assert isinstance(pub.simulator, FactorySimulator)


def test_publisher_connect_and_disconnect(publisher):
    """Verify publisher connects to local Mosquitto broker and disconnects cleanly."""
    connected = publisher.connect(timeout=3.0)
    assert connected is True
    assert publisher.is_connected is True

    publisher.disconnect()
    assert publisher.is_connected is False


def test_mqtt_publish_and_receive_end_to_end(publisher):
    """Verify published messages are successfully received on all factory topics."""
    received_messages = {}

    # Set up test subscriber client
    sub_client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="test_sub_phase3",
    )

    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            received_messages[message.topic] = payload
        except Exception as e:
            received_messages[message.topic] = str(e)

    sub_client.on_message = on_message
    sub_client.connect("localhost", 1883, 60)
    sub_client.subscribe("factory/#")
    sub_client.loop_start()

    # Wait briefly for subscription to register
    time.sleep(0.3)

    try:
        # Connect publisher and publish one step
        assert publisher.connect(timeout=3.0) is True
        publisher.publish_step(dt=1.0)

        # Wait for messages to arrive (up to 2 seconds)
        start = time.time()
        expected_topics = {
            "factory/machine1/telemetry",
            "factory/machine2/telemetry",
            "factory/machine3/telemetry",
            "factory/production/telemetry",
        }
        while (
            not expected_topics.issubset(set(received_messages.keys()))
            and (time.time() - start) < 2.0
        ):
            time.sleep(0.05)

        # Verify all expected topics were received
        for topic in expected_topics:
            assert topic in received_messages, f"Topic '{topic}' was not received"
            data = received_messages[topic]
            assert isinstance(data, dict)
            assert "timestamp" in data

        # Verify machine payload contents
        m1_data = received_messages["factory/machine1/telemetry"]
        assert m1_data["machine_id"] == "M01"
        assert "temperature" in m1_data
        assert "vibration" in m1_data
        assert "rpm" in m1_data

        m2_data = received_messages["factory/machine2/telemetry"]
        assert m2_data["machine_id"] == "M02"

        m3_data = received_messages["factory/machine3/telemetry"]
        assert m3_data["machine_id"] == "M03"

        prod_data = received_messages["factory/production/telemetry"]
        assert "production_summary" in prod_data
        assert "active_machines" in prod_data["production_summary"]

    finally:
        sub_client.loop_stop()
        sub_client.disconnect()
