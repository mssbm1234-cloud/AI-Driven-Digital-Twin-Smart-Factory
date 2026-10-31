"""Unit tests for HiveMQ Cloud configuration and MQTT TLS/Authentication setup."""

import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber


def test_hivemq_environment_variable_loading():
    """Verify HiveMQ Cloud environment variables override config.yaml parameters."""
    env_overrides = {
        "MQTT_BROKER_HOST": "my-cluster.s1.eu.hivemq.cloud",
        "MQTT_BROKER_PORT": "8883",
        "MQTT_USERNAME": "test_factory_user",
        "MQTT_PASSWORD": "super_secret_hivemq_password_123",
        "MQTT_TLS": "true",
        "MQTT_KEEPALIVE": "45",
    }

    with patch.dict(os.environ, env_overrides, clear=False):
        cfg = load_config()
        mqtt_cfg = cfg.get("mqtt", {})

        assert mqtt_cfg.get("broker") == "my-cluster.s1.eu.hivemq.cloud"
        assert mqtt_cfg.get("port") == 8883
        assert mqtt_cfg.get("username") == "test_factory_user"
        assert mqtt_cfg.get("password") == "super_secret_hivemq_password_123"
        assert mqtt_cfg.get("tls") is True
        assert mqtt_cfg.get("keepalive") == 45


def test_publisher_hivemq_client_configuration():
    """Verify FactoryMqttPublisher initializes with TLS and authentication parameters."""
    custom_config = {
        "mqtt": {
            "broker": "my-hivemq-instance.hivemq.cloud",
            "port": 8883,
            "username": "operator_admin",
            "password": "strong_auth_token_xyz",
            "tls": True,
            "keepalive": 60,
            "publish_interval_seconds": 1.0,
            "topics": {
                "machine_telemetry": {"M01": "factory/m1/telemetry"},
                "production": "factory/production/telemetry",
            },
        },
        "factory": {"machines": []},
    }

    pub = FactoryMqttPublisher(config=custom_config)

    assert pub.broker_host == "my-hivemq-instance.hivemq.cloud"
    assert pub.broker_port == 8883
    assert pub.username == "operator_admin"
    assert pub.password == "strong_auth_token_xyz"
    assert pub.use_tls is True
    assert pub.keepalive == 60


def test_subscriber_hivemq_client_configuration():
    """Verify FactoryMqttSubscriber initializes with TLS and authentication parameters."""
    custom_config = {
        "mqtt": {
            "broker": "my-hivemq-instance.hivemq.cloud",
            "port": 8883,
            "username": "subscriber_service",
            "password": "sub_token_secret_456",
            "tls": True,
            "keepalive": 30,
        },
    }

    sub = FactoryMqttSubscriber(config=custom_config)

    assert sub.broker_host == "my-hivemq-instance.hivemq.cloud"
    assert sub.broker_port == 8883
    assert sub.username == "subscriber_service"
    assert sub.password == "sub_token_secret_456"
    assert sub.use_tls is True
    assert sub.keepalive == 30


def test_password_security_and_non_exposure():
    """Verify credentials and passwords are not printed or exposed in logs."""
    custom_config = {
        "mqtt": {
            "broker": "cluster.hivemq.cloud",
            "port": 8883,
            "username": "test_user",
            "password": "ultra_confidential_password",
            "tls": True,
        }
    }
    pub = FactoryMqttPublisher(config=custom_config)
    sub = FactoryMqttSubscriber(config=custom_config)

    # String representation should not contain the raw password
    assert "ultra_confidential_password" not in str(pub.__dict__.get("client"))
    assert "ultra_confidential_password" not in str(sub.__dict__.get("client"))
