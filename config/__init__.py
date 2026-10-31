"""Configuration loader for the Digital Twin project."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

CONFIG_FILE = BASE_DIR / "config" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml with environment overrides."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Allow environment variable overrides for HiveMQ Cloud / Local MQTT
    mqtt = config.setdefault("mqtt", {})
    if "MQTT_BROKER_HOST" in os.environ and os.environ["MQTT_BROKER_HOST"].strip():
        mqtt["broker"] = os.environ["MQTT_BROKER_HOST"].strip()
    if "MQTT_BROKER_PORT" in os.environ and os.environ["MQTT_BROKER_PORT"].strip():
        mqtt["port"] = int(os.environ["MQTT_BROKER_PORT"].strip())
    if "MQTT_USERNAME" in os.environ and os.environ["MQTT_USERNAME"].strip():
        mqtt["username"] = os.environ["MQTT_USERNAME"].strip()
    if "MQTT_PASSWORD" in os.environ and os.environ["MQTT_PASSWORD"].strip():
        mqtt["password"] = os.environ["MQTT_PASSWORD"].strip()
    if "MQTT_TLS" in os.environ and os.environ["MQTT_TLS"].strip():
        val = os.environ["MQTT_TLS"].strip().lower()
        mqtt["tls"] = val in ("true", "1", "yes", "on")
    if "MQTT_KEEPALIVE" in os.environ and os.environ["MQTT_KEEPALIVE"].strip():
        mqtt["keepalive"] = int(os.environ["MQTT_KEEPALIVE"].strip())

    return config


# Cached singleton configuration dictionary
CONFIG = load_config()
