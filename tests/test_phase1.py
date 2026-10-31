"""Phase 1 Verification Test Script.

Validates:
1. Python environment and required third-party libraries (PyTorch, Pandas, NumPy, paho-mqtt, FastAPI, Scikit-learn, etc.)
2. Project directory structure and configuration loading.
3. Local Mosquitto MQTT broker connectivity.
"""

import sys
import socket
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Safe status symbols
OK_MARK = "[OK]"
FAIL_MARK = "[FAILED]"
WARN_MARK = "[WARN]"


def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]}", end=" ")
    if sys.version_info >= (3, 9):
        print(f"{OK_MARK}")
        return True
    print(f"{FAIL_MARK} - Python 3.9+ required")
    return False



def check_dependencies() -> bool:
    print("[*] Checking Core Libraries:")
    packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("paho.mqtt.client", "Paho-MQTT"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
    ]
    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name, fromlist=["__version__"])
            ver = getattr(mod, "__version__", "installed")
            print(f"    - {display_name:<15}: {ver:<15} {OK_MARK}")
        except ImportError as e:
            print(f"    - {display_name:<15}: NOT FOUND ({e}) {FAIL_MARK}")
            all_ok = False
    return all_ok


def check_project_structure() -> bool:
    print("[*] Checking Project Folders:")
    required_dirs = [
        "simulator",
        "mqtt",
        "database",
        "ai",
        "digital_twin",
        "analytics",
        "alerts",
        "api",
        "config",
        "tests",
        "data",
    ]
    all_ok = True
    for folder in required_dirs:
        dir_path = PROJECT_ROOT / folder
        dir_path.mkdir(exist_ok=True)
        if dir_path.is_dir():
            print(f"    - {folder:<15}: {OK_MARK}")
        else:
            print(f"    - {folder:<15}: {FAIL_MARK} Missing")
            all_ok = False
    return all_ok


def check_config_loading() -> bool:
    print("[*] Checking Configuration Loading:")
    try:
        from config import load_config
        cfg = load_config()
        broker = cfg.get("mqtt", {}).get("broker", "unknown")
        machines = cfg.get("factory", {}).get("machines", [])
        print(f"    - Loaded config for '{cfg.get('factory', {}).get('name')}'")
        print(f"    - MQTT Broker: {broker}")
        print(f"    - Configured Machines: {[m['id'] for m in machines]}")
        print(f"    - Config Status: {OK_MARK}")
        return True
    except Exception as e:
        print(f"    - Config Loading Failed: {e} {FAIL_MARK}")
        return False


def check_mqtt_broker() -> bool:
    print("[*] Checking Mosquitto Broker Connectivity (localhost:1883):", end=" ")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(("127.0.0.1", 1883))
        s.close()
        print(f"{OK_MARK} Broker is listening!")
        return True
    except Exception as e:
        print(f"{WARN_MARK} Broker not reachable ({e}). Ensure Mosquitto service is started.")
        return False


def main():
    print("=" * 60)
    print("AI-Driven Digital Twin - Phase 1 Environment Verification")
    print("=" * 60)

    results = [
        check_python_version(),
        check_dependencies(),
        check_project_structure(),
        check_config_loading(),
        check_mqtt_broker(),
    ]

    print("=" * 60)
    if all(results):
        print(">>> PHASE 1 VERIFICATION PASSED: All requirements are ready! <<<")
        sys.exit(0)
    else:
        print(">>> PHASE 1 VERIFICATION INCOMPLETE: Resolve items marked [FAILED]. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
