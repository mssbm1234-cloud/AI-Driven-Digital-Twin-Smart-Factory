"""Unit and integration tests for Phase 2: Factory Simulator."""

import json
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from simulator.machine import Machine, MachineState
from simulator.factory_simulator import FactorySimulator


@pytest.fixture
def sample_config():
    return load_config()


@pytest.fixture
def single_machine(sample_config):
    m_cfg = sample_config["factory"]["machines"][0]
    return Machine(m_cfg)


def test_machine_initialization(sample_config):
    """Verify machine initializes with expected nominal parameters and state."""
    m_cfg = sample_config["factory"]["machines"][0]
    m = Machine(m_cfg)
    assert m.machine_id == "M01"
    assert m.state == MachineState.RUNNING
    assert m.degradation_level == 0.0
    assert m.production_count == 0
    assert m.downtime_seconds == 0.0


def test_telemetry_fields_and_json_serializable(single_machine):
    """Verify all required telemetry fields exist and serialize to valid JSON."""
    t = single_machine.step(dt=1.0)
    required_fields = [
        "timestamp",
        "machine_id",
        "machine_name",
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
        "defect_rate",
        "cycle_time",
        "downtime_seconds",
    ]
    for field in required_fields:
        assert field in t, f"Missing required telemetry field: {field}"

    # Verify JSON serializability
    serialized = json.dumps(t)
    deserialized = json.loads(serialized)
    assert deserialized["machine_id"] == "M01"
    assert deserialized["machine_state"] == "RUNNING"


def test_realistic_noise_variance(single_machine):
    """Verify telemetry varies with realistic noise around nominal values."""
    readings = [single_machine.step(dt=1.0)["temperature"] for _ in range(5)]
    # Values should not be identical (demonstrating dynamic noise)
    assert len(set(readings)) > 1
    # Values should remain within realistic +/- 3 degrees of nominal
    for r in readings:
        assert abs(r - single_machine.nominal["temperature"]) < 3.0


def test_gradual_degradation(single_machine):
    """Verify degradation increases temperature and vibration while dropping RPM."""
    initial_temp = single_machine.nominal["temperature"]
    initial_vib = single_machine.nominal["vibration"]
    initial_rpm = single_machine.nominal["rpm"]

    single_machine.start_degradation(rate=0.08)
    for _ in range(12):
        t = single_machine.step(dt=1.0)

    assert single_machine.degradation_level > 0.5
    assert t["temperature"] > initial_temp
    assert t["vibration"] > initial_vib
    assert t["rpm"] < initial_rpm
    assert t["machine_state"] in (MachineState.WARNING.value, MachineState.FAILURE.value)


def test_simulated_failure_and_downtime(single_machine):
    """Verify failure state stops production and increments downtime."""
    single_machine.trigger_failure()
    assert single_machine.state == MachineState.FAILURE

    t1 = single_machine.step(dt=2.0)
    assert t1["machine_state"] == "FAILURE"
    assert t1["downtime_seconds"] >= 2.0
    assert t1["rpm"] < 200.0  # RPM dropped


def test_maintenance_recovery(single_machine):
    """Verify performing maintenance clears degradation and restores nominal parameters."""
    single_machine.trigger_failure()
    single_machine.perform_maintenance(restore_state=MachineState.RUNNING)

    assert single_machine.state == MachineState.RUNNING
    assert single_machine.degradation_level == 0.0

    t = single_machine.step(dt=1.0)
    assert t["machine_state"] == "RUNNING"
    assert abs(t["temperature"] - single_machine.nominal["temperature"]) < 2.0


def test_factory_simulator_all_three_machines(sample_config):
    """Verify FactorySimulator runs M01, M02, M03 and produces factory-wide summary."""
    sim = FactorySimulator(sample_config)
    assert set(sim.machines.keys()) == {"M01", "M02", "M03"}

    snapshot = sim.step(dt=1.0)
    assert "timestamp" in snapshot
    assert "factory_name" in snapshot
    assert "factory_status" in snapshot
    assert "machines" in snapshot
    assert len(snapshot["machines"]) == 3
    assert "M01" in snapshot["machines"]
    assert "M02" in snapshot["machines"]
    assert "M03" in snapshot["machines"]

    # Verify summary counters
    prod_summary = snapshot["production_summary"]
    assert "total_production" in prod_summary
    assert "active_machines" in prod_summary
    assert prod_summary["active_machines"] == 3
    assert prod_summary["failed_machines"] == 0
