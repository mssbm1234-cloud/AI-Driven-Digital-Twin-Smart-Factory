"""Demonstration script for Phase 2: Factory Simulator.

Executes consecutive simulation steps, demonstrates normal telemetry,
gradual degradation, failure injection, and maintenance recovery.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from simulator.factory_simulator import FactorySimulator


def run_demo():
    print("=" * 70)
    print("AI-Driven Digital Twin - Phase 2 Factory Simulator Demonstration")
    print("=" * 70)

    sim = FactorySimulator()
    print(f"Factory: {sim.factory_name}")
    print(f"Machines configured: {list(sim.machines.keys())}")
    print("=" * 70)

    records = []

    # 1. Normal steps (Steps 1 to 4)
    print("\n--- [Phase 2.1] Normal Steady-State Operations (Steps 1-4) ---")
    for i in range(1, 5):
        data = sim.step(dt=1.0)
        records.append(data)
        m01 = data["machines"]["M01"]
        print(
            f"Step {i:02d} | M01 State: {m01['machine_state']:<8} | "
            f"Temp: {m01['temperature']:5.1f}C | Vib: {m01['vibration']:4.2f} mm/s | "
            f"Curr: {m01['current']:4.1f}A | RPM: {m01['rpm']:6.1f} | "
            f"Prod: {m01['production_count']}"
        )

    # 2. Inject degradation on M02 (Steps 5 to 7)
    print("\n--- [Phase 2.2] Inducing Degradation on M02 (Steps 5-7) ---")
    sim.inject_degradation("M02", rate=0.20)
    for i in range(5, 8):
        data = sim.step(dt=2.0)
        records.append(data)
        m02 = data["machines"]["M02"]
        print(
            f"Step {i:02d} | M02 State: {m02['machine_state']:<8} | "
            f"Temp: {m02['temperature']:5.1f}C | Vib: {m02['vibration']:4.2f} mm/s | "
            f"Curr: {m02['current']:4.1f}A | RPM: {m02['rpm']:6.1f} | "
            f"Degradation: {m02['degradation_level']*100:4.1f}%"
        )

    # 3. Inject Failure on M01 (Step 8)
    print("\n--- [Phase 2.3] Simulating Breakdown Failure on M01 (Step 8) ---")
    sim.inject_failure("M01")
    data = sim.step(dt=1.0)
    records.append(data)
    m01 = data["machines"]["M01"]
    print(
        f"Step 08 | M01 State: {m01['machine_state']:<8} | "
        f"Temp: {m01['temperature']:5.1f}C | Vib: {m01['vibration']:4.2f} mm/s | "
        f"RPM: {m01['rpm']:6.1f} | Downtime: {m01['downtime_seconds']}s"
    )

    # 4. Perform Maintenance and Recovery on M01 & M02 (Steps 9 to 10)
    print("\n--- [Phase 2.4] Performing Maintenance & Recovery (Steps 9-10) ---")
    sim.repair_machine("M01")
    sim.repair_machine("M02")
    for i in range(9, 11):
        data = sim.step(dt=1.0)
        records.append(data)
        m01 = data["machines"]["M01"]
        m02 = data["machines"]["M02"]
        print(
            f"Step {i:02d} | M01: {m01['machine_state']:<8} (Temp: {m01['temperature']}C) | "
            f"M02: {m02['machine_state']:<8} (Temp: {m02['temperature']}C)"
        )

    # Output Sample JSON Telemetry Record
    print("\n" + "=" * 70)
    print(f"Total Simulation Records Captured: {len(records)}")
    print("Sample Telemetry JSON for Machine M01 (Step 1):")
    print(json.dumps(records[0]["machines"]["M01"], indent=2))
    print("\nSample Factory Production Summary (Step 10):")
    print(json.dumps(records[-1]["production_summary"], indent=2))
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
