"""Industrial machine simulator model for Smart Factory operations.

Simulates realistic telemetry, degradation dynamics, failure modes,
maintenance recovery, and production counters for individual machines.
"""

from datetime import datetime, timezone
from enum import Enum
import random
from typing import Any, Dict, Optional


class MachineState(str, Enum):
    """Enumeration of realistic operational machine states."""
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    WARNING = "WARNING"
    FAILURE = "FAILURE"
    MAINTENANCE = "MAINTENANCE"


class Machine:
    """Reusable simulation model for an industrial machine."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the machine with configuration parameters.

        Args:
            config: Machine configuration dictionary containing id, name, type,
                    nominal sensor values, noise amplitudes, and thresholds.
        """
        self.machine_id: str = config.get("id", "M00")
        self.name: str = config.get("name", "Generic Machine")
        self.machine_type: str = config.get("type", "generic")

        # Operational parameters
        self.nominal: Dict[str, float] = config.get("nominal", {
            "temperature": 65.0,
            "vibration": 2.5,
            "current": 8.5,
            "voltage": 230.0,
            "rpm": 1800.0,
            "pressure": 6.0,
            "power": 3.2,
            "cycle_time": 12.0,
        })

        self.noise: Dict[str, float] = config.get("noise", {
            "temperature": 0.4,
            "vibration": 0.1,
            "current": 0.15,
            "voltage": 1.0,
            "rpm": 10.0,
            "pressure": 0.08,
            "power": 0.05,
        })

        self.thresholds: Dict[str, Dict[str, float]] = config.get("thresholds", {
            "temperature": {"warning": 78.0, "failure": 90.0},
            "vibration": {"warning": 4.2, "failure": 6.5},
            "current": {"warning": 12.0, "failure": 16.0},
            "rpm_min": {"warning": 1550.0, "failure": 1200.0},
        })

        # Machine state & degradation
        self.state: MachineState = MachineState.RUNNING
        self.is_degrading: bool = False
        self.degradation_level: float = 0.0  # 0.0 = healthy, 1.0 = severe breakdown
        self.degradation_rate: float = 0.015  # increase per second when degrading

        # Production counters
        self.production_count: int = 0
        self.good_products: int = 0
        self.defective_products: int = 0
        self.downtime_seconds: float = 0.0
        self.total_operating_seconds: float = 0.0

        # Cycle time accumulator for discreet part completion
        self._cycle_progress: float = 0.0

        # Current sensor snapshot (smoothed values)
        self.temperature: float = self.nominal["temperature"]
        self.vibration: float = self.nominal["vibration"]
        self.current: float = self.nominal["current"]
        self.voltage: float = self.nominal["voltage"]
        self.rpm: float = self.nominal["rpm"]
        self.pressure: float = self.nominal["pressure"]
        self.power: float = self.nominal["power"]
        self.cycle_time: float = self.nominal["cycle_time"]

    def set_state(self, state: MachineState):
        """Explicitly update the machine state."""
        self.state = state

    def start_degradation(self, rate: Optional[float] = None):
        """Trigger gradual mechanical/electrical wear and degradation."""
        self.is_degrading = True
        if rate is not None:
            self.degradation_rate = rate

    def stop_degradation(self):
        """Halt progressive degradation."""
        self.is_degrading = False

    def trigger_failure(self, failure_type: str = "critical_overload"):
        """Instantly trip the machine into FAILURE state."""
        self.state = MachineState.FAILURE
        self.is_degrading = False
        # In catastrophic failure, spike temp/vibration and drop RPM
        self.rpm = max(0.0, self.nominal["rpm"] * 0.1)
        self.vibration = self.thresholds.get("vibration", {}).get("failure", 7.0) * 1.15
        self.temperature = self.thresholds.get("temperature", {}).get("failure", 95.0) * 1.05
        self.current = self.thresholds.get("current", {}).get("failure", 18.0) * 1.1

    def perform_maintenance(self, restore_state: MachineState = MachineState.RUNNING):
        """Simulate maintenance service: repairs faults and restores nominal baselines."""
        self.state = restore_state
        self.is_degrading = False
        self.degradation_level = 0.0
        self.temperature = self.nominal["temperature"]
        self.vibration = self.nominal["vibration"]
        self.current = self.nominal["current"]
        self.voltage = self.nominal["voltage"]
        self.rpm = self.nominal["rpm"]
        self.pressure = self.nominal["pressure"]
        self.power = self.nominal["power"]
        self.cycle_time = self.nominal["cycle_time"]
        self._cycle_progress = 0.0

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """Advance machine simulation by dt seconds and return current telemetry.

        Args:
            dt: Time delta in seconds (typically 1.0).

        Returns:
            Dictionary containing realistic timestamped telemetry.
        """
        self.total_operating_seconds += dt

        # Handle degradation progression
        if self.is_degrading and self.state not in (MachineState.FAILURE, MachineState.MAINTENANCE):
            self.degradation_level = min(1.0, self.degradation_level + self.degradation_rate * dt)

        # Dynamic simulation depending on state
        if self.state in (MachineState.RUNNING, MachineState.WARNING):
            self._simulate_operating(dt)
        elif self.state == MachineState.FAILURE:
            self._simulate_failure(dt)
        elif self.state == MachineState.MAINTENANCE:
            self._simulate_maintenance(dt)
        elif self.state == MachineState.IDLE:
            self._simulate_idle(dt)

        # Calculate defect rate
        defect_rate = (
            self.defective_products / self.production_count
            if self.production_count > 0
            else 0.0
        )

        # Format standardized telemetry payload
        telemetry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_name": self.name,
            "machine_state": self.state.value,
            "temperature": round(self.temperature, 2),
            "vibration": round(self.vibration, 2),
            "current": round(self.current, 2),
            "voltage": round(self.voltage, 2),
            "rpm": round(self.rpm, 1),
            "pressure": round(self.pressure, 2),
            "power": round(self.power, 2),
            "production_count": self.production_count,
            "good_products": self.good_products,
            "defect_count": self.defective_products,
            "defect_rate": round(defect_rate, 4),
            "cycle_time": round(self.cycle_time, 2),
            "downtime_seconds": round(self.downtime_seconds, 1),
            "degradation_level": round(self.degradation_level, 4),
        }

        return telemetry

    def _simulate_operating(self, dt: float):
        """Simulate running machine physics with noise and degradation."""
        d = self.degradation_level

        # Baseline + degradation delta + realistic Gaussian noise
        temp_delta = d * 22.0
        vib_delta = d * 3.5
        curr_delta = d * 6.0
        rpm_delta = -d * (self.nominal["rpm"] * 0.40)  # RPM drops under heavy wear
        power_delta = curr_delta * 0.25

        target_temp = self.nominal["temperature"] + temp_delta + random.gauss(0, self.noise["temperature"])
        target_vib = self.nominal["vibration"] + vib_delta + random.gauss(0, self.noise["vibration"])
        target_curr = self.nominal["current"] + curr_delta + random.gauss(0, self.noise["current"])
        target_volt = self.nominal["voltage"] + random.gauss(0, self.noise["voltage"])
        target_rpm = max(100.0, self.nominal["rpm"] + rpm_delta + random.gauss(0, self.noise["rpm"]))
        target_press = max(0.5, self.nominal["pressure"] + random.gauss(0, self.noise["pressure"]))
        target_power = max(0.2, self.nominal["power"] + power_delta + random.gauss(0, self.noise["power"]))
        target_cycle = self.nominal["cycle_time"] * (1.0 + d * 0.5)

        # First-order inertia smoothing for realistic physical responsiveness
        alpha = min(1.0, 0.35 * dt)
        self.temperature += alpha * (target_temp - self.temperature)
        self.vibration += alpha * (target_vib - self.vibration)
        self.current += alpha * (target_curr - self.current)
        self.voltage += alpha * (target_volt - self.voltage)
        self.rpm += alpha * (target_rpm - self.rpm)
        self.pressure += alpha * (target_press - self.pressure)
        self.power += alpha * (target_power - self.power)
        self.cycle_time = target_cycle

        # Automatic threshold-based state transitions
        warn_temp = self.thresholds.get("temperature", {}).get("warning", 78.0)
        fail_temp = self.thresholds.get("temperature", {}).get("failure", 90.0)
        warn_vib = self.thresholds.get("vibration", {}).get("warning", 4.2)
        fail_vib = self.thresholds.get("vibration", {}).get("failure", 6.5)

        if self.temperature >= fail_temp or self.vibration >= fail_vib or d >= 0.95:
            self.state = MachineState.FAILURE
        elif self.temperature >= warn_temp or self.vibration >= warn_vib or d >= 0.45:
            self.state = MachineState.WARNING
        else:
            self.state = MachineState.RUNNING

        # Simulate production unit completion
        self._cycle_progress += dt
        if self._cycle_progress >= self.cycle_time:
            self._cycle_progress -= self.cycle_time
            self.production_count += 1

            # Base defect rate is ~1.0%. Degraded condition raises defect probability up to 30%.
            defect_prob = 0.01 + (d * 0.25)
            if self.state == MachineState.WARNING:
                defect_prob = max(defect_prob, 0.12)

            if random.random() < defect_prob:
                self.defective_products += 1
            else:
                self.good_products += 1

    def _simulate_failure(self, dt: float):
        """Simulate failed condition: halted production, accumulating downtime."""
        self.downtime_seconds += dt
        # Motor RPM drops to zero
        self.rpm = max(0.0, self.rpm - 150.0 * dt)
        # Power drops to standby/fault draw
        self.power = max(0.4, self.power - 0.5 * dt)
        self.current = max(1.2, self.current - 1.0 * dt)
        # Vibration settles or fluctuates violently depending on failure
        self.vibration = max(0.2, self.vibration * 0.92 + random.gauss(0, 0.05))
        # Temperature gradually dissipates toward ambient (~25°C)
        self.temperature = max(25.0, self.temperature - 0.15 * dt)

    def _simulate_maintenance(self, dt: float):
        """Simulate maintenance downtime: technicians servicing equipment."""
        self.downtime_seconds += dt
        self.rpm = 0.0
        self.power = 0.2
        self.current = 0.5
        self.vibration = 0.05 + abs(random.gauss(0, 0.02))
        self.temperature = max(25.0, self.temperature - 0.5 * dt)

    def _simulate_idle(self, dt: float):
        """Simulate idle state: machine powered on but awaiting input material."""
        self.rpm = self.nominal["rpm"] * 0.35 + random.gauss(0, 5.0)
        self.current = self.nominal["current"] * 0.30 + random.gauss(0, 0.05)
        self.power = self.nominal["power"] * 0.25 + random.gauss(0, 0.02)
        self.vibration = self.nominal["vibration"] * 0.40 + random.gauss(0, 0.03)
        self.temperature = max(35.0, self.temperature - 0.1 * dt)
