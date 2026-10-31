"""Virtual Factory Simulator engine coordinating multiple machines and production line stages."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from config import load_config
from simulator.machine import Machine, MachineState


class FactorySimulator:
    """Orchestrates industrial equipment simulation for the smart factory."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the factory simulator.

        Args:
            config: Optional config dict. If omitted, loaded from config/config.yaml.
        """
        self.config: Dict[str, Any] = config or load_config()
        self.factory_name: str = self.config.get("factory", {}).get(
            "name", "Smart Factory Virtual Line"
        )
        self.step_interval: float = float(
            self.config.get("factory", {}).get("simulation_interval_seconds", 1.0)
        )

        # Initialize simulated machines
        machine_configs: List[Dict[str, Any]] = self.config.get("factory", {}).get(
            "machines", []
        )
        self.machines: Dict[str, Machine] = {}
        for m_cfg in machine_configs:
            m_id = m_cfg["id"]
            self.machines[m_id] = Machine(m_cfg)

        # Line layout stations definition
        self.stations: List[str] = [
            "Raw Material Input",
            "Infeed Conveyor",
            "M01 - CNC Milling",
            "Intermediate Conveyor",
            "M02 - Stamping Press",
            "M03 - Robotic Welder",
            "Assembly Station",
            "Quality Inspection",
            "Finished Goods Output",
        ]

    def get_machine(self, machine_id: str) -> Machine:
        """Retrieve machine instance by identifier."""
        if machine_id not in self.machines:
            raise KeyError(f"Machine with ID '{machine_id}' does not exist.")
        return self.machines[machine_id]

    def inject_degradation(self, machine_id: str, rate: Optional[float] = None):
        """Induce progressive wear & degradation on a specific machine."""
        machine = self.get_machine(machine_id)
        machine.start_degradation(rate=rate)

    def inject_failure(self, machine_id: str):
        """Induce sudden breakdown failure on a specific machine."""
        machine = self.get_machine(machine_id)
        machine.trigger_failure()

    def inject_maintenance(self, machine_id: str):
        """Put a specific machine into scheduled maintenance."""
        machine = self.get_machine(machine_id)
        machine.set_state(MachineState.MAINTENANCE)

    def repair_machine(self, machine_id: str):
        """Perform service and restore machine to normal running state."""
        machine = self.get_machine(machine_id)
        machine.perform_maintenance(restore_state=MachineState.RUNNING)

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Execute one simulation cycle across all machines.

        Args:
            dt: Elapsed time in seconds. Defaults to simulation_interval_seconds.

        Returns:
            Dictionary containing individual machine telemetries and factory production summary.
        """
        step_dt = dt if dt is not None else self.step_interval
        now_iso = datetime.now(timezone.utc).isoformat()

        machine_telemetries: Dict[str, Dict[str, Any]] = {}
        total_prod = 0
        total_good = 0
        total_defects = 0
        running_count = 0
        warning_count = 0
        failed_count = 0
        maintenance_count = 0
        idle_count = 0

        for m_id, machine in self.machines.items():
            t = machine.step(step_dt)
            machine_telemetries[m_id] = t

            total_prod += t["production_count"]
            total_good += t["good_products"]
            total_defects += t["defect_count"]

            state = t["machine_state"]
            if state == MachineState.RUNNING.value:
                running_count += 1
            elif state == MachineState.WARNING.value:
                warning_count += 1
            elif state == MachineState.FAILURE.value:
                failed_count += 1
            elif state == MachineState.MAINTENANCE.value:
                maintenance_count += 1
            elif state == MachineState.IDLE.value:
                idle_count += 1

        overall_defect_rate = (
            round(total_defects / total_prod, 4) if total_prod > 0 else 0.0
        )

        # Factory health level
        if failed_count > 0:
            factory_status = "CRITICAL"
        elif warning_count > 0:
            factory_status = "WARNING"
        else:
            factory_status = "NORMAL"

        return {
            "timestamp": now_iso,
            "factory_name": self.factory_name,
            "factory_status": factory_status,
            "stations": self.stations,
            "machines": machine_telemetries,
            "production_summary": {
                "total_production": total_prod,
                "total_good": total_good,
                "total_defects": total_defects,
                "defect_rate": overall_defect_rate,
                "active_machines": running_count,
                "warning_machines": warning_count,
                "failed_machines": failed_count,
                "maintenance_machines": maintenance_count,
                "idle_machines": idle_count,
            },
        }
