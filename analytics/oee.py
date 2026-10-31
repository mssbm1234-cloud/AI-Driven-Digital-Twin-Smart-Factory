"""Production Analytics and Overall Equipment Effectiveness (OEE) Engine."""

from typing import Any, Dict, Optional
import numpy as np


class OEEEngine:
    """Calculates Availability, Performance, Quality, and OEE for machines and production line."""

    @staticmethod
    def calculate_machine_oee(
        telemetry: Dict[str, Any],
        nominal_cycle_time: float = 12.0,
        planned_operating_seconds: float = 3600.0,
    ) -> Dict[str, float]:
        """Compute OEE metrics for an individual machine.

        OEE = Availability x Performance x Quality
        """
        downtime = float(telemetry.get("downtime_seconds", 0.0))
        total_prod = int(telemetry.get("production_count", 0))
        good_prod = int(telemetry.get("good_products", 0))
        defect_prod = int(telemetry.get("defect_count", 0))

        # 1. Availability = Operating Time / Planned Production Time
        operating_time = max(0.0, planned_operating_seconds - downtime)
        availability = operating_time / planned_operating_seconds if planned_operating_seconds > 0 else 1.0
        availability = float(np.clip(availability, 0.0, 1.0))

        # 2. Performance = (Ideal Cycle Time * Total Count) / Operating Time
        if operating_time > 0 and total_prod > 0:
            performance = (nominal_cycle_time * total_prod) / operating_time
            performance = float(np.clip(performance, 0.0, 1.0))
        else:
            performance = 1.0 if total_prod == 0 and downtime == 0 else 0.0

        # 3. Quality = Good Units / Total Units
        if total_prod > 0:
            quality = good_prod / total_prod
            quality = float(np.clip(quality, 0.0, 1.0))
        else:
            quality = 1.0

        # Composite OEE
        oee = availability * performance * quality

        return {
            "availability": round(availability * 100.0, 2),
            "performance": round(performance * 100.0, 2),
            "quality": round(quality * 100.0, 2),
            "oee": round(oee * 100.0, 2),
        }
