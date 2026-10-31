"""Data preprocessing and feature scaling for industrial telemetry."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from config import load_config

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "current",
    "rpm",
    "pressure",
    "power",
]


class TelemetryPreprocessor:
    """Preprocesses and normalizes raw machine telemetry for PyTorch model input."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with nominal machine baselines."""
        self.config = config or load_config()
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self._init_nominal_stats()

    def _init_nominal_stats(self):
        """Compute standardizing statistics based on factory nominal baselines and expected variances."""
        machines = self.config.get("factory", {}).get("machines", [])
        if not machines:
            # Fallback typical industrial defaults
            self.means = {
                "temperature": 70.0,
                "vibration": 2.5,
                "current": 12.0,
                "rpm": 1300.0,
                "pressure": 6.5,
                "power": 4.5,
            }
            self.stds = {
                "temperature": 15.0,
                "vibration": 1.5,
                "current": 4.0,
                "rpm": 400.0,
                "pressure": 2.0,
                "power": 1.5,
            }
            return

        # Calculate population statistics across all configured machines
        for feat in self.feature_names:
            vals = [m.get("nominal", {}).get(feat, 1.0) for m in machines]
            self.means[feat] = float(np.mean(vals))
            # Standard deviation with safety epsilon to avoid division by zero
            std_val = float(np.std(vals))
            # Provide sensible spread for normalization
            spread = {
                "temperature": 12.0,
                "vibration": 1.5,
                "current": 3.5,
                "rpm": 350.0,
                "pressure": 1.8,
                "power": 1.2,
            }
            self.stds[feat] = max(std_val, spread.get(feat, 1.0))

    def extract_features(self, telemetry: Dict[str, Any]) -> np.ndarray:
        """Extract ordered feature vector from a telemetry dictionary.

        Args:
            telemetry: Machine telemetry dictionary.

        Returns:
            1D numpy array of float32 values [temperature, vibration, current, rpm, pressure, power].
        """
        vals = []
        for feat in self.feature_names:
            if feat not in telemetry:
                raise KeyError(f"Telemetry missing required AI feature: '{feat}'")
            vals.append(float(telemetry[feat]))
        return np.array(vals, dtype=np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector using standardized z-score scaling.

        Args:
            features: 1D or 2D array of raw telemetry values.

        Returns:
            Normalized array with zero-mean unit-variance scaling.
        """
        mean_vec = np.array([self.means[f] for f in self.feature_names], dtype=np.float32)
        std_vec = np.array([self.stds[f] for f in self.feature_names], dtype=np.float32)

        return (features - mean_vec) / std_vec

    def inverse_transform(self, normalized_features: np.ndarray) -> np.ndarray:
        """Denormalize features back to physical engineering units."""
        mean_vec = np.array([self.means[f] for f in self.feature_names], dtype=np.float32)
        std_vec = np.array([self.stds[f] for f in self.feature_names], dtype=np.float32)

        return (normalized_features * std_vec) + mean_vec

    def save(self, filepath: Union[str, Path]):
        """Persist scaler parameters to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TelemetryPreprocessor":
        """Load preprocessor from persisted JSON file."""
        p = Path(filepath)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        preprocessor = cls()
        preprocessor.feature_names = data["feature_names"]
        preprocessor.means = data["means"]
        preprocessor.stds = data["stds"]
        return preprocessor
