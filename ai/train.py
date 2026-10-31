"""Model training pipeline for PyTorch Industrial Telemetry Autoencoder."""

import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from ai.preprocessing import TelemetryPreprocessor
from ai.anomaly_model import TelemetryAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AITrainer")

MODEL_DIR = PROJECT_ROOT / "ai" / "models"
MODEL_PATH = MODEL_DIR / "anomaly_detector.pt"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.json"


def generate_baseline_training_data(
    config: Dict, num_samples_per_machine: int = 600
) -> np.ndarray:
    """Synthesize normal baseline operating telemetry for training.

    Args:
        config: Factory configuration dict.
        num_samples_per_machine: Number of normal operating observations per machine.

    Returns:
        2D numpy array of shape (N, 6) containing normal sensor values.
    """
    machines = config.get("factory", {}).get("machines", [])
    data = []

    for m in machines:
        nom = m.get("nominal", {})
        noise = m.get("noise", {})
        for _ in range(num_samples_per_machine):
            sample = [
                nom.get("temperature", 65.0) + random.gauss(0, noise.get("temperature", 0.4)),
                nom.get("vibration", 2.5) + random.gauss(0, noise.get("vibration", 0.12)),
                nom.get("current", 8.5) + random.gauss(0, noise.get("current", 0.15)),
                nom.get("rpm", 1800.0) + random.gauss(0, noise.get("rpm", 12.0)),
                nom.get("pressure", 6.0) + random.gauss(0, noise.get("pressure", 0.08)),
                nom.get("power", 3.2) + random.gauss(0, noise.get("power", 0.06)),
            ]
            data.append(sample)

    return np.array(data, dtype=np.float32)


def train_anomaly_model(
    epochs: int = 35,
    batch_size: int = 32,
    learning_rate: float = 0.005,
    save_artifacts: bool = True,
) -> Tuple[TelemetryAutoencoder, TelemetryPreprocessor, float]:
    """Train the autoencoder on normal baseline data and calibrate anomaly threshold.

    Returns:
        Tuple of (trained_model, preprocessor, threshold).
    """
    config = load_config()
    preprocessor = TelemetryPreprocessor(config=config)

    # 1. Synthesize normal dataset
    logger.info("Generating normal baseline dataset for training...")
    raw_data = generate_baseline_training_data(config, num_samples_per_machine=600)
    logger.info("Generated %d baseline samples across %d features.", raw_data.shape[0], raw_data.shape[1])

    # 2. Normalize features
    normalized_data = preprocessor.transform(raw_data)

    # 3. Create PyTorch DataLoader
    tensor_x = torch.tensor(normalized_data, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Model, Loss, Optimizer
    model = TelemetryAutoencoder(input_dim=6, latent_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training loop
    model.train()
    logger.info("Starting Autoencoder training for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_in, batch_target in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_in)
            loss = criterion(reconstructed, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_in.size(0)

        epoch_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == epochs:
            logger.info("Epoch %02d/%02d - Reconstruction Loss (MSE): %.6f", epoch, epochs, epoch_loss)

    # 6. Calibrate Anomaly Threshold on normal training data
    model.eval()
    with torch.no_grad():
        recon_errors = model.compute_reconstruction_error(tensor_x).numpy()

    # Threshold: 99th percentile + small buffer
    threshold = float(np.percentile(recon_errors, 99.0) * 1.35)
    logger.info("Calibrated Anomaly Threshold: %.6f (Max normal error: %.6f)", threshold, float(np.max(recon_errors)))

    # 7. Save model and preprocessor artifacts
    if save_artifacts:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "threshold": threshold,
                "input_dim": 6,
                "latent_dim": 4,
            },
            MODEL_PATH,
        )
        preprocessor.save(PREPROCESSOR_PATH)
        logger.info("Saved trained model to %s", MODEL_PATH)
        logger.info("Saved preprocessor to %s", PREPROCESSOR_PATH)

    return model, preprocessor, threshold


if __name__ == "__main__":
    train_anomaly_model()
