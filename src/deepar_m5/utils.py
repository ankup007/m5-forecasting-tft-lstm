from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch


logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure process-wide CLI logging with a compact module-aware format."""

    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level}")
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s",
        force=True,
    )


def batch_to_torch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a NumPy batch produced by ``WindowSampler`` onto a PyTorch device."""

    torch_batch = {
        "target": torch.as_tensor(batch["target"], dtype=torch.float32, device=device),
        "covariates": torch.as_tensor(batch["covariates"], dtype=torch.float32, device=device),
        "static_cats": torch.as_tensor(batch["static_cats"], dtype=torch.long, device=device),
        "scale": torch.as_tensor(batch["scale"], dtype=torch.float32, device=device),
        "loss_mask": torch.as_tensor(batch.get("loss_mask", batch["target"]), dtype=torch.float32, device=device),
    }
    if "prior_target" in batch:
        torch_batch["prior_target"] = torch.as_tensor(batch["prior_target"], dtype=torch.float32, device=device)
    
    return torch_batch


def choose_device(device_arg: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise return the requested device."""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)
