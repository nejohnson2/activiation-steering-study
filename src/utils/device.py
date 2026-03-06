"""Device detection and configuration."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def get_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate dtype for device."""
    if device.type == "cuda":
        return torch.bfloat16
    elif device.type == "mps":
        return torch.float16
    return torch.float32
