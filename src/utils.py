import os
import sys
from pathlib import Path
from loguru import logger
import numpy as np


def setup_logger(level: str = "DEBUG", output_dir: str = "outputs") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(
        Path(output_dir) / "pipeline.log",
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


def validate_video_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise ValueError(f"Unsupported video format: {p.suffix}")
    return p


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std < 1e-10:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / std


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    p_sig = np.mean(signal ** 2)
    p_noi = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(p_sig / p_noi)