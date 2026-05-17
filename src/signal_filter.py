"""
SignalFilter
------------
1. Remove slow drift via moving-average detrending
2. Butterworth bandpass (0.7 – 3.0 Hz) zero-phase via filtfilt
3. Returns filtered signal + residual noise for SNR computation
"""
 
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple
from loguru import logger
 
from config import CONFIG
 
 
class SignalFilter:
    def __init__(self, fps: float):
        self.fps = fps
        nyq = fps / 2.0
        low = CONFIG.bandpass_low / nyq
        high = CONFIG.bandpass_high / nyq
 
        # Guard Nyquist
        low = np.clip(low, 1e-4, 0.999)
        high = np.clip(high, low + 1e-4, 0.999)
 
        self._b, self._a = butter(
            CONFIG.filter_order, [low, high], btype="band"
        )
        logger.debug(
            f"Butterworth BP filter: {CONFIG.bandpass_low}–{CONFIG.bandpass_high} Hz "
            f"@ {fps:.1f} FPS"
        )
 
    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            detrended  : signal after moving-average removal
            filtered   : bandpass-filtered signal
            noise      : residual (detrended – filtered)
        """
        detrended = self._detrend(signal)
        # Minimum padlen for filtfilt = 3 * max(len(a), len(b))
        min_len = 3 * (max(len(self._a), len(self._b)) + 1)
        if len(detrended) < min_len:
            logger.warning(
                f"Signal too short ({len(detrended)}) for filtfilt "
                f"(need ≥ {min_len}). Returning detrended only."
            )
            return detrended, detrended.copy(), np.zeros_like(detrended)
 
        filtered = filtfilt(self._b, self._a, detrended)
        noise = detrended - filtered
        return detrended, filtered, noise
 
    # ── Private ────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _detrend(signal: np.ndarray, window: int = 0) -> np.ndarray:
        """Subtract a moving average; auto-window = 1/3 of signal."""
        if window == 0:
            window = max(3, len(signal) // 3)
            window |= 1   # ensure odd
        kernel = np.ones(window) / window
        trend = np.convolve(signal, kernel, mode="same")
        return signal - trend
 