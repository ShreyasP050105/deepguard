"""
SyncChecker
-----------
Validates that blood-volume-pulse signals from left cheek and right cheek
are physiologically synchronized:
  1. Normalised cross-correlation peak ≥ threshold
  2. Phase lag ≤ max_phase_shift_s (biological limit)
 
Rationale: Real faces produce synchronized rPPG across bilateral regions.
Deepfake textures either show no correlation or spurious phase shifts.
"""
 
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from loguru import logger
 
from config import CONFIG
 
 
@dataclass
class SyncResult:
    correlation: float          # peak normalised cross-correlation [-1, 1]
    lag_frames: int             # lag at peak correlation
    lag_seconds: float
    phase_ok: bool              # |lag| within physiological bound
    corr_ok: bool               # correlation ≥ threshold
    synchronized: bool          # overall pass
 
 
class SyncChecker:
    def __init__(self, fps: float):
        self.fps = fps
        self._max_lag_frames = int(CONFIG.sync_max_phase_shift_s * fps)
        logger.debug(
            f"SyncChecker: max_lag={self._max_lag_frames} frames "
            f"({CONFIG.sync_max_phase_shift_s * 1000:.0f} ms)"
        )
 
    def check(self, left: np.ndarray, right: np.ndarray) -> SyncResult:
        """Cross-correlate left and right cheek filtered signals."""
        if len(left) != len(right):
            n = min(len(left), len(right))
            left, right = left[:n], right[:n]
 
        left_n  = self._normalise(left)
        right_n = self._normalise(right)
 
        # Full cross-correlation (O(N log N) via FFT)
        xcorr = np.correlate(left_n, right_n, mode="full")
        xcorr /= (len(left_n) + 1e-10)   # normalise to [-1, 1] approx
 
        lags = np.arange(-(len(left_n) - 1), len(left_n))
        peak_idx = np.argmax(xcorr)
        peak_corr = float(xcorr[peak_idx])
        lag = int(lags[peak_idx])
        lag_s = lag / self.fps
 
        phase_ok = abs(lag) <= self._max_lag_frames
        corr_ok  = peak_corr >= CONFIG.sync_min_correlation
        sync     = phase_ok and corr_ok
 
        logger.info(
            f"Sync: corr={peak_corr:.3f} | lag={lag} frames ({lag_s*1000:.1f} ms) "
            f"| phase_ok={phase_ok} | corr_ok={corr_ok} → sync={sync}"
        )
 
        return SyncResult(
            correlation=peak_corr,
            lag_frames=lag,
            lag_seconds=lag_s,
            phase_ok=phase_ok,
            corr_ok=corr_ok,
            synchronized=sync,
        )
 
    @staticmethod
    def _normalise(sig: np.ndarray) -> np.ndarray:
        std = sig.std()
        if std < 1e-10:
            return np.zeros_like(sig)
        return (sig - sig.mean()) / std
 