"""
SignalExtractor
---------------
Accumulates per-ROI green channel mean intensity across frames.
Produces a dict of 1-D time-series arrays (one per ROI).
"""
 
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
 
from config import CONFIG
from src.roi_extractor import ROIData
 
 
class SignalExtractor:
    def __init__(self):
        self._buffers: Dict[str, List[float]] = {
            "forehead": [], "left": [], "right": []
        }
        self._frame_count = 0
 
    def push_frame(self, rois: Dict[str, Optional[ROIData]]) -> None:
        self._frame_count += 1
        for key in self._buffers:
            roi = rois.get(key)
            if roi is not None and roi.pixel_count > 0:
                value = self._extract_value(roi)
            else:
                # Interpolation placeholder — filled in finalise()
                value = np.nan
            self._buffers[key].append(value)
 
    def finalise(self) -> Dict[str, np.ndarray]:
        """
        Returns per-ROI signals after NaN interpolation.
        Raises RuntimeError if signal is too short.
        """
        result: Dict[str, np.ndarray] = {}
        for key, buf in self._buffers.items():
            arr = np.array(buf, dtype=np.float64)
            arr = self._interpolate_nans(arr)
            n_valid = int(np.isfinite(arr).sum())
            logger.debug(
                f"ROI '{key}': {len(arr)} frames, {n_valid} valid, "
                f"mean={np.nanmean(arr):.2f}"
            )
            if n_valid < CONFIG.min_signal_length:
                raise RuntimeError(
                    f"ROI '{key}': only {n_valid} valid frames "
                    f"(need ≥ {CONFIG.min_signal_length})"
                )
            result[key] = arr
        return result
 
    # ── Private ────────────────────────────────────────────────────────────────
 
    def _extract_value(self, roi: ROIData) -> float:
        if CONFIG.channel == "green":
            return float(roi.pixels[:, 1].mean())   # BGR → index 1 = G
        # rgb_mean fallback
        return float(roi.pixels.mean())
 
    @staticmethod
    def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
        if not np.any(np.isnan(arr)):
            return arr
        nans = np.isnan(arr)
        not_nan = ~nans
        if not_nan.sum() < 2:
            return np.zeros_like(arr)
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[not_nan], arr[not_nan])
        return arr
 