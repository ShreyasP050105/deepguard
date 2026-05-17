"""
ROIExtractor
------------
Extracts three skin ROI masks from a frame given face landmarks:
  - forehead
  - left_cheek
  - right_cheek
 
Each ROI is returned as a masked BGR patch (background zeroed).
Includes YCrCb-based skin segmentation to exclude non-skin pixels.
"""
 
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from loguru import logger
 
from config import CONFIG
from src.face_detector import (
    LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS, FOREHEAD_CENTER
)
 
 
@dataclass
class ROIData:
    name: str
    pixels: np.ndarray      # (N, 3) BGR skin pixels
    mask: np.ndarray        # (H, W) uint8 binary mask
    bbox: tuple             # (x, y, w, h) in original frame
    pixel_count: int
 
 
class ROIExtractor:
    # YCrCb skin thresholds (empirically validated, illumination-robust)
    _SKIN_CR_LOW, _SKIN_CR_HIGH = 133, 173
    _SKIN_CB_LOW, _SKIN_CB_HIGH = 77, 127
 
    def extract(
        self,
        bgr_frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> Dict[str, Optional[ROIData]]:
        """Returns {'forehead': ROIData|None, 'left': ROIData|None, 'right': ROIData|None}."""
        skin_mask = self._skin_mask(bgr_frame)
 
        return {
            "forehead": self._roi_from_landmark_cluster(
                bgr_frame, landmarks, skin_mask,
                self._forehead_bbox(landmarks),
                "forehead",
            ),
            "left": self._roi_from_landmark_cluster(
                bgr_frame, landmarks, skin_mask,
                self._cheek_bbox(landmarks, LEFT_CHEEK_LANDMARKS),
                "left_cheek",
            ),
            "right": self._roi_from_landmark_cluster(
                bgr_frame, landmarks, skin_mask,
                self._cheek_bbox(landmarks, RIGHT_CHEEK_LANDMARKS),
                "right_cheek",
            ),
        }
 
    # ── Private helpers ────────────────────────────────────────────────────────
 
    def _skin_mask(self, bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, self._SKIN_CR_LOW, self._SKIN_CB_LOW], dtype=np.uint8)
        upper = np.array([255, self._SKIN_CR_HIGH, self._SKIN_CB_HIGH], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask
 
    def _forehead_bbox(self, lm: np.ndarray) -> tuple:
        """Bounding box above eye line using landmark 10 (top of skull)."""
        h_pts = lm[[10, 338, 297, 332, 284, 251, 109, 103, 67, 54, 21, 162]]
        x_min = int(h_pts[:, 0].min())
        x_max = int(h_pts[:, 0].max())
        y_top = int(lm[10, 1])
        y_bot = int(lm[168, 1])  # landmark 168 ≈ mid-glabella
        w = x_max - x_min
        h = max(y_bot - y_top, 10)
        # shrink horizontally 15% each side to avoid hair/shadow
        margin = int(w * 0.15)
        return (x_min + margin, y_top, w - 2 * margin, h)
 
    def _cheek_bbox(self, lm: np.ndarray, indices: list) -> tuple:
        pts = lm[indices]
        x_min = int(pts[:, 0].min())
        x_max = int(pts[:, 0].max())
        y_min = int(pts[:, 1].min())
        y_max = int(pts[:, 1].max())
        return (x_min, y_min, x_max - x_min, y_max - y_min)
 
    def _roi_from_landmark_cluster(
        self,
        bgr: np.ndarray,
        lm: np.ndarray,
        skin_mask: np.ndarray,
        bbox: tuple,
        name: str,
    ) -> Optional[ROIData]:
        x, y, w, h = bbox
        H, W = bgr.shape[:2]
 
        # Clamp to frame bounds
        x = max(0, x); y = max(0, y)
        w = min(w, W - x); h = min(h, H - y)
        if w <= 0 or h <= 0:
            logger.debug(f"ROI '{name}' outside frame bounds")
            return None
 
        roi_mask = skin_mask[y:y+h, x:x+w]
        roi_bgr  = bgr[y:y+h, x:x+w]
        pixel_count = int(roi_mask.sum() // 255)
 
        if pixel_count < CONFIG.roi_min_pixels:
            logger.warning(f"ROI '{name}' has only {pixel_count} skin pixels — rejecting")
            return None
 
        # Apply mask
        roi_masked = cv2.bitwise_and(roi_bgr, roi_bgr, mask=roi_mask)
        skin_pixels = roi_bgr[roi_mask > 0]   # (N, 3)
 
        return ROIData(
            name=name,
            pixels=skin_pixels,
            mask=roi_mask,
            bbox=(x, y, w, h),
            pixel_count=pixel_count,
        )
 