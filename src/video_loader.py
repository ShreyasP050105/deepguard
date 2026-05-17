"""
VideoLoader
-----------
Loads a video file, resamples to target FPS if needed, and yields BGR frames.
"""
 
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from loguru import logger
 
from config import CONFIG
from src.utils import validate_video_path
 
 
class VideoLoader:
    def __init__(self, path: str):
        self.path = validate_video_path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open: {self.path}")
 
        self.source_fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        # Frame-skip ratio to hit target FPS (≤1 means take every frame)
        self._skip_ratio: float = self.source_fps / CONFIG.target_fps
        self.effective_fps: float = min(self.source_fps, CONFIG.target_fps)
 
        logger.info(
            f"Video: {self.path.name} | {self.width}x{self.height} | "
            f"source={self.source_fps:.1f} FPS | effective={self.effective_fps:.1f} FPS"
        )
 
    # ── Public API ─────────────────────────────────────────────────────────────
 
    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, bgr_frame) resampled to target FPS."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        output_idx = 0
        source_idx = 0
        next_wanted = 0.0
 
        while output_idx < CONFIG.max_frames:
            ret, frame = self._cap.read()
            if not ret:
                break
 
            if source_idx >= next_wanted:
                yield output_idx, frame
                output_idx += 1
                next_wanted += self._skip_ratio
 
            source_idx += 1
 
        logger.debug(f"Loaded {output_idx} frames from {self.path.name}")
 
    def release(self) -> None:
        self._cap.release()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *_):
        self.release()
 