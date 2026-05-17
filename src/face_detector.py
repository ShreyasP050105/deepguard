from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
from loguru import logger
import urllib.request
import os

from config import CONFIG

LEFT_CHEEK_LANDMARKS = [116, 117, 118, 119, 120, 121, 126, 142,
                         36, 205, 206, 207, 213, 192, 214, 215]
RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 348, 349, 350, 355, 371,
                          266, 425, 426, 427, 436, 416, 434, 435]
FOREHEAD_TOP = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109]
FOREHEAD_CENTER = [10]
NOSE_TIP = 1

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading MediaPipe face landmarker model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model downloaded.")


class FaceDetector:
    def __init__(self):
        _download_model()

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=CONFIG.face_detection_confidence,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        logger.debug("FaceDetector initialised (MediaPipe Tasks API)")

    def detect(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        import mediapipe as mp

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        for attempt in range(CONFIG.face_detection_retries):
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = self._detector.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                coords = np.array([[p.x * w, p.y * h] for p in lm],
                                   dtype=np.float32)
                return coords

            rgb = cv2.convertScaleAbs(rgb, alpha=1.15, beta=10)
            logger.debug(f"Face detection retry {attempt + 1}")

        logger.warning("Face not detected in frame")
        return None

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()