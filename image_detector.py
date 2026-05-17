"""
ImageDeepfakeDetector
---------------------
Detects deepfakes in still images using:
1. DCT frequency domain artifact analysis
2. Camera noise pattern consistency (SPN)
3. Face landmark geometric asymmetry
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from config import CONFIG
from src.face_detector import FaceDetector
from src.utils import setup_logger


@dataclass
class ImageResult:
    verdict: str
    composite_score: float
    dct_score: float
    noise_score: float
    symmetry_score: float
    details: dict


class ImageDeepfakeDetector:
    def __init__(self):
        self._face_det = FaceDetector()

    def run(self, image_path: str) -> ImageResult:
        logger.info(f"━━━ Image Analysis START: {image_path} ━━━")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        # ── Stage 1: Face detection ───────────────────────────────────────────
        landmarks = self._face_det.detect(bgr)
        if landmarks is None:
            raise RuntimeError("No face detected in image")

        # ── Stage 2: Crop face region ─────────────────────────────────────────
        face_bgr = self._crop_face(bgr, landmarks)

        # ── Stage 3: Three forensic signals ───────────────────────────────────
        dct_score      = self._dct_artifact_score(face_bgr)
        noise_score    = self._noise_consistency_score(face_bgr)
        symmetry_score = self._landmark_symmetry_score(landmarks)

        # ── Stage 4: Weighted composite ───────────────────────────────────────
        composite = (
            0.40 * dct_score +
            0.35 * noise_score +
            0.25 * symmetry_score
        )
        if composite >= 0.50:
            verdict = "REAL"
        elif composite >= 0.35:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAKE"

        logger.info(
            f"━━━ VERDICT: {verdict} "
            f"(score={composite:.3f}, dct={dct_score:.3f}, "
            f"noise={noise_score:.3f}, sym={symmetry_score:.3f}) ━━━"
        )

        return ImageResult(
            verdict=verdict,
            composite_score=float(composite),
            dct_score=float(dct_score),
            noise_score=float(noise_score),
            symmetry_score=float(symmetry_score),
            details={
                "dct_score": dct_score,
                "noise_score": noise_score,
                "symmetry_score": symmetry_score,
            }
        )

    # ── Forensic signal 1: DCT artifact analysis ──────────────────────────────

    def _dct_artifact_score(self, face_bgr: np.ndarray) -> float:
        """
        GAN-generated faces leave characteristic high-frequency artifacts
        in the DCT domain. Real photos have natural frequency falloff.
        Higher score = more natural = more likely REAL.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128)).astype(np.float32)

        # Block DCT (8x8 blocks like JPEG)
        h, w = gray.shape
        block_size = 8
        high_freq_ratios = []

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block)
                total_energy = np.sum(dct_block ** 2) + 1e-10
                # High frequency = bottom-right quadrant of DCT block
                hf_energy = np.sum(dct_block[4:, 4:] ** 2)
                high_freq_ratios.append(hf_energy / total_energy)

        mean_hf = float(np.mean(high_freq_ratios))

        # Real images: moderate high-freq content (0.05–0.25)
        # Deepfakes: either too smooth (low HF) or too noisy (high HF)
        natural_low  = 0.05
        natural_high = 0.25
        if natural_low <= mean_hf <= natural_high:
            score = 1.0 - abs(mean_hf - 0.15) / 0.15
        else:
            score = max(0.0, 1.0 - abs(mean_hf - 0.15) / 0.30)

        logger.debug(f"DCT: mean_hf_ratio={mean_hf:.4f} → score={score:.3f}")
        return float(np.clip(score, 0.0, 1.0))

    # ── Forensic signal 2: Noise pattern consistency ───────────────────────────

    def _noise_consistency_score(self, face_bgr: np.ndarray) -> float:
        """
        Real camera sensors produce spatially consistent noise patterns.
        GAN-generated images have inconsistent or periodic noise artifacts.
        Higher score = more consistent = more likely REAL.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (128, 128))

        # Extract noise residual via Gaussian blur subtraction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        # Divide into quadrants and compare noise statistics
        h, w = noise.shape
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            noise[:mid_h, :mid_w],
            noise[:mid_h, mid_w:],
            noise[mid_h:, :mid_w],
            noise[mid_h:, mid_w:],
        ]

        stds = [q.std() for q in quadrants]
        mean_std = np.mean(stds)
        std_of_stds = np.std(stds)

        # Consistency: low variance across quadrant noise levels = real
        consistency = 1.0 - np.clip(std_of_stds / (mean_std + 1e-10), 0.0, 1.0)

        # Also check for periodic noise (GAN artifact) via FFT of noise
        fft_noise = np.abs(np.fft.fft2(noise))
        fft_noise_shifted = np.fft.fftshift(fft_noise)
        center = fft_noise_shifted[mid_h-8:mid_h+8, mid_w-8:mid_w+8]
        periphery_mean = (fft_noise_shifted.sum() - center.sum()) / (h * w - 256 + 1)
        center_mean = center.mean()
        # Real images: energy concentrated at center (low freq)
        energy_ratio = float(np.clip(center_mean / (periphery_mean + 1e-10) / 10.0, 0.0, 1.0))

        score = 0.5 * consistency + 0.5 * energy_ratio
        logger.debug(f"Noise: consistency={consistency:.3f}, energy_ratio={energy_ratio:.3f} → score={score:.3f}")
        return float(np.clip(score, 0.0, 1.0))

    # ── Forensic signal 3: Landmark geometric symmetry ─────────────────────────

    def _landmark_symmetry_score(self, landmarks: np.ndarray) -> float:
        """
        Human faces have natural bilateral symmetry with small natural asymmetry.
        Deepfake faces often show unnatural symmetry (too perfect) or asymmetry
        (warping artifacts near face edges).
        Higher score = natural asymmetry range = more likely REAL.
        """
        # Mirror landmark pairs (left index, right index)
        pairs = [
            (33, 263),   # outer eye corners
            (133, 362),  # inner eye corners
            (61, 291),   # mouth corners
            (117, 346),  # cheek points
            (234, 454),  # jaw points
            (70, 300),   # eyebrow outer
            (107, 336),  # eyebrow inner
        ]

        nose_x = float(landmarks[1, 0])  # nose tip as vertical axis
        asymmetries = []

        for l_idx, r_idx in pairs:
            if l_idx >= len(landmarks) or r_idx >= len(landmarks):
                continue
            l_pt = landmarks[l_idx]
            r_pt = landmarks[r_idx]
            # Distance from nose axis
            l_dist = abs(l_pt[0] - nose_x)
            r_dist = abs(r_pt[0] - nose_x)
            if l_dist + r_dist < 1e-6:
                continue
            asym = abs(l_dist - r_dist) / (l_dist + r_dist)
            asymmetries.append(asym)

        if not asymmetries:
            return 0.5

        mean_asym = float(np.mean(asymmetries))

        # Adjusted for real-world compressed images (WhatsApp, social media)
        # Natural human asymmetry: 0.02–0.35
        # Too symmetric (<0.02): suspicious
        # Too asymmetric (>0.40): warping artifact
        if 0.02 <= mean_asym <= 0.35:
            score = 1.0
        elif mean_asym < 0.02:
            score = mean_asym / 0.02
        else:
            score = max(0.0, 1.0 - (mean_asym - 0.35) / 0.15)

        logger.debug(f"Symmetry: mean_asym={mean_asym:.4f} → score={score:.3f}")
        return float(np.clip(score, 0.0, 1.0))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _crop_face(self, bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x_min = int(np.clip(landmarks[:, 0].min() * 0.95, 0, w))
        x_max = int(np.clip(landmarks[:, 0].max() * 1.05, 0, w))
        y_min = int(np.clip(landmarks[:, 1].min() * 0.95, 0, h))
        y_max = int(np.clip(landmarks[:, 1].max() * 1.05, 0, h))
        return bgr[y_min:y_max, x_min:x_max]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image Deepfake Detector")
    parser.add_argument("image", help="Path to input image file")
    args = parser.parse_args()

    setup_logger(CONFIG.log_level, CONFIG.output_dir)

    try:
        detector = ImageDeepfakeDetector()
        result = detector.run(args.image)

        print("\n" + "═" * 50)
        print(f"  VERDICT          : {result.verdict}")
        print(f"  Composite Score  : {result.composite_score:.4f}")
        print(f"  DCT Score        : {result.dct_score:.4f}")
        print(f"  Noise Score      : {result.noise_score:.4f}")
        print(f"  Symmetry Score   : {result.symmetry_score:.4f}")
        print("═" * 50)

    except Exception as exc:
        logger.error(f"Detection failed: {exc}")
        print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()