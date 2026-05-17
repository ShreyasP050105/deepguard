"""
Biometric Liveness Detection — rPPG Pipeline
=============================================
Entry point: python main.py <video_path> [--debug]
 
Decision:  REAL  → valid heartbeat (BPM in range, high confidence) AND synchronized ROIs
           FAKE  → flat / noisy signal OR unsynchronized cheek signals
"""
 
from __future__ import annotations
import sys
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger
 
from config import CONFIG
from src.utils import setup_logger
from src.video_loader import VideoLoader
from src.face_detector import FaceDetector
from src.roi_extractor import ROIExtractor
from src.signal_extractor import SignalExtractor
from src.signal_filter import SignalFilter
from src.fft_analyzer import FFTAnalyzer, FrequencyResult
from src.sync_checker import SyncChecker
from src.debug_visualiser import DebugVisualiser
 
 
# ── Result dataclass ──────────────────────────────────────────────────────────
 
@dataclass
class LivenessResult:
    verdict: str                    # "REAL" | "FAKE" | "INCONCLUSIVE"
    composite_score: float          # [0, 1] — higher → more likely REAL
    dominant_bpm: float
    confidence: float
    synchronized: bool
    correlation: float
    details: dict
 
 
# ── Pipeline ──────────────────────────────────────────────────────────────────
 
class LivenessPipeline:
    def __init__(self):
        self._face_det   = FaceDetector()
        self._roi_ext    = ROIExtractor()
        self._sig_ext    = SignalExtractor()
        self._visualiser = DebugVisualiser(CONFIG.output_dir) if CONFIG.debug else None
 
    def run(self, video_path: str) -> LivenessResult:
        logger.info(f"━━━ Pipeline START: {video_path} ━━━")
 
        # ── Stage 1: Frame extraction + ROI accumulation ──────────────────────
        raw_signals = self._extract_signals(video_path)
 
        # ── Stage 2: Signal filtering ─────────────────────────────────────────
        fps = self._get_fps(video_path)
        sig_filter = SignalFilter(fps)
 
        detrended, filtered, noise = {}, {}, {}
        for key, sig in raw_signals.items():
            det, filt, noi = sig_filter.process(sig)
            detrended[key] = det
            filtered[key]  = filt
            noise[key]      = noi
            logger.debug(f"ROI '{key}' filtered — len={len(filt)}")
 
        if CONFIG.debug and self._visualiser:
            self._visualiser.plot_signals(raw_signals, detrended, filtered, fps)
 
        # ── Stage 3: FFT / PSD analysis (use forehead as primary) ─────────────
        fft_analyzer = FFTAnalyzer(fps)
        freq_results: Dict[str, FrequencyResult] = {}
        for key in filtered:
            freq_results[key] = fft_analyzer.analyse(filtered[key])
 
        if CONFIG.debug and self._visualiser:
            self._visualiser.plot_fft(freq_results)
 
        primary_fr = freq_results.get("forehead") or next(iter(freq_results.values()))
 
        # ── Stage 4: Multi-region synchronization ─────────────────────────────
        sync_checker = SyncChecker(fps)
        sync_result = sync_checker.check(filtered["left"], filtered["right"])
 
        if CONFIG.debug and self._visualiser:
            self._visualiser.plot_sync(
                filtered["left"], filtered["right"], fps, sync_result
            )
 
        # ── Stage 5: Decision engine ───────────────────────────────────────────
        result = self._decide(primary_fr, sync_result, freq_results)
 
        logger.info(
            f"━━━ VERDICT: {result.verdict}  "
            f"(score={result.composite_score:.3f}, "
            f"BPM={result.dominant_bpm:.1f}, "
            f"sync={result.synchronized}) ━━━"
        )
        return result
 
    # ── Private helpers ────────────────────────────────────────────────────────
 
    def _get_fps(self, video_path: str) -> float:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(min(fps, CONFIG.target_fps))
 
    def _extract_signals(self, video_path: str) -> Dict[str, np.ndarray]:
        preview_saved = False
        with VideoLoader(video_path) as loader:
            fps = loader.effective_fps
            for frame_idx, frame in loader.frames():
                landmarks = self._face_det.detect(frame)
                if landmarks is None:
                    # Push NaN frames so signal length stays consistent
                    self._sig_ext.push_frame({"forehead": None, "left": None, "right": None})
                    continue
 
                rois = self._roi_ext.extract(frame, landmarks)
                self._sig_ext.push_frame(rois)
 
                # Save one ROI preview
                if CONFIG.debug and not preview_saved and self._visualiser:
                    self._visualiser.save_roi_preview(frame, rois, frame_idx)
                    preview_saved = True
 
        return self._sig_ext.finalise()
 
    def _decide(
        self,
        primary_fr: FrequencyResult,
        sync: SyncResult,
        all_fr: Dict[str, FrequencyResult],
    ) -> LivenessResult:
        # Score components (all in [0, 1])
        bpm_score = 1.0 if primary_fr.plausible else 0.0
        conf_score = primary_fr.confidence
        sync_score = float(sync.synchronized)
        corr_score = np.clip(sync.correlation, 0.0, 1.0)
 
        # Region agreement: BPM deviation across ROIs
        bpms = [fr.dominant_bpm for fr in all_fr.values() if fr.plausible]
        if len(bpms) >= 2:
            bpm_std = np.std(bpms)
            agreement_score = float(np.clip(1.0 - bpm_std / 15.0, 0.0, 1.0))
        else:
            agreement_score = 0.5
 
        # Weighted composite
        composite = (
            0.30 * bpm_score     +
            0.25 * conf_score    +
            0.25 * sync_score    +
            0.10 * corr_score    +
            0.10 * agreement_score
        )
 
        # Hard-fail conditions
        if not primary_fr.plausible:
            verdict = "FAKE"
        elif composite >= CONFIG.decision_threshold:
            verdict = "REAL"
        elif composite >= CONFIG.decision_threshold * 0.75:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAKE"
 
        return LivenessResult(
            verdict=verdict,
            composite_score=float(composite),
            dominant_bpm=primary_fr.dominant_bpm,
            confidence=primary_fr.confidence,
            synchronized=sync.synchronized,
            correlation=sync.correlation,
            details={
                "bpm_score":       bpm_score,
                "conf_score":      conf_score,
                "sync_score":      sync_score,
                "corr_score":      corr_score,
                "agreement_score": agreement_score,
                "all_bpms":        {k: v.dominant_bpm for k, v in all_fr.items()},
                "phase_lag_ms":    sync.lag_seconds * 1000,
            },
        )
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="rPPG Liveness Detection")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--debug", action="store_true",
                        help="Override config.debug=True")
    args = parser.parse_args()
 
    if args.debug:
        CONFIG.debug = True
 
    setup_logger(CONFIG.log_level, CONFIG.output_dir)
 
    try:
        pipeline = LivenessPipeline()
        result = pipeline.run(args.video)
 
        print("\n" + "═" * 50)
        print(f"  VERDICT        : {result.verdict}")
        print(f"  Composite Score: {result.composite_score:.4f}")
        print(f"  Heart Rate     : {result.dominant_bpm:.1f} BPM")
        print(f"  FFT Confidence : {result.confidence:.4f}")
        print(f"  Synchronized   : {result.synchronized}")
        print(f"  Correlation    : {result.correlation:.4f}")
        print(f"  Phase Lag      : {result.details['phase_lag_ms']:.1f} ms")
        print("═" * 50)
 
    except RuntimeError as exc:
        logger.error(f"Pipeline failure: {exc}")
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
 