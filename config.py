from dataclasses import dataclass, field
from typing import Tuple
 
 
@dataclass
class PipelineConfig:
    # ── Video ──────────────────────────────────────────────────────────────────
    target_fps: int = 30                # resample target; real FPS read from file
    max_frames: int = 900               # cap at 30 s @ 30 FPS
 
    # ── Face detection ─────────────────────────────────────────────────────────
    face_detection_confidence: float = 0.6
    face_detection_retries: int = 3
    landmark_model: str = "mediapipe"   # "mediapipe" | "dlib"
 
    # ── ROI ───────────────────────────────────────────────────────────────────
    forehead_top_ratio: float = 0.10    # % of face height above eyebrows
    forehead_bot_ratio: float = 0.25
    cheek_x_inset: float = 0.05        # inset from face bbox edge
    cheek_y_top: float = 0.35
    cheek_y_bot: float = 0.65
    roi_min_pixels: int = 100           # reject ROI if fewer skin pixels
 
    # ── Signal extraction ──────────────────────────────────────────────────────
    channel: str = "green"              # "green" | "rgb_mean"
 
    # ── Signal filtering ──────────────────────────────────────────────────────
    bandpass_low: float = 0.7           # Hz  (~42 BPM)
    bandpass_high: float = 3.0          # Hz  (~180 BPM)
    filter_order: int = 4
 
    # ── FFT / PSD ─────────────────────────────────────────────────────────────
    min_signal_length: int = 64         # frames required for FFT
    fft_interpolation: int = 4          # zero-pad multiplier
 
    # ── Physiological plausibility ─────────────────────────────────────────────
    bpm_low: float = 42.0
    bpm_high: float = 180.0
 
    # ── Synchronization ───────────────────────────────────────────────────────
    sync_min_correlation: float = 0.50  # cross-correlation threshold
    sync_max_phase_shift_s: float = 0.10  # seconds
 
    # ── Decision engine ───────────────────────────────────────────────────────
    min_confidence: float = 0.50        # FFT confidence to pass
    decision_threshold: float = 0.60   # composite score → REAL
 
    # ── Debug ─────────────────────────────────────────────────────────────────
    debug: bool = True
    output_dir: str = "outputs"
    save_plots: bool = True
    save_roi_preview: bool = True
    log_level: str = "DEBUG"
 
 
CONFIG = PipelineConfig()