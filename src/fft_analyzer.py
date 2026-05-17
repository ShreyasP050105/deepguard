"""
FFTAnalyzer
-----------
Computes Welch PSD + windowed FFT, extracts dominant heartbeat frequency,
converts to BPM, and computes a confidence score.
"""
 
from __future__ import annotations
import numpy as np
from scipy.signal import welch, windows
from dataclasses import dataclass
from typing import Tuple
from loguru import logger
 
from config import CONFIG
 
 
@dataclass
class FrequencyResult:
    dominant_hz: float
    dominant_bpm: float
    confidence: float           # ratio: peak PSD / mean PSD in passband
    psd_freqs: np.ndarray
    psd_power: np.ndarray
    fft_freqs: np.ndarray
    fft_magnitude: np.ndarray
    plausible: bool
 
 
class FFTAnalyzer:
    def __init__(self, fps: float):
        self.fps = fps
        # Welch segment size: ~4 s worth of frames (must be ≤ signal length)
        self._nperseg_base = max(32, int(fps * 4))
        logger.debug(f"FFTAnalyzer: fps={fps:.1f}, nperseg_base={self._nperseg_base}")
 
    def analyse(self, signal: np.ndarray) -> FrequencyResult:
        N = len(signal)
        nperseg = min(self._nperseg_base, N)
 
        # ── Welch PSD ──────────────────────────────────────────────────────────
        freqs_psd, psd = welch(signal, fs=self.fps, nperseg=nperseg,
                               noverlap=nperseg // 2, window="hann")
 
        # ── Windowed FFT (zero-padded) ─────────────────────────────────────────
        nfft = CONFIG.fft_interpolation * N
        win = windows.hann(N)
        fft_vals = np.fft.rfft(signal * win, n=nfft)
        fft_freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fps)
        fft_mag = np.abs(fft_vals)
 
        # ── Restrict to physiological band ────────────────────────────────────
        band_mask_psd = (freqs_psd >= CONFIG.bandpass_low) & (freqs_psd <= CONFIG.bandpass_high)
        band_mask_fft = (fft_freqs >= CONFIG.bandpass_low) & (fft_freqs <= CONFIG.bandpass_high)
 
        if band_mask_psd.sum() == 0 or band_mask_fft.sum() == 0:
            logger.error("No frequency bins in physiological band — signal too short")
            return self._null_result(freqs_psd, psd, fft_freqs, fft_mag)
 
        band_psd   = psd[band_mask_psd]
        band_freqs = freqs_psd[band_mask_psd]
        peak_idx   = np.argmax(band_psd)
        dominant_hz = float(band_freqs[peak_idx])
        dominant_bpm = dominant_hz * 60.0
 
        # Confidence: peak / mean of band PSD (>2× → recognisable peak)
        confidence = float(band_psd[peak_idx] / (band_psd.mean() + 1e-10))
        confidence = np.clip((confidence - 1.0) / 9.0, 0.0, 1.0)  # normalise to [0,1]
 
        plausible = CONFIG.bpm_low <= dominant_bpm <= CONFIG.bpm_high
 
        logger.info(
            f"FFT: {dominant_bpm:.1f} BPM  ({dominant_hz:.3f} Hz) "
            f"| confidence={confidence:.3f} | plausible={plausible}"
        )
 
        return FrequencyResult(
            dominant_hz=dominant_hz,
            dominant_bpm=dominant_bpm,
            confidence=confidence,
            psd_freqs=freqs_psd,
            psd_power=psd,
            fft_freqs=fft_freqs,
            fft_magnitude=fft_mag,
            plausible=plausible,
        )
 
    # ── Helpers ────────────────────────────────────────────────────────────────
 
    def _null_result(self, fp, psd, ff, fm) -> FrequencyResult:
        return FrequencyResult(0.0, 0.0, 0.0, fp, psd, ff, fm, False)
 