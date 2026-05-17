"""
DebugVisualiser
---------------
Saves diagnostic artefacts to outputs/:
  - signal plots (raw / detrended / filtered) per ROI
  - FFT/PSD overlay
  - cross-correlation plot
  - ROI overlay on sampled frames
"""
 
from __future__ import annotations
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
 
from config import CONFIG
from src.fft_analyzer import FrequencyResult
from src.sync_checker import SyncResult
from src.roi_extractor import ROIData
 
 
class DebugVisualiser:
    def __init__(self, output_dir: str = "outputs"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
 
    # ── Signal plot ────────────────────────────────────────────────────────────
 
    def plot_signals(
        self,
        raw: Dict[str, np.ndarray],
        detrended: Dict[str, np.ndarray],
        filtered: Dict[str, np.ndarray],
        fps: float,
    ) -> None:
        t = np.arange(len(next(iter(raw.values())))) / fps
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("rPPG Signals — Raw / Detrended / Filtered", fontsize=14)
 
        colors = {"forehead": "#e05c5c", "left": "#5cb8e0", "right": "#5ce07e"}
 
        for ax, (label, signals) in zip(axes, [
            ("Raw", raw), ("Detrended", detrended), ("Filtered", filtered)
        ]):
            for roi_name, sig in signals.items():
                ax.plot(t[:len(sig)], sig, label=roi_name,
                        color=colors.get(roi_name, "grey"), linewidth=0.9, alpha=0.85)
            ax.set_ylabel(label)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
 
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        path = self.out / "signal_plots.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.debug(f"Signal plot saved → {path}")
 
    # ── FFT / PSD plot ─────────────────────────────────────────────────────────
 
    def plot_fft(
        self,
        freq_results: Dict[str, FrequencyResult],
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Frequency Analysis — PSD + FFT Magnitude", fontsize=14)
 
        colors = {"forehead": "#e05c5c", "left": "#5cb8e0", "right": "#5ce07e"}
        ax_psd, ax_fft = axes
 
        for roi_name, fr in freq_results.items():
            c = colors.get(roi_name, "grey")
            mask_psd = fr.psd_freqs <= 4.0
            ax_psd.semilogy(fr.psd_freqs[mask_psd], fr.psd_power[mask_psd],
                            label=f"{roi_name} ({fr.dominant_bpm:.1f} BPM)", color=c)
            mask_fft = fr.fft_freqs <= 4.0
            ax_fft.plot(fr.fft_freqs[mask_fft], fr.fft_magnitude[mask_fft],
                        label=roi_name, color=c)
            ax_psd.axvline(fr.dominant_hz, color=c, linestyle="--", alpha=0.5)
            ax_fft.axvline(fr.dominant_hz, color=c, linestyle="--", alpha=0.5)
 
        for ax in axes:
            ax.axvspan(CONFIG.bandpass_low, CONFIG.bandpass_high,
                       alpha=0.08, color="yellow", label="passband")
            ax.set_xlabel("Frequency (Hz)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
 
        ax_psd.set_ylabel("PSD (log)")
        ax_fft.set_ylabel("FFT Magnitude")
        plt.tight_layout()
        path = self.out / "fft_psd.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.debug(f"FFT plot saved → {path}")
 
    # ── Cross-correlation plot ─────────────────────────────────────────────────
 
    def plot_sync(self, left: np.ndarray, right: np.ndarray, fps: float,
                  sync: SyncResult) -> None:
        xcorr = np.correlate(
            (left - left.mean()) / (left.std() + 1e-10),
            (right - right.mean()) / (right.std() + 1e-10),
            mode="full"
        ) / len(left)
        lags = np.arange(-(len(left) - 1), len(left)) / fps
 
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lags, xcorr, color="#5cb8e0", linewidth=0.9)
        ax.axvline(sync.lag_seconds, color="red", linestyle="--",
                   label=f"Peak lag = {sync.lag_seconds*1000:.1f} ms")
        ax.axhline(CONFIG.sync_min_correlation, color="orange", linestyle=":",
                   label=f"Threshold ({CONFIG.sync_min_correlation})")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalised Cross-Correlation")
        ax.set_title(f"Cheek Sync — corr={sync.correlation:.3f} | sync={sync.synchronized}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.out / "sync_xcorr.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.debug(f"Sync plot saved → {path}")
 
    # ── ROI overlay ───────────────────────────────────────────────────────────
 
    def save_roi_preview(
        self,
        frame: np.ndarray,
        rois: Dict[str, Optional[ROIData]],
        frame_idx: int,
    ) -> None:
        vis = frame.copy()
        colours = {
            "forehead": (0, 0, 230),
            "left": (230, 120, 0),
            "right": (0, 200, 80),
        }
        for roi_name, roi in rois.items():
            if roi is None:
                continue
            x, y, w, h = roi.bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), colours.get(roi_name, (255,255,255)), 2)
            cv2.putText(vis, roi_name, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours.get(roi_name, (255,255,255)), 1)
        path = self.out / f"roi_preview_frame{frame_idx:04d}.jpg"
        cv2.imwrite(str(path), vis)
        logger.debug(f"ROI preview saved → {path}")
 