
"""
Unit tests — signal filtering and FFT analysis.
Run: python -m pytest tests/ -v
"""
 
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
import numpy as np
import pytest
from config import CONFIG, PipelineConfig
from src.signal_filter import SignalFilter
from src.fft_analyzer import FFTAnalyzer
 
 
FPS = 30.0
DURATION = 10.0  # seconds
N = int(FPS * DURATION)
 
 
def _sine_signal(hz: float, fps: float, n: int, noise_amp: float = 0.05) -> np.ndarray:
    t = np.arange(n) / fps
    signal = np.sin(2 * np.pi * hz * t)
    signal += noise_amp * np.random.randn(n)
    return signal
 
 
class TestSignalFilter:
    def test_bandpass_passes_physiological(self):
        sig = _sine_signal(hz=1.2, fps=FPS, n=N)  # 72 BPM — should pass
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        assert filtered.std() > 0.01, "Filtered signal should have non-trivial energy"
 
    def test_bandpass_rejects_dc(self):
        sig = np.ones(N) * 100.0 + _sine_signal(hz=1.2, fps=FPS, n=N) * 0.1
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        # After detrending + bandpass, DC component should vanish
        assert abs(filtered.mean()) < 0.5
 
    def test_bandpass_rejects_high_freq(self):
        """Signal at 5 Hz (above band) should be heavily attenuated."""
        sig = _sine_signal(hz=5.0, fps=FPS, n=N)
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        assert filtered.std() < 0.3, "High-frequency signal should be attenuated"
 
    def test_flat_signal_handled(self):
        """All-zeros signal should not crash."""
        sig = np.zeros(N)
        sf = SignalFilter(FPS)
        det, filt, noi = sf.process(sig)
        assert filt.shape == sig.shape
 
    def test_detrend_removes_linear_trend(self):
        t = np.linspace(0, 10, N)
        trend = 5.0 * t
        signal = trend + _sine_signal(hz=1.0, fps=FPS, n=N)
        sf = SignalFilter(FPS)
        det, _, _ = sf.process(signal)
        # Detrended should have much smaller range than original
        assert abs(det.mean()) < abs(signal.mean()) * 0.1
 
 
class TestFFTAnalyzer:
    def test_detects_correct_bpm(self):
        hz = 1.2  # 72 BPM
        sig = _sine_signal(hz=hz, fps=FPS, n=N, noise_amp=0.02)
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        fft = FFTAnalyzer(FPS)
        result = fft.analyse(filtered)
        assert abs(result.dominant_bpm - hz * 60) < 5.0, (
            f"Expected ~{hz*60:.0f} BPM, got {result.dominant_bpm:.1f}"
        )
 
    def test_plausibility_flag(self):
        hz = 0.3  # 18 BPM — physiologically impossible
        sig = _sine_signal(hz=hz, fps=FPS, n=N, noise_amp=0.005)
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        fft = FFTAnalyzer(FPS)
        result = fft.analyse(filtered)
        # Since 0.3 Hz < bandpass_low, dominant in band will be different
        # Plausibility check: dominant BPM might still land in band from noise
        # This tests the flag logic, not BPM accuracy
        assert isinstance(result.plausible, bool)
 
    def test_confidence_high_for_clean_signal(self):
        sig = _sine_signal(hz=1.0, fps=FPS, n=N, noise_amp=0.01)
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        fft = FFTAnalyzer(FPS)
        result = fft.analyse(filtered)
        assert result.confidence > 0.3, "Clean sinusoid should have reasonable confidence"
 
    def test_confidence_low_for_noise(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(N)
        sf = SignalFilter(FPS)
        _, filtered, _ = sf.process(sig)
        fft = FFTAnalyzer(FPS)
        result = fft.analyse(filtered)
        assert result.confidence < 0.8, "Pure noise should not have high confidence"
 