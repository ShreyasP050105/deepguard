# DeepGuard — AI Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B4-orange?style=flat-square&logo=pytorch)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-red?style=flat-square)

A production-grade deepfake detection web app that combines **rPPG biometric liveness detection** for videos and **EfficientNet-B4 neural network** for images — achieving 99.89% validation accuracy.

---

## 🧠 How It Works

### For Videos — rPPG Liveness Detection
Real human faces have blood flow that causes subtle, periodic color changes in the skin. DeepGuard extracts this signal from three facial regions (forehead, left cheek, right cheek) and validates it biometrically.

Video Input
↓
MediaPipe FaceMesh (468 landmarks)
↓
ROI Extraction (forehead + bilateral cheeks)
↓
Green Channel Signal Extraction
↓
Butterworth Bandpass Filter (0.7–3.0 Hz)
↓
FFT + Welch PSD Analysis
↓
Cross-Region Synchronization Check
↓
REAL / FAKE / INCONCLUSIVE

### For Images — EfficientNet-B4
A fine-tuned EfficientNet-B4 model trained on 140,000 real and fake faces from the 140k Real and Fake Faces dataset.
Image Input
↓
Face Detection (MediaPipe)
↓
EfficientNet-B4 Inference
↓
REAL / FAKE

---

## 🚀 Features

- **Dual-mode detection** — automatically routes images and videos to the correct detector
- **99.89% accuracy** on image deepfake detection (EfficientNet-B4)
- **rPPG liveness analysis** — detects heart rate, signal synchronization, phase lag
- **REST API** built with FastAPI
- **Web frontend** — drag and drop interface, works on any device
- **Debug mode** — saves signal plots, FFT visualizations, ROI previews

---

## 📁 Project Structure
biometric_liveness/
├── main.py                  # Video rPPG pipeline
├── image_detector.py        # Image detection (EfficientNet-B4)
├── api.py                   # FastAPI backend
├── frontend.html            # Web UI
├── config.py                # All pipeline parameters
├── requirements.txt
├── face_landmarker.task     # MediaPipe model
├── efficientnet_b4_deepfake.pth  # Trained model
│
├── src/
│   ├── video_loader.py      # FPS-aware frame extraction
│   ├── face_detector.py     # MediaPipe FaceMesh detection
│   ├── roi_extractor.py     # Skin ROI isolation
│   ├── signal_extractor.py  # Green channel extraction
│   ├── signal_filter.py     # Butterworth bandpass filter
│   ├── fft_analyzer.py      # FFT + Welch PSD analysis
│   ├── sync_checker.py      # Cross-region synchronization
│   └── debug_visualiser.py  # Diagnostic plots
│
├── tests/
│   ├── test_signal.py       # Signal processing tests
│   └── test_sync.py         # Synchronization tests
│
└── outputs/                 # Debug plots and logs

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepguard.git
cd deepguard

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm fastapi uvicorn python-multipart
```

---

## 🖥️ Usage

### Run the API server
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Open the web UI
Open `frontend.html` in your browser

### Test via command line

**Image:**
```bash
python image_detector.py "your_image.jpg"
```

**Video:**
```bash
python main.py "your_video.mp4" --debug
```

### API endpoint
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@your_file.jpg"
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.95% |
| Validation Accuracy | 99.89% |
| Dataset | 140k Real and Fake Faces |
| Architecture | EfficientNet-B4 |
| Training Epochs | 10 |
| Device | NVIDIA T4 GPU |

---

## 🔬 rPPG Pipeline Details

| Parameter | Value |
|-----------|-------|
| Bandpass Filter | 0.7 – 3.0 Hz |
| Heart Rate Range | 42 – 180 BPM |
| Min Sync Correlation | 0.50 |
| Max Phase Lag | 100 ms |
| Face Landmarks | 468 (MediaPipe FaceMesh) |
| ROI Regions | Forehead, Left Cheek, Right Cheek |

---

## 🛠️ Tech Stack

- **Python 3.12**
- **FastAPI** — REST API backend
- **PyTorch + timm** — EfficientNet-B4 training and inference
- **MediaPipe** — Face landmark detection
- **OpenCV** — Video processing
- **SciPy** — Signal filtering and FFT analysis
- **NumPy** — Vectorized signal processing
- **Matplotlib** — Debug visualizations

---

## 📝 API Reference

### `GET /health`
Returns server status.

### `POST /detect`
Upload an image or video for deepfake detection.

**Request:** `multipart/form-data` with `file` field

**Response (image):**
```json
{
  "mode": "image",
  "verdict": "REAL",
  "composite_score": 0.8821,
  "dct_score": 0.9012,
  "noise_score": 0.8543,
  "symmetry_score": 0.8911
}
```

**Response (video):**
```json
{
  "mode": "video",
  "verdict": "REAL",
  "composite_score": 0.8233,
  "heart_rate_bpm": 74.2,
  "fft_confidence": 0.6541,
  "synchronized": true,
  "correlation": 0.8202,
  "phase_lag_ms": 23.4
}
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Expected: **14 passed**

---

## 📌 Limitations

- Video analysis requires minimum 5 seconds of clear frontal face footage
- Image detection accuracy may vary on heavily compressed images
- rPPG detection requires adequate lighting conditions

---

## 👤 Author

**Shreyas**
Software Engineering Student

---

## 📄 License

MIT License