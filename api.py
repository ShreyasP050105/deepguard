"""
Deepfake Detection API
----------------------
POST /detect — accepts image or video, auto-routes to correct detector
GET  /health — health check
"""

from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from config import CONFIG
from src.utils import setup_logger
from image_detector import ImageDeepfakeDetector
from main import LivenessPipeline

setup_logger(CONFIG.log_level, CONFIG.output_dir)

app = FastAPI(
    title="Deepfake Detection API",
    description="Detects deepfakes in images (visual forensics) and videos (rPPG liveness)",
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialise detectors once at startup
image_detector = ImageDeepfakeDetector()
video_pipeline = LivenessPipeline()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. "
                   f"Supported: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}"
        )

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if suffix in IMAGE_EXTENSIONS:
            result = image_detector.run(tmp_path)
            return JSONResponse({
                "mode":             "image",
                "verdict":          result.verdict,
                "composite_score":  round(result.composite_score, 4),
                "dct_score":        round(result.dct_score, 4),
                "noise_score":      round(result.noise_score, 4),
                "symmetry_score":   round(result.symmetry_score, 4),
            })

        else:
            result = video_pipeline.run(tmp_path)
            return JSONResponse({
                "mode":             "video",
                "verdict":          result.verdict,
                "composite_score":  round(result.composite_score, 4),
                "heart_rate_bpm":   round(result.dominant_bpm, 1),
                "fft_confidence":   round(result.confidence, 4),
                "synchronized":     result.synchronized,
                "correlation":      round(result.correlation, 4),
                "phase_lag_ms":     round(result.details["phase_lag_ms"], 1),
            })

    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    finally:
        Path(tmp_path).unlink(missing_ok=True)