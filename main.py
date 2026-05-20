# backend/main.py
# ═══════════════════════════════════════════════════════════════
# Radiologix — FastAPI Inference Server
# Model: YOLOv8m  |  Dataset: 1 class (Fracture)
# ═══════════════════════════════════════════════════════════════

import io
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("radiologix")

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Radiologix Inference API",
    description="YOLOv8 bone fracture detection endpoint",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Add your production domain here:
        # "https://radiologix.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model path — update this to your best.pt location ────────────
MODEL_PATH = Path(__file__).parent / "best.pt"

# ── Your dataset has 1 class: Fracture (class_id = 0)
# ── The frontend maps class_id → fracture TYPE using its own
#    FRACTURE_CLASSES array.  We return the raw class_id + label.
# ── If you retrain with multiple classes, update this dict:
YOLO_CLASS_MAP = {
    0: "Fracture",   # single-class model
    # If you train a 7-class model later:
    # 1: "Comminuted", 2: "Oblique", 3: "Transverse",
    # 4: "Spiral", 5: "Stress", 6: "Greenstick",
}

# ── Confidence / box-format note ─────────────────────────────────
# We return YOLO-normalised (x_top_left, y_top_left, w, h) so that
# the React frontend can overlay boxes on the displayed image
# without knowing the original pixel dimensions.
# ─────────────────────────────────────────────────────────────────

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        from ultralytics import YOLO
        if not MODEL_PATH.exists():
            logger.error(f"best.pt not found at {MODEL_PATH}")
            logger.error("Place your trained best.pt next to main.py")
            return
        model = YOLO(str(MODEL_PATH))
        # Warm-up inference (avoids cold-start latency on first request)
        dummy = Image.new("RGB", (640, 640), color=(128, 128, 128))
        model(dummy, verbose=False)
        logger.info(f"✅ Model loaded: {MODEL_PATH.name}")
        logger.info(f"   Classes: {model.names}")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        model = None


# ─────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model":  str(MODEL_PATH.name) if model else None,
        "classes": model.names if model else None,
    }


# ─────────────────────────────────────────────────────────────────
# PREDICT  —  POST /predict
# ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=1.0,
                        description="Confidence threshold"),
    iou:  float = Query(default=0.45, ge=0.01, le=1.0,
                        description="IoU threshold for NMS"),
):
    """
    Run YOLOv8 inference on an uploaded X-ray image.

    Returns a list of detections, each with:
    - classId     : integer class index
    - label       : human-readable class name
    - confidence  : float [0,1]
    - x, y        : top-left corner (normalised 0-1)
    - w, h        : box width/height (normalised 0-1)

    Query params:
    - conf  (default 0.25): minimum detection confidence
    - iou   (default 0.45): NMS IoU threshold
    """
    # Guard
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for best.pt path.",
        )

    # Validate file type
    allowed = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed)}",
        )

    try:
        # Read & convert image
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_w, img_h = pil_image.size

        # Run inference
        t0      = time.perf_counter()
        results = model(
            pil_image,
            conf    = conf,
            iou     = iou,
            verbose = False,
        )
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        result = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                # xyxyn = normalised [x1,y1,x2,y2]
                x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
                class_id  = int(box.cls[0])
                confidence = round(float(box.conf[0]), 4)

                # Label from model OR our override map
                label = (model.names.get(class_id)
                         or YOLO_CLASS_MAP.get(class_id, "Fracture"))

                detections.append({
                    "classId":    class_id,
                    "label":      label,
                    "confidence": confidence,
                    # top-left + size (normalised), matches React frontend format
                    "x": round(x1n, 4),
                    "y": round(y1n, 4),
                    "w": round(x2n - x1n, 4),
                    "h": round(y2n - y1n, 4),
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        logger.info(
            f"Predict | file={file.filename} | "
            f"detections={len(detections)} | {inference_ms}ms"
        )

        return {
            "detections":   detections,
            "inferenceMs":  inference_ms,
            "imageSize":    { "w": img_w, "h": img_h },
            "modelConf":    conf,
            "modelIou":     iou,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")