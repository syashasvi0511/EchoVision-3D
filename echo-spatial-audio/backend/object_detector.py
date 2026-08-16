"""
object_detector.py — optional real object identification.

Everything else in this backend (vision.py) is a from-scratch classical
computer-vision approach that identifies "something is here" without
knowing what it is. This module adds actual object classification using
YOLOv8n (the "nano" — smallest, fastest — variant), pretrained by
Ultralytics on the COCO dataset: 80 everyday object classes (person,
chair, backpack, dog, car, bicycle, ...). This is a real trained model
on a real, standard, widely-used object-detection dataset — not a
placeholder.

This is intentionally optional and isolated:
- If the `ultralytics` package isn't installed, AI_AVAILABLE is False
  and every function here becomes a no-op. app.py and vision.py check
  this flag and fall back to the classical detection automatically —
  nothing else breaks.
- The model weights (~6MB) download automatically the first time
  get_model() is called, from Ultralytics' GitHub releases. That
  requires internet on whatever machine runs this backend, once.

To enable: `pip install ultralytics` (see requirements-ai.txt).

Import cost: importing `ultralytics` transitively imports `torch`,
which is slow (can take many seconds, sometimes much longer). To keep
backend startup fast for everyone — including people who never touch
this feature — that import is deferred until the very first time AI
detection actually runs, not at module load. Availability is checked
cheaply via `importlib.util.find_spec` instead, which doesn't pay
that cost.

Design note: this module only answers "what is it and roughly where in
the frame" — it does NOT estimate distance. Real depth still comes from
the calibrated background-subtraction pipeline already built and tested
in vision.py. ObstacleDetector fuses the two: YOLO's box tells us WHAT
and WHERE (tight, labeled, real detection), the calibrated closeness
map at that same region tells us HOW FAR. This keeps the tested
distance/tracking/TTC pipeline completely unchanged either way.
"""

import importlib.util
import logging

logger = logging.getLogger("echo-backend")

# Cheap presence check — doesn't import the package (and therefore
# doesn't import torch) just to answer "is it installed at all".
AI_AVAILABLE = importlib.util.find_spec("ultralytics") is not None

_model = None
_load_attempted = False
_load_error = None


def get_model():
    """Lazily loads (and downloads, first time) the YOLOv8n model.
    Returns None if unavailable or if loading failed. The actual heavy
    ultralytics/torch import happens here, on first real use — not at
    module import time, so backend startup stays fast either way."""
    global _model, _load_attempted, _load_error
    if not AI_AVAILABLE:
        return None
    if _model is not None:
        return _model
    if _load_attempted and _load_error is not None:
        return None  # already tried and failed this session; don't retry every frame
    _load_attempted = True
    try:
        from ultralytics import YOLO  # deferred: this is the slow (torch) import
        logger.info("Loading YOLOv8n (COCO-pretrained)... (downloads weights on first run)")
        _model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded — %d COCO classes available.", len(_model.names))
        return _model
    except Exception as e:
        _load_error = e
        logger.exception("Failed to load YOLOv8n — falling back to classical detection.")
        return None


def is_ready() -> bool:
    """True only if the package is installed AND the model actually loaded."""
    return AI_AVAILABLE and _model is not None


def detect_objects(frame_bgr, conf_threshold: float = 0.35) -> list:
    """
    Runs object detection on a BGR frame (numpy array, same coordinate
    space the caller will use for everything else — no internal resize).

    Returns a list of:
      {"x0","y0","x1","y1": pixel coords in frame_bgr,
       "label": str (COCO class name),
       "confidence": float 0..1}
    Empty list if the model isn't available.
    """
    model = get_model()
    if model is None:
        return []

    results = model.predict(frame_bgr, verbose=False, conf=conf_threshold)
    if not results:
        return []

    r = results[0]
    names = r.names
    detections = []
    for box in r.boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "label": names.get(cls_id, "object"),
            "confidence": round(conf, 2),
        })
    return detections
