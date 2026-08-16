"""
app.py — Backend server for the Echo spatial-audio obstacle detector.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in a browser — this server serves the
frontend directly, so no second static file server is needed. This also
makes it trivial to expose the whole thing through a single HTTPS tunnel
(e.g. ngrok) for mobile camera testing, since getUserMedia requires HTTPS
on phones.

Optional built-in ngrok tunnel
--------------------------------
Paste your ngrok authtoken into NGROK_AUTHTOKEN below (get one free at
ngrok.com -> dashboard -> "Your Authtoken") and this server will open a
public HTTPS tunnel itself on startup and print the URL right here in
this terminal — no second window, no separate `ngrok http 8000` command.
Leave it blank to run purely locally as before; nothing else changes.
Needs `pip install pyngrok` (see requirements.txt).

Exposes:
    GET  /health                       simple liveness check
    WS   /ws/obstacles                 streaming obstacle detection
    GET  /                             the frontend (frontend/index.html)

WebSocket protocol
-------------------
Server -> Client, sent immediately on connect:
    {"type": "capabilities", "ai_available": true}

    "ai_available" is true only if object_detector.py's YOLOv8n model is
    both installed (`pip install ultralytics`) AND loaded successfully.
    Use this to decide whether to show/enable an "AI object ID" toggle —
    there's no point offering it if the backend can't actually do it.

Client -> Server, first message (JSON), optional config:
    {"type": "config", "num_zones": 7, "max_range_m": 7.0, "sensitivity": 0.5,
     "use_ai": false}

    "use_ai" switches boxes from classical contour detection (default,
    always available) to real object identification via YOLOv8n — only
    takes effect if the "capabilities" message reported ai_available.
    When true, each box additionally carries "label" (a COCO class name
    like "person" or "chair") and "confidence" (0..1).

Client -> Server, subsequent messages (JSON):
    {"type": "frame", "image": "<base64 JPEG, no data-URL prefix>"}

Client -> Server, to relearn the background baseline (point camera at a
clear, obstacle-free view first):
    {"type": "reset"}

Server -> Client:
    {"type": "zones",
     "zones": [ {"dist": 2.3, "height": "mid"} | null, ... ],
     "boxes": [ {"x":0.1,"y":0.2,"w":0.15,"h":0.3,"dist":1.8,
                 "height":"mid","severity":"close",
                 "track_id":3,"velocity_mps":0.6,"ttc_seconds":3.0,
                 "label":"person","confidence":0.87}, ... ],
     "num_zones": 7, "calibrating": false, "t": 1234567.89}

    "boxes" are real detected regions — either contour bounding boxes on
    the calibrated closeness map (classical, default), or real YOLOv8n
    detections (if use_ai is on) — normalized 0..1 relative to frame
    width/height, for drawing a live overlay. "severity" is "close" |
    "medium" | "safe" based on distance thresholds. "label" and
    "confidence" are only present when use_ai is on; otherwise "label"
    is null.

    Each box also carries object-tracking fields:
      "track_id"      stable identity across frames (same object keeps
                       the same id as long as it isn't lost for more
                       than a handful of frames)
      "velocity_mps"  closing speed in meters/second; positive means
                       approaching, negative means moving away
      "ttc_seconds"   estimated time-to-collision if still approaching
                       at the current rate, else null. A low TTC
                       escalates "severity" to "close" even if the raw
                       distance alone wouldn't — something closing fast
                       deserves urgency sooner than distance alone implies.

    While "calibrating" is true, zones and boxes will be empty/null —
    the detector is still learning what the background normally looks like.

    {"type": "error", "message": "..."}

On-demand scene description (separate from the real-time loop above)
-------------------------------------------------------------------------
Client -> Server:
    {"type": "describe_scene", "image": "<base64 JPEG>"}

Server -> Client:
    {"type": "scene_description", "ok": true, "description": "..."}
    {"type": "scene_description", "ok": false, "error": "..."}

This is deliberately NOT part of the ~10fps frame loop — it's a single
request/response pair triggered by a button press or voice command,
since it calls out to an LLM (seconds of latency, real cost per call),
unlike the always-on classical/AI detection above.

The server keeps one ObstacleDetector instance per connection so optical
flow state (previous frame) isn't shared between simultaneous clients.
"""

import base64
import time
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from vision import ObstacleDetector
import object_detector
import scene_describer

# ============================================================
# Paste your ngrok authtoken here to have this server open a public
# HTTPS tunnel automatically on startup and print the URL below.
# Get a free token at https://dashboard.ngrok.com -> "Your Authtoken".
# Leave as "" to run purely locally (default) — nothing else changes.
# ============================================================
NGROK_AUTHTOKEN = ""

# ============================================================
# Paste your Groq API key here to enable on-demand natural-language
# scene descriptions (free tier at https://console.groq.com/keys). You
# can also set the GROQ_API_KEY environment variable instead — either
# works, this field just takes priority if both are set.
# Leave as "" to skip this feature — nothing else changes.
# ============================================================
GROQ_API_KEY = ""
scene_describer.configure(api_key=GROQ_API_KEY or None)

PORT = 8000  # keep in sync with however you actually run uvicorn (--port)

try:
    from pyngrok import ngrok as _ngrok
    _PYNGROK_AVAILABLE = True
except ImportError:
    _PYNGROK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echo-backend")

app = FastAPI(title="EchoVision Obstacle Detection Backend")

# Wide-open CORS for local demo use. Tighten this before deploying anywhere
# other than your own machine / local network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up_ai_model():
    if not object_detector.AI_AVAILABLE:
        return

    def _warm_up():
        try:
            logger.info("Pre-warming AI object detection model in the background...")
            t0 = time.time()
            dummy_frame = np.zeros((270, 480, 3), dtype=np.uint8)
            object_detector.detect_objects(dummy_frame)  # loads + runs one inference
            logger.info("AI model ready (warm-up took %.1fs).", time.time() - t0)
        except Exception:
            logger.exception("AI model warm-up failed — it'll still lazy-load on first real use.")

    # runs in a background thread so it never delays server startup or
    # blocks the event loop; if nobody ever turns AI mode on, this was
    # cheap and harmless, just some idle CPU for a few seconds at boot
    threading.Thread(target=_warm_up, daemon=True).start()


@app.on_event("startup")
def start_ngrok_tunnel():
    if not NGROK_AUTHTOKEN:
        return  # running purely locally — the normal case, nothing to do
    if not _PYNGROK_AVAILABLE:
        logger.warning(
            "NGROK_AUTHTOKEN is set but pyngrok isn't installed. "
            "Run: pip install pyngrok"
        )
        return

    try:
        _ngrok.set_auth_token(NGROK_AUTHTOKEN)

        # with --reload, this can fire again on every file change — reuse
        # an existing tunnel on this port instead of opening duplicates.
        # ngrok's v2 API puts the local address in tunnel.config["addr"];
        # v3 (now the default for new accounts) puts it in
        # tunnel.upstream["url"] instead — check both.
        def _tunnel_targets_our_port(t):
            addr = t.config.get("addr", "") or t.upstream.get("url", "")
            return addr.endswith(f":{PORT}")

        existing = [t for t in _ngrok.get_tunnels() if _tunnel_targets_our_port(t)]
        if existing:
            public_url = existing[0].public_url
        else:
            tunnel = _ngrok.connect(PORT, "http")
            public_url = tunnel.public_url

        if public_url.startswith("http://"):
            public_url = "https://" + public_url[len("http://"):]

        banner = f" Public URL (paste into the site's Backend Server field, or open directly): {public_url} "
        border = "=" * len(banner)
        logger.info(border)
        logger.info(banner)
        logger.info(border)
        print(f"\n{border}\n{banner}\n{border}\n")
    except Exception:
        logger.exception("Failed to start ngrok tunnel — continuing without it.")


@app.on_event("shutdown")
def stop_ngrok_tunnel():
    if NGROK_AUTHTOKEN and _PYNGROK_AVAILABLE:
        try:
            _ngrok.kill()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok", "service": "echovision-obstacle-backend"}


def decode_jpeg_base64(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


@app.websocket("/ws/obstacles")
async def ws_obstacles(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected: %s", websocket.client)

    # Let the frontend know whether real object identification and/or
    # scene description are even possible on this backend, so it can
    # show/enable those controls only when they'll actually do something.
    await websocket.send_json({
        "type": "capabilities",
        "ai_available": object_detector.AI_AVAILABLE,
        "describe_scene_available": scene_describer.is_ready(),
    })

    detector = ObstacleDetector(num_zones=7, max_range_m=7.0, use_ai=False)
    frame_count = 0

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "config":
                num_zones = msg.get("num_zones")
                max_range_m = msg.get("max_range_m")
                sensitivity = msg.get("sensitivity")
                ignore_bottom_fraction = msg.get("ignore_bottom_fraction")
                use_ai = msg.get("use_ai")
                detector.configure(
                    num_zones=num_zones, max_range_m=max_range_m,
                    sensitivity=sensitivity, ignore_bottom_fraction=ignore_bottom_fraction,
                    use_ai=use_ai
                )
                logger.info(
                    "Configured: num_zones=%s max_range_m=%s sensitivity=%s ignore_bottom_fraction=%s use_ai=%s",
                    detector.num_zones, detector.max_range_m, detector.sensitivity,
                    detector.ignore_bottom_fraction, detector.use_ai
                )
                continue

            if msg_type == "frame":
                try:
                    frame = decode_jpeg_base64(msg["image"])
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"bad frame: {e}"})
                    continue

                # Run off the event loop: classical CV was fast enough that
                # blocking here never mattered, but AI inference — especially
                # the first call, which also loads/downloads the model — can
                # take seconds, long enough to miss WebSocket keepalive pings
                # and get the connection dropped if run inline.
                zones, boxes, calibrating = await run_in_threadpool(detector.process, frame)
                frame_count += 1

                await websocket.send_json({
                    "type": "zones",
                    "zones": zones,
                    "boxes": boxes,
                    "num_zones": detector.num_zones,
                    "calibrating": calibrating,
                    "t": time.time(),
                })
                continue

            if msg_type == "reset":
                detector.reset()
                continue

            if msg_type == "describe_scene":
                try:
                    frame = decode_jpeg_base64(msg["image"])
                except Exception as e:
                    await websocket.send_json({"type": "scene_description", "ok": False, "error": f"bad frame: {e}"})
                    continue

                # LLM calls take real seconds — always off the event loop,
                # same reasoning as the AI-detection path above but more so
                result = await run_in_threadpool(scene_describer.describe_scene, frame)
                await websocket.send_json({"type": "scene_description", **result})
                continue

            await websocket.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s (frames processed: %d)", websocket.client, frame_count)
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# Serve the frontend from the same server/port. Mounted last so it acts
# as a catch-all and doesn't shadow /health or /ws/obstacles above.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s — only the API will be served.", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
