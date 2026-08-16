"""
scene_describer.py — optional natural-language scene description.

Everything else in this backend answers "is something close, and roughly
what/where" — useful for moment-to-moment navigation, but terse by
design (it has to be, since it's driving continuous audio cues). This
module answers a different question, on demand: "what's actually around
me right now," in plain language, the way a sighted companion might
describe it — "You're in a hallway, there's a chair a few feet to your
left, and a door straight ahead."

It sends the current camera frame to a vision-capable model on Groq and
asks for a short, navigation-focused description. Groq is a strong fit
here specifically because of inference speed — a live demo can't afford
a multi-second stall waiting on a description, and Groq's hosted
inference is built for low latency.

This is intentionally NOT part of the real-time detection loop — even a
fast LLM call is a fraction of a second to a couple seconds, and costs
money per call, so this is a deliberate, on-demand action (a button
press or voice command), not something firing automatically many times
a second like the rest of the pipeline.

Setup
-----
Needs a Groq API key (free tier available). Either:
  - set the GROQ_API_KEY environment variable (the SDK picks this up
    automatically), or
  - paste it into GROQ_API_KEY in app.py, same pattern as
    NGROK_AUTHTOKEN.
Needs `pip install groq` (see requirements.txt).

Fully optional and isolated: if no key is configured, or the package
isn't installed, is_ready() returns False and app.py skips this
feature entirely — nothing else breaks.
"""

import base64
import logging

import cv2

logger = logging.getLogger("echo-backend")

SDK_AVAILABLE = False
try:
    import groq
    SDK_AVAILABLE = True
except ImportError:
    groq = None

_client = None
_api_key = None

# Groq's current vision-capable models (Llama 4, natively multimodal).
# Scout is smaller/faster, Maverick is larger/higher-quality — Scout is
# the default here since demo responsiveness matters more than squeezing
# out the last bit of description quality. Swap the string below if
# Groq's model lineup has moved on by the time you're reading this —
# check https://console.groq.com/docs/models for what's current.
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SCENE_PROMPT = (
    "You are describing a scene to a blind person for navigation purposes. "
    "Look at this image from their forward-facing camera and describe what's "
    "immediately relevant in 1-2 short, plain sentences: obstacles, people, "
    "doorways, furniture, changes in the walking surface, anything they'd "
    "want to know before taking their next few steps. Be concrete and "
    "practical, not flowery — mention approximate position (left/right/ahead) "
    "where you can tell. If the scene is unclear or mostly empty, say so "
    "briefly rather than guessing."
)


def configure(api_key: str = None):
    """Call once at startup with an explicit key (e.g. from app.py's
    GROQ_API_KEY constant). If never called, falls back to the
    GROQ_API_KEY environment variable, which the SDK reads automatically
    when the client is constructed."""
    global _api_key
    _api_key = api_key or None


def _get_client():
    global _client
    if not SDK_AVAILABLE:
        return None
    if not is_ready():
        return None  # no key anywhere — fail cleanly with a friendly message rather than a confusing SDK error at call time
    if _client is not None:
        return _client
    try:
        _client = groq.Groq(api_key=_api_key) if _api_key else groq.Groq()
        return _client
    except Exception:
        logger.exception("Failed to construct Groq client.")
        return None


def is_ready() -> bool:
    """Best-effort check: package installed and *some* key is available
    (either explicitly configured or via environment variable). Doesn't
    guarantee the key is valid — that's only knowable on a real call."""
    if not SDK_AVAILABLE:
        return False
    import os
    return bool(_api_key or os.environ.get("GROQ_API_KEY"))


def describe_scene(frame_bgr, model: str = DEFAULT_MODEL) -> dict:
    """
    Returns {"ok": True, "description": str} on success, or
    {"ok": False, "error": str} on any failure — never raises, so the
    caller (a WebSocket handler) doesn't need special-case exception
    handling for this specific feature.
    """
    client = _get_client()
    if client is None:
        return {"ok": False, "error": "Scene description isn't configured (no API key or package missing)."}

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return {"ok": False, "error": "Could not encode frame."}
    b64_image = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64_image}"

    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        description = (completion.choices[0].message.content or "").strip()
        if not description:
            return {"ok": False, "error": "Empty response from the model."}
        return {"ok": True, "description": description}

    except groq.AuthenticationError:
        return {"ok": False, "error": "Invalid or missing Groq API key."}
    except groq.RateLimitError:
        return {"ok": False, "error": "Rate limited — try again in a moment."}
    except groq.APIConnectionError:
        return {"ok": False, "error": "Couldn't reach the Groq API — check the backend's internet connection."}
    except Exception as e:
        logger.exception("Scene description failed.")
        return {"ok": False, "error": f"Unexpected error: {e}"}
