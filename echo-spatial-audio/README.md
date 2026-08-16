# EchoVision — Spatial Audio Obstacle Detector

Turns nearby obstacles into directional sound: angle → stereo position,
distance → loudness + pulse rate, height → pitch. Built as a live-demo
prototype for spatial-audio-assisted navigation.

```
project/
├── backend/          Python CV server (real-time obstacle detection)
│   ├── app.py         FastAPI + WebSocket server
│   ├── vision.py      optical-flow / edge-density detection logic
│   └── requirements.txt
├── frontend/
│   └── index.html    the demo UI — open this in a browser
└── README.md          you are here
```

## Quick start (no backend needed)

Just open `frontend/index.html` in a browser and click **Start**.
**Simulation mode** works immediately with no setup — a virtual room
with obstacles you navigate with WASD/arrow keys, with full real-time
HRTF spatial audio. This is the most reliable mode for a live demo.

## Running the camera backend

The backend now serves the frontend itself, so you only run **one**
server for everything — API and UI, same port.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then open **http://localhost:8000** in a browser (not the HTML file
directly — the server is now the thing serving it). Click **Start**,
switch to **Camera (experimental)** mode, allow camera access. The
Backend URL field auto-fills based on whatever host you loaded the page
from, so this works unmodified whether you're on `localhost`, a LAN IP,
or a tunnel URL (see below).

If the backend isn't running, or the WS connection fails/times out
after 3s, the frontend automatically falls back to a simpler in-browser
heuristic instead of breaking the demo — you'll see a note explaining
that, and a **Reconnect** button once the backend is up.

### Calibration

Both the backend and the local fallback learn your background for
~2 seconds before flagging anything as an obstacle — point the camera
at a normal, obstacle-free view when you switch into Camera mode. This
is what stops ordinary desk clutter, wall texture, or a hand resting in
frame from reading as a constant obstacle in every zone: only things
that appear or move closer *relative to that learned baseline* trigger
a cue. If you move the camera to a very different scene mid-session
(different room, very different lighting), hit **Recalibrate**. There's
also a **Camera sensitivity** slider if the default feels too twitchy
or too dull for your lighting/webcam.

### Live bounding-box overlay

Camera mode shows the actual video feed with colored rectangles drawn
around detected regions, updated in real time:
- 🟥 **Red** — close (inner third of your detection range)
- 🟨 **Yellow** — medium distance
- 🟩 **Green** — detected but comfortably far

When connected to the backend these are real bounding boxes from
contour detection on the calibrated closeness map (see
`ObstacleDetector.process()` in `vision.py`), each labeled with its
estimated distance. In the local-fallback mode (no backend), this is
approximated with a coarser full-height rectangle per active zone,
since the in-browser heuristic doesn't do proper contour detection.

### Voice announcements and alert log

Toggle **Voice announcements** on to have it speak up when something
enters the close range ("Obstacle ahead-left, 1.8 meters, waist
height"), and again once the path clears. Announcements are
transition-triggered and rate-limited, not continuous — the spatial
audio pulses already give fine-grained awareness; voice is for the
moments that actually need your attention. There's also an **Announce
current surroundings** button for an on-demand status check anytime.

The **Alert Log** panel keeps a running history of close-proximity
events (distance, direction, height, how long ago), useful for
reviewing what happened during a walk-through or demo without having to
remember it in the moment.

### Accessibility

- **Hands-free voice control** — toggle "🎙️ Voice control" on and
  operate the whole thing by speaking: *"start"*, *"stop"*, *"camera
  mode"*, *"simulation mode"*, *"recalibrate"*, *"what's ahead"*,
  *"louder"*, *"quieter"*, *"mute"*, or *"help"* to hear the list again.
  It automatically mutes its own mic while speaking a response, so it
  doesn't hear itself and misfire on its own confirmations. Needs a
  browser with speech recognition support (Chrome or Edge); the button
  disables itself with an explanation otherwise.
- **Haptic feedback** — on devices that support vibration, a short buzz
  fires when something enters the close range, so proximity can be felt
  as well as heard (useful in loud environments, or as a second
  channel alongside audio).
- **Screen reader support on the interface itself, not just the
  concept** — every slider now has a properly associated label (they
  weren't before — a real bug this fixed), the calibration/connection
  status is an announced live region, and there's a skip-to-controls
  link for keyboard users to bypass the header.
- **🔊 Voice Guide button** — sits right at the top of the page, works
  immediately with no setup. Press it (or say "guide" once voice
  control is on) anytime for a full spoken orientation: what the site
  does, how to start, current status, and every available command. This
  is meant to be the very first thing a blind user reaches, so they
  don't have to explore the whole interface via screen reader just to
  find out what's possible.

### Object tracking and time-to-collision

Camera mode (backend-connected) doesn't just report distance per frame
— it tracks the same object across frames (`vision.py`'s
`ObjectTracker`, a simple centroid tracker) and estimates its closing
velocity in m/s. From that it computes **time-to-collision**: how many
seconds until contact at the current closing rate.

This matters because raw distance alone can't distinguish a slowly
drifting object at 2m from one sprinting toward you from 5m — TTC can.
When TTC drops below ~2.5s, that box is escalated to "close" severity
regardless of its raw distance, the overlay shows a distinct
"⚠ Xs TO CONTACT" label, and — if voice announcements are on — you get
a distinct, faster-repeating spoken warning ("Fast approaching
obstacle... contact in about 1.8 seconds") separate from the normal
close-range announcement.

### Object voices (Doppler)

An optional upgrade to the audio itself, in the **Advanced Audio**
panel. Normally, spatial audio comes from 7 fixed left-to-right zones —
coarse but always-on directional coverage. Turn on **Object voices**
(camera mode, backend connected) and each individually *tracked*
object instead gets its own persistent tone that follows its real
position continuously — rather than snapping between 7 fixed slots —
and pitch-shifts based on closing speed: faster approach, higher pitch;
moving away, lower pitch. It's a stylized cue inspired by the Doppler
effect, not physically exact acoustics, but an intuitive "this one's
coming in hot" signal that rides on the same object-tracking data
driving the time-to-collision warnings above. The fixed zone audio
automatically steps aside while this is active, so you don't get two
competing sounds for the same obstacle.

## Using it on a phone

### Audio-only (Simulation mode) over local WiFi

No HTTPS needed for this — plain HTTP is fine since it doesn't touch
the camera.

1. Find your computer's LAN IP (`ipconfig` on Windows / `ifconfig` or
   `ip addr` on Mac/Linux) — something like `192.168.1.23`.
2. Start the backend as above (or just serve `frontend/` with
   `python3 -m http.server 5500` if you don't need the backend at all).
3. On your phone, join the same WiFi network, then visit
   `http://192.168.1.23:8000` (or `:5500` if you used the simple static
   server) in a mobile browser.

### Camera mode on a phone

Mobile browsers require **HTTPS** to grant camera access
(`getUserMedia`) — a plain `http://192.168.x.x` URL will not be allowed
to prompt for the camera at all, even on your own network. The
easiest way around this for a demo is a tunnel like
[ngrok](https://ngrok.com). Two ways to run it:

**Option A — built into the server (one window, no separate ngrok command)**

1. Get a free authtoken from your [ngrok dashboard](https://dashboard.ngrok.com) → "Your Authtoken".
2. Open `backend/app.py` and paste it into `NGROK_AUTHTOKEN = ""` near the top.
3. `pip install pyngrok` if you haven't already (it's in `requirements.txt`).
4. Start the backend as usual. The public URL prints directly in this
   same terminal, boxed in `====` lines so it's easy to spot:
   ```
   ====================================================================
    Public URL (paste into the site's Backend Server field, or open directly): https://abcd-1234.ngrok-free.app
   ====================================================================
   ```
5. Open that URL on your phone, or paste it into the **Backend Server**
   field on the page if you're accessing the frontend some other way.

Leave `NGROK_AUTHTOKEN` blank (the default) to skip this entirely and
run purely locally — nothing else about the server changes.

**Option B — run ngrok yourself, in a second terminal**

1. Install ngrok and authenticate once (`ngrok config add-authtoken ...`,
   from your ngrok dashboard).
2. Start the backend as above (`uvicorn app:app --host 0.0.0.0 --port 8000`).
3. In a separate terminal:
   ```bash
   ngrok http 8000
   ```
4. ngrok prints a URL like `https://abcd1234.ngrok-free.app` — open
   that exact URL on your phone. Because the backend now serves the
   frontend on the same port, this single HTTPS URL gives you the full
   app: UI, camera permission prompt, and a working `wss://` connection
   back to your computer, all through one tunnel.

Either way, the page's **Backend Server** field also accepts the raw
`https://...` link directly and converts it to the right WebSocket URL
itself — no need to hand-edit it into a `wss://.../ws/obstacles` form.

## AI scene description

The rest of this app describes obstacles tersely by design — it's
driving continuous audio cues, so it has to be quick. The **Describe
what's around me** button (Camera mode) is different: it's an on-demand
action that sends the current frame to a vision-capable model on
[Groq](https://groq.com) and asks for a short, natural description of
the whole scene — *"You're in a hallway, there's a chair a few feet to
your left, and a door straight ahead"* — the way a sighted companion
might describe it, not just a list of proximity readings. Groq's a good
fit here specifically for its inference speed — a live demo can't
afford a multi-second stall waiting on a description.

**To enable it:**
1. Get a free API key from [console.groq.com/keys](https://console.groq.com/keys).
2. Either paste it into `GROQ_API_KEY = ""` near the top of
   `backend/app.py`, or set it as an environment variable of the same
   name — either works.
3. `pip install groq` (included in `requirements.txt`).

Leave it unset and this feature just doesn't appear — nothing else
changes. Reachable by button, or by saying *"describe"*, *"what do you
see"*, or *"look around"* if Voice control is on.

This is deliberately kept separate from the real-time detection loop:
LLM calls take real time and cost money per call, so it only fires when
you actually ask for it, not continuously. The default model
(`meta-llama/llama-4-scout-17b-16e-instruct`, set in
`backend/scene_describer.py`) is Groq's smaller/faster vision option —
swap in the larger Maverick variant there if you want higher-quality
descriptions at the cost of some speed.

## How the backend detection works

`backend/vision.py` implements two classical, model-free computer-vision
techniques per vertical "zone" of the camera frame, combined and
temporally smoothed:

1. **Dense optical flow** (Farneback) — large flow magnitude in a zone
   suggests something is approaching or moving quickly through it.
2. **Edge/contrast density** — a proxy for "something textured and
   probably solid is close," even with no motion.

A zone's vertical energy centroid is used to guess whether the obstacle
is ground-level, waist-level ("mid"), or head-height.

This is an approximation, not true depth — it needs no model weights
and runs instantly on CPU, which makes it reliable for a live demo. For
a production system, swap `ObstacleDetector.process()` in `vision.py`
for a real monocular depth model (MiDaS, Depth Anything) or LiDAR input,
keeping the same per-zone `{"dist": meters, "height": str}` output
contract — nothing else in the pipeline needs to change.

## Extending it

- `backend/vision.py` — swap in a real depth model here.
- `frontend/index.html` — sonification mapping (pan/volume/pitch logic)
  lives in the `pulseZone()` and `frame()` functions.
- The WebSocket protocol (see `backend/app.py` docstring) is simple
  JSON in/out — easy to reimplement in another language if you want a
  non-Python backend later.
