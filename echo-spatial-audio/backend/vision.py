"""
vision.py — Monocular obstacle sensing without a depth model.

Real depth estimation from a single moving camera normally needs either
specialized hardware (LiDAR, stereo rig) or a trained depth model
(MiDaS, Depth Anything, etc). This module is a from-scratch classical
computer-vision fallback that needs nothing but OpenCV — no model
weights to download, so it works fully offline and starts instantly.

Two outputs are produced from the same underlying analysis:

1. Per-zone "closeness" (unchanged concept from earlier versions) —
   drives the spatial audio: coarse left-to-right zones, each either
   clear or carrying a distance + height estimate.

2. Bounding boxes — for the live visual overlay. Detected regions are
   found by contour detection on a thresholded "how much closer than
   background is this area" map, then colored by proximity: red
   (close), yellow (medium), green (further but still flagged).

Technique, computed on a fine grid over the frame:

  a. Dense optical flow (Farneback) between consecutive frames — large
     flow magnitude suggests something approaching or moving quickly.
  b. Edge / contrast density (Sobel) — a proxy for "something textured
     and probably solid is here," useful even with zero motion.

Calibration
-----------
A raw closeness score is not comparable across scenes on its own: a
patterned wall or a cluttered desk can sit at a "high contrast"
baseline with nothing actually close. The detector spends its first
~2 seconds after start/reset learning the room — averaging the raw
score per grid cell — and only flags a cell (or the zone/box built from
it) as "obstacle present" once it rises meaningfully above its own
learned baseline. Calibrate while pointing the camera at a normal,
obstacle-free view.

Persistence filtering
----------------------
A single noisy or momentarily-bright frame shouldn't trigger a cue.
Each grid cell tracks how many *consecutive* frames it has stayed above
threshold; only cells sustained for PERSISTENCE_FRAMES in a row count
as a real detection. This filters one-off camera noise and quick
incidental motion (a hand passing briefly through frame) while still
reacting quickly to something that's actually approaching, since a real
approaching obstacle keeps registering frame after frame.

Bottom-frame exclusion
------------------------
On a laptop, the webcam often has your hands/keyboard in the lower part
of the frame during ordinary use — not an obstacle in a walking path.
`ignore_bottom_fraction` crops that region out of detection entirely
(default 15% of frame height). Set it to 0 if the camera is mounted
somewhere this doesn't apply (chest-mounted, glasses, cane-mounted).

This is intentionally honest about being an approximation, not true
depth. Swap this module for a real depth model later and keep the same
output contract (zones + boxes) — the frontend doesn't need to change.

Optional real object identification
-------------------------------------
By default, boxes come from contour detection on the closeness map —
real detection of "something is here," but with no idea what "it" is.
If `object_detector.py`'s optional YOLOv8n model is installed and
enabled (see that module), boxes instead come from real object
classification against the COCO dataset (person, chair, backpack, dog,
...), while distance is still computed from this module's own
calibrated closeness map at that region — the tested distance/tracking
pipeline doesn't change, only the box source and the added label.
"""

import time
import cv2
import numpy as np

import object_detector


class ObjectTracker:
    """
    Lightweight centroid tracker that gives each detected box a stable
    identity across frames, purely from the outputs ObstacleDetector
    already computes — no extra model, no extra camera pass.

    Why this matters: a single frame's distance only tells you how far
    something is *right now*. Tracking the same object across frames
    lets us also compute how fast it's closing the gap (velocity) and,
    from that, a time-to-collision estimate — the standard "how many
    seconds until this reaches me" metric used in real collision-warning
    systems. A slowly-drifting object at 2m and a sprinting one at 5m
    can warrant very different urgency; raw distance alone can't tell
    them apart, TTC can.

    Matching is deliberately simple (greedy nearest-centroid, not the
    Hungarian algorithm) since there are only ever a handful of boxes
    per frame — no need for anything fancier here.
    """
    MAX_MISSED_FRAMES = 6      # frames a track can go unmatched before being dropped
    MATCH_MAX_DIST = 0.18      # max normalized centroid movement to still count as the same object
    VELOCITY_HISTORY = 6       # frames of distance history used to estimate closing speed
    URGENT_TTC_SECONDS = 2.5   # time-to-collision below which severity is escalated regardless of raw distance

    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    def reset(self):
        self.tracks = {}
        self._next_id = 1

    def update(self, boxes: list) -> list:
        now = time.time()
        detections = [
            {"cx": b["x"] + b["w"] / 2, "cy": b["y"] + b["h"] / 2, "box": b}
            for b in boxes
        ]
        used = set()
        matched_ids = set()

        # match existing tracks to this frame's detections (greedy nearest-centroid)
        for tid, track in list(self.tracks.items()):
            best_i, best_d = None, None
            for i, det in enumerate(detections):
                if i in used:
                    continue
                d = ((det["cx"] - track["cx"]) ** 2 + (det["cy"] - track["cy"]) ** 2) ** 0.5
                if d <= self.MATCH_MAX_DIST and (best_d is None or d < best_d):
                    best_d, best_i = d, i
            if best_i is not None:
                det = detections[best_i]
                used.add(best_i)
                matched_ids.add(tid)
                track["cx"], track["cy"] = det["cx"], det["cy"]
                track["missed"] = 0
                track["age"] = track.get("age", 0) + 1
                track["dist_history"].append((now, det["box"]["dist"]))
                if len(track["dist_history"]) > self.VELOCITY_HISTORY:
                    track["dist_history"] = track["dist_history"][-self.VELOCITY_HISTORY:]
                det["box"]["track_id"] = tid
            else:
                track["missed"] = track.get("missed", 0) + 1

        # unmatched detections become new tracks
        for i, det in enumerate(detections):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "cx": det["cx"], "cy": det["cy"], "missed": 0, "age": 1,
                "dist_history": [(now, det["box"]["dist"])],
            }
            det["box"]["track_id"] = tid
            matched_ids.add(tid)

        # drop stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].get("missed", 0) > self.MAX_MISSED_FRAMES:
                del self.tracks[tid]

        # velocity + time-to-collision from each track's distance history
        for det in detections:
            box = det["box"]
            track = self.tracks.get(box.get("track_id"))
            if not track:
                box["velocity_mps"] = 0.0
                box["ttc_seconds"] = None
                continue

            hist = track["dist_history"]
            velocity = 0.0
            if len(hist) >= 2:
                (t0, d0), (t1, d1) = hist[0], hist[-1]
                dt = t1 - t0
                if dt > 0.05:
                    velocity = (d0 - d1) / dt  # positive = approaching (distance shrinking)

            box["velocity_mps"] = round(float(velocity), 2)
            if velocity > 0.05:
                ttc = hist[-1][1] / velocity
                box["ttc_seconds"] = round(float(min(ttc, 999)), 1)
                if box["ttc_seconds"] < self.URGENT_TTC_SECONDS:
                    box["severity"] = "close"  # escalate: fast approach matters even if not yet "close" by distance
            else:
                box["ttc_seconds"] = None

        return [d["box"] for d in detections]


class ObstacleDetector:
    CALIBRATION_FRAMES = 20  # ~2s at 10fps
    PERSISTENCE_FRAMES = 3   # consecutive frames above threshold before counting
    GRID_COLS = 16
    GRID_ROWS = 9
    MAX_BOXES = 8

    def __init__(self, num_zones: int = 7, max_range_m: float = 7.0,
                 sensitivity: float = 0.5, ignore_bottom_fraction: float = 0.22,
                 use_ai: bool = True,
                 proc_width: int = 480, proc_height: int = 270):
        self.num_zones = num_zones
        self.max_range_m = max_range_m
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        self.ignore_bottom_fraction = float(np.clip(ignore_bottom_fraction, 0.0, 0.6))
        self.use_ai = use_ai
        self.proc_width = proc_width
        self.proc_height = proc_height
        self.prev_gray = None

        self.smoothing_alpha = 0.35
        self._smoothed_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self._streak = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int32)
        self.tracker = ObjectTracker()

        self._reset_calibration()

    def _reset_calibration(self):
        self.calibrating = True
        self._calib_count = 0
        self._baseline_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self._streak[:] = 0

    def configure(self, num_zones: int = None, max_range_m: float = None,
                  sensitivity: float = None, ignore_bottom_fraction: float = None,
                  use_ai: bool = None):
        if num_zones is not None:
            self.num_zones = num_zones
        if max_range_m is not None:
            self.max_range_m = max_range_m
        if sensitivity is not None:
            self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        if ignore_bottom_fraction is not None:
            self.ignore_bottom_fraction = float(np.clip(ignore_bottom_fraction, 0.0, 0.6))
        if use_ai is not None:
            self.use_ai = use_ai

    def reset(self):
        """Re-run calibration against whatever the camera currently sees."""
        self.prev_gray = None
        self._smoothed_grid[:] = 0
        self._reset_calibration()
        self.tracker.reset()

    def _delta_threshold(self) -> float:
        return 0.22 - self.sensitivity * 0.18  # 0.22 (low sens) .. 0.04 (high sens)

    def _severity(self, dist: float) -> str:
        if dist < self.max_range_m * 0.33:
            return "close"
        if dist < self.max_range_m * 0.66:
            return "medium"
        return "safe"

    @staticmethod
    def _height_from_edges(edge_mag, y0, y1, x0, x1, frame_h):
        region = edge_mag[y0:y1, x0:x1]
        if region.size == 0:
            return "mid"
        row_energy = region.sum(axis=1)
        total = row_energy.sum()
        if total <= 1e-6:
            return "mid"
        local_rows = np.arange(region.shape[0])
        centroid_local = float((row_energy * local_rows).sum() / total)
        centroid_y = (y0 + centroid_local) / frame_h
        if centroid_y > 0.62:
            return "ground"
        if centroid_y < 0.38:
            return "head"
        return "mid"

    def process(self, frame_bgr: np.ndarray):
        """
        Returns (zones, boxes, calibrating).

        zones: list[num_zones] of None | {"dist": float, "height": str}
        boxes: list of {"x","y","w","h" (all 0..1 normalized),
                        "dist": float, "height": str, "severity": str}
               empty while calibrating.
        calibrating: bool
        """
        frame = cv2.resize(frame_bgr, (self.proc_width, self.proc_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        else:
            flow_mag = np.zeros_like(gray, dtype=np.float32)
        self.prev_gray = gray

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        h, w = gray.shape
        cell_h = h // self.GRID_ROWS
        cell_w = w // self.GRID_COLS

        raw_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        for r in range(self.GRID_ROWS):
            y0, y1 = r * cell_h, (h if r == self.GRID_ROWS - 1 else (r + 1) * cell_h)
            for c in range(self.GRID_COLS):
                x0, x1 = c * cell_w, (w if c == self.GRID_COLS - 1 else (c + 1) * cell_w)
                flow_score = float(np.clip(np.mean(flow_mag[y0:y1, x0:x1]) / 3.0, 0, 1))
                edge_score = float(np.clip(np.mean(edge_mag[y0:y1, x0:x1]) / 32.0, 0, 1))
                raw_grid[r, c] = 0.55 * flow_score + 0.45 * edge_score

        self._smoothed_grid = (
            self.smoothing_alpha * raw_grid + (1 - self.smoothing_alpha) * self._smoothed_grid
        )

        if self.calibrating:
            self._baseline_grid = (
                self._baseline_grid * self._calib_count + self._smoothed_grid
            ) / (self._calib_count + 1)
            self._calib_count += 1
            if self._calib_count >= self.CALIBRATION_FRAMES:
                self.calibrating = False
            return [None] * self.num_zones, [], self.calibrating

        threshold = self._delta_threshold()
        # smoothed delta drives the reported distance (stable, low-jitter);
        # raw delta drives the persistence streak below — using the smoothed
        # value there would let one bright/noisy frame "fake" several frames
        # of apparent presence as the EMA decays back down.
        delta_grid = self._smoothed_grid - self._baseline_grid
        raw_delta_grid = raw_grid - self._baseline_grid
        headroom_grid = np.clip(1.0 - self._baseline_grid, 1e-3, None)

        # bottom-frame exclusion: zero out delta in the excluded rows so
        # hands/keyboard near a laptop webcam can't trigger detections
        excluded_rows = int(np.ceil(self.GRID_ROWS * self.ignore_bottom_fraction))
        if excluded_rows > 0:
            delta_grid[self.GRID_ROWS - excluded_rows:, :] = -1.0  # force below any threshold
            raw_delta_grid[self.GRID_ROWS - excluded_rows:, :] = -1.0

        # persistence: a cell only counts once its RAW (not smoothed) score
        # has been above threshold for PERSISTENCE_FRAMES consecutive frames
        above = raw_delta_grid > threshold
        self._streak = np.where(above, self._streak + 1, 0)
        persistent = self._streak >= self.PERSISTENCE_FRAMES
        # zero out delta for non-persistent cells so both zone and box
        # logic below naturally ignore them
        delta_grid = np.where(persistent, delta_grid, -1.0)

        # ---------------- per-zone aggregation (drives audio) ----------------
        zones = []
        for i in range(self.num_zones):
            c0 = int(np.floor(i * self.GRID_COLS / self.num_zones))
            c1 = int(np.floor((i + 1) * self.GRID_COLS / self.num_zones))
            c1 = max(c1, c0 + 1)
            zone_cells = delta_grid[:, c0:c1]
            zone_persistent = persistent[:, c0:c1]
            if not np.any(zone_persistent):
                zones.append(None)
                continue
            # average only over the persistent cells so a couple of
            # flickering cells don't dilute a real close reading
            zone_delta = float(np.mean(zone_cells[zone_persistent]))
            zone_headroom = float(np.mean(headroom_grid[:, c0:c1]))

            if zone_delta < threshold:
                zones.append(None)
                continue

            effective = float(np.clip(zone_delta / zone_headroom, 0, 1))
            dist = float(np.clip((1.0 - effective) * self.max_range_m, 0.3, self.max_range_m))
            x0px, x1px = c0 * cell_w, (w if c1 >= self.GRID_COLS else c1 * cell_w)
            height = self._height_from_edges(edge_mag, 0, h, x0px, x1px, h)
            zones.append({"dist": round(dist, 2), "height": height})

        # ---------------- bounding boxes (drives visual overlay) ----------------
        # delta_map / headroom_map are the shared distance-estimation surface —
        # used whether boxes come from real AI detection or classical contours
        delta_map = cv2.resize(delta_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        delta_map = np.clip(delta_map, 0, None)
        headroom_map = cv2.resize(headroom_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        use_ai_this_frame = self.use_ai and object_detector.AI_AVAILABLE
        if use_ai_this_frame:
            boxes = self._boxes_from_ai(frame, edge_mag, delta_map, headroom_map, h, w)
        else:
            boxes = self._boxes_from_contours(delta_grid, threshold, edge_mag, delta_map, headroom_map, h, w)

        boxes = self.tracker.update(boxes)

        return zones, boxes, self.calibrating

    def _boxes_from_contours(self, delta_grid, threshold, edge_mag, delta_map, headroom_map, h, w):
        """Classical path: contour detection on the calibrated closeness map.
        No labels, but needs nothing but OpenCV — always available."""
        mask = (delta_grid > threshold).astype(np.uint8) * 255
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((5, 5), np.uint8)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.006 * w * h  # ignore specks smaller than ~0.6% of frame
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            region_delta = float(np.mean(delta_map[y:y + bh, x:x + bw]))
            region_headroom = float(np.mean(headroom_map[y:y + bh, x:x + bw]))
            effective = float(np.clip(region_delta / max(region_headroom, 1e-3), 0, 1))
            dist = float(np.clip((1.0 - effective) * self.max_range_m, 0.3, self.max_range_m))
            height = self._height_from_edges(edge_mag, y, y + bh, x, x + bw, h)
            candidates.append({
                "x": x / w, "y": y / h, "w": bw / w, "h": bh / h,
                "dist": round(dist, 2),
                "height": height,
                "severity": self._severity(dist),
                "label": None,
                "_area": area,
            })

        candidates.sort(key=lambda b: b["_area"], reverse=True)
        boxes = candidates[:self.MAX_BOXES]
        for b in boxes:
            b.pop("_area", None)
        return boxes

    def _boxes_from_ai(self, frame_bgr, edge_mag, delta_map, headroom_map, h, w):
        """AI path: real object detection (see object_detector.py) supplies
        WHAT and WHERE with a real trained model; distance still comes from
        this module's own calibrated closeness map at that region, so the
        tested distance/tracking pipeline is unchanged."""
        detections = object_detector.detect_objects(frame_bgr)
        excluded_from_row = h * (1.0 - self.ignore_bottom_fraction)

        candidates = []
        for det in detections:
            x0, y0 = max(0, int(det["x0"])), max(0, int(det["y0"]))
            x1, y1 = min(w, int(det["x1"])), min(h, int(det["y1"]))
            bw, bh = x1 - x0, y1 - y0
            if bw <= 2 or bh <= 2:
                continue
            cy = (y0 + y1) / 2
            if cy >= excluded_from_row:
                continue  # centered in the excluded hand/keyboard region

            region_delta = float(np.mean(delta_map[y0:y1, x0:x1]))
            region_headroom = float(np.mean(headroom_map[y0:y1, x0:x1]))
            effective = float(np.clip(region_delta / max(region_headroom, 1e-3), 0, 1))
            dist = float(np.clip((1.0 - effective) * self.max_range_m, 0.3, self.max_range_m))
            height = self._height_from_edges(edge_mag, y0, y1, x0, x1, h)
            candidates.append({
                "x": x0 / w, "y": y0 / h, "w": bw / w, "h": bh / h,
                "dist": round(dist, 2),
                "height": height,
                "severity": self._severity(dist),
                "label": det["label"],
                "confidence": det["confidence"],
                "_conf_sort": det["confidence"],
            })

        candidates.sort(key=lambda b: b["_conf_sort"], reverse=True)
        boxes = candidates[:self.MAX_BOXES]
        for b in boxes:
            b.pop("_conf_sort", None)
        return boxes
