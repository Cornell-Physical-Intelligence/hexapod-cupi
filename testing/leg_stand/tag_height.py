#!/usr/bin/env python3
"""Leg-stand carriage height from the AprilTag plate, seen by the ELP global-shutter camera.

  uv run python testing/leg_stand/tag_height.py serve      # live dashboard, http://localhost:8765
  uv run python testing/leg_stand/tag_height.py selftest   # simulated frames, must print PASS

README.md in this folder describes the setup, the method and the measured performance.
"""

import argparse
import collections
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LOCK, CAL = HERE / "lock.json", HERE / "calibration.json"

# ELP USBGS720P02 (AR0144, USB 32e4:0144), 1280x720 MJPG, cv2.fisheye model. One unit, calibrated
# 2026-11-01 with a 12 mm ChArUco; focal length agrees within 0.3 % across units, the principal
# point varies by tens of pixels, so the zero and reference points absorb the scale.
SIZE = (1280, 720)
K = np.array([[752.053404, 0.0, 619.258506], [0.0, 752.887674, 426.128195], [0.0, 0.0, 1.0]])
D = np.array([-0.023059609, 0.018462433, -0.024844754, 0.003574485]).reshape(4, 1)
CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)

# Plate frame (mm): origin midway between the tag centres on the top face, +x toward the rail,
# +y toward the top of the printed tags, +z toward the camera. tag36h11 ids 0 (leg side) and
# 1 (rail side); black squares 20.8 mm (8 bits x 2.6 mm), centres 35.55 mm apart.
TAGS = {0: (-17.775, 0.0), 1: (17.775, 0.0)}
HALF = 10.4
SQUARE = ((-1, 1), (1, 1), (1, -1), (-1, -1))  # cv2.aruco corner order: TL, TR, BR, BL
MIN_TRAVEL = 5.0  # mm from the zero before a reference point can set the scale
CORNERS = {
    i: np.array([[x + dx * HALF, y + dy * HALF, 0.0] for dx, dy in SQUARE])
    for i, (x, y) in TAGS.items()
}


def undistort(px, P=None):
    """Pixels -> normalised coordinates (or pixels of P) with the fisheye distortion removed."""
    p = np.ascontiguousarray(px, dtype=np.float64).reshape(-1, 1, 2)
    P = np.eye(3) if P is None else P
    return cv2.fisheye.undistortPoints(p, K, D, R=np.eye(3), P=P, criteria=CRIT).reshape(-1, 2)


class Detector:
    """tag36h11 via cv2.aruco with AprilTag corner refinement (least distance bias in the
    self-test). Thresholds are loosened for marker-inked tags on coloured filament, and each image
    is tried as max-of-channels (coloured filament turns light), luma and CLAHE."""

    VARIANTS = ("max", "luma", "clahe")

    def __init__(self):
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax = 3, 33
        p.adaptiveThreshWinSizeStep, p.adaptiveThreshConstant = 6, 4
        p.maxErroneousBitsInBorderRate = 0.5
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.det = cv2.aruco.ArucoDetector(dic, p)
        self.clahe = cv2.createCLAHE(3.0, (4, 4))
        self.variant, self.prev, self.n = "max", None, 0

    def _gray(self, img, name):
        if name == "max":
            return np.ascontiguousarray(img.max(axis=2))
        g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(g) if name == "clahe" else g

    def _find(self, img, x0, y0, names):
        """Union of the tags found over the variants, stopping once both are found."""
        out = {}
        for name in names if img.ndim == 3 else names[:1]:
            gray = self._gray(img, name) if img.ndim == 3 else np.ascontiguousarray(img)
            corners, ids, _ = self.det.detectMarkers(gray)
            found = {
                int(i): c.reshape(4, 2).astype(np.float64) + (x0, y0)
                for c, i in zip(corners, [] if ids is None else ids.ravel())
                if int(i) in TAGS
            }
            if len(found) == len(TAGS):
                self.variant = name
            for i, c in found.items():
                out.setdefault(i, c)
            if len(out) == len(TAGS):
                break
        return out

    def __call__(self, img):
        """Search a crop around the last position, sticky variant first. Fall back to the full
        frame (also every 15th partial frame), one variant per frame so the loop never stalls."""
        self.n += 1
        names = (self.variant, *(v for v in self.VARIANTS if v != self.variant))
        if self.prev:
            pts = np.vstack(list(self.prev.values()))
            m = max(60.0, 0.6 * float(np.ptp(pts, axis=0).max()))
            x0, y0 = np.maximum(pts.min(0) - m, 0).astype(int)
            x1, y1 = (pts.max(0) + m).astype(int)
            found = self._find(img[y0:y1, x0:x1], x0, y0, names)
            if len(found) == len(TAGS) or (found and self.n % 15):
                self.prev = found
                return found
        found = self._find(img, 0, 0, (self.VARIANTS[self.n % len(self.VARIANTS)],))
        self.prev = found or None
        return found


def solve(dets, R_lock=None):
    """Plate pose in the camera frame (X_cam = R X_plate + t). With R_lock only the translation is
    solved, by linear least squares: the rail only translates the plate, and a free 6-DOF pose
    drifts by about a millimetre at the far end of the stroke."""
    if not dets:
        return None
    ids = sorted(dets)
    obj = np.vstack([CORNERS[i] for i in ids])
    img = np.vstack([dets[i] for i in ids])
    nrm = undistort(img)
    if R_lock is None:
        ok, rvec, tvec = cv2.solvePnP(obj, nrm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        _, rvec, tvec = cv2.solvePnP(
            obj, nrm, np.eye(3), None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE
        )
        R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
    else:
        R, PX, (u, v) = R_lock, obj @ R_lock.T, nrm.T
        A, b = np.zeros((2 * len(obj), 3)), np.zeros(2 * len(obj))
        A[0::2, 0], A[0::2, 2], b[0::2] = 1, -u, u * PX[:, 2] - PX[:, 0]
        A[1::2, 1], A[1::2, 2], b[1::2] = 1, -v, v * PX[:, 2] - PX[:, 1]
        t = np.linalg.lstsq(A, b, rcond=None)[0]
    C = -R.T @ t  # lens centre in the plate frame (mm)
    if C[2] < 0:  # mirror solution
        return None
    proj = cv2.fisheye.projectPoints(
        obj.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, D
    )[0]
    return {
        "height_mm": float(C[2]),
        "tilt_deg": math.degrees(math.acos(min(1.0, max(-1.0, -R[2, 2])))),
        "reproj_px": float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1)))),
        "R": R,
        "ids": ids,
    }


def mean_rotation(Rs):
    """The rotation nearest the mean of the matrices. The plate faces the camera, so the angle is
    near 180 degrees, where a rotation vector flips sign from frame to frame and has no mean."""
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


class Calibration:
    """z = a (h - h0). Set zero stores h0; a is -1 (toward the camera is +) or +1 (flipped) until
    reference points exist, then the least-squares scale through the zero."""

    def __init__(self):
        c = json.loads(CAL.read_text()) if CAL.exists() else {}
        self.h0, self.sign, self.points = c.get("h0"), c.get("sign", -1.0), c.get("points", [])

    @property
    def a(self):
        if self.h0 is None or not self.points:
            return self.sign
        dh = np.array([p["height_mm"] - self.h0 for p in self.points])
        ref = np.array([p["ref_mm"] for p in self.points])
        return float(dh @ ref / (dh @ dh)) if dh @ dh > 1e-6 else self.sign

    def save(self):
        CAL.write_text(json.dumps({"h0": self.h0, "sign": self.sign, "points": self.points}))

    def summary(self):
        a, h0 = self.a, self.h0
        pts = [dict(p, z_mm=a * (p["height_mm"] - h0)) for p in self.points if h0 is not None]
        for p in pts:
            p["resid_mm"] = p["z_mm"] - p["ref_mm"]
        fitted = h0 is not None and bool(self.points)
        return {"h0": h0, "a": a, "points": pts, "n_ref": len(pts), "fitted": fitted}


class Tracker:
    """A capture thread keeps the newest frame; a processing thread measures it and wakes the
    /events and /stream.mjpg clients. After the orientation lock, a frame with one tag uses that
    tag alone, corrected by its offset from the two-tag height (learned on two-tag frames)."""

    def __init__(self, a):
        self.a, self.det, self.cal = a, Detector(), Calibration()
        self.cond, self.fcond = threading.Condition(), threading.Condition()
        self.raw = self.sample = self.jpeg = None
        self.raw_no = self.frame_no = 0
        self.hist = collections.deque(maxlen=8000)  # (t, height or None, both tags)
        self.cap_ts, self.proc_ts = collections.deque(maxlen=60), collections.deque(maxlen=60)
        self.seen = collections.deque(maxlen=60)
        self.status, self.error, self.camera, self.reopen = "starting", None, a.camera, True
        self.R_lock = np.array(json.loads(LOCK.read_text())["R"]) if LOCK.exists() else None
        self.lock_Rs, self.relock, self.lock_t = [], False, -math.inf
        self.off, self.off_n = dict.fromkeys(TAGS, 0.0), dict.fromkeys(TAGS, 0)
        self.t0 = time.monotonic()

    def _open(self):
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, self.a.fps)
        ok, f = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            cap.release()
            return None, f"camera {self.camera} gave no frame (macOS: allow Camera access)"
        if (f.shape[1], f.shape[0]) != SIZE:
            cap.release()
            return None, f"camera {self.camera} delivered {f.shape[1]}x{f.shape[0]}, not 1280x720"
        return cap, None

    def capture_loop(self):
        cap, rng = None, np.random.default_rng(0)
        while True:
            if self.a.sim:
                hgt = 380 + 80 * math.sin((time.monotonic() - self.t0) / 6)
                frame = cv2.cvtColor(render_plate(hgt, 0.5, (1, -2), 2, rng)[0], cv2.COLOR_GRAY2BGR)
                self.status = "simulated"
            else:
                if self.reopen:
                    if cap is not None:
                        cap.release()
                    self.status, self.reopen = "opening camera", False
                    cap, self.error = self._open()
                    if cap is None:
                        self.status, self.reopen = "no camera", True
                        time.sleep(2)
                        continue
                    self.status, self.det.prev = "live", None
                ok, frame = cap.read()
                if not ok:
                    self.error, self.reopen = "the camera stopped delivering frames", True
                    continue
            with self.fcond:
                self.raw, self.raw_t, self.raw_no = frame, time.monotonic(), self.raw_no + 1
                self.cap_ts.append(self.raw_t)
                self.fcond.notify_all()

    def process_loop(self):
        last = 0
        while True:
            with self.fcond:
                if not self.fcond.wait_for(lambda: self.raw_no != last, timeout=1):
                    continue
                frame, ts, last = self.raw, self.raw_t, self.raw_no
            self.process(frame, ts)

    def process(self, frame, ts):
        if self.relock:
            self.R_lock, self.lock_Rs, self.relock = None, [], False
        dets = self.det(frame)
        res = solve(dets, self.R_lock)
        both = bool(res) and len(res["ids"]) == len(TAGS)
        h = res["height_mm"] if both else None
        if both and self.R_lock is None and res["reproj_px"] < 2:
            self.lock_Rs.append(res["R"])
            if len(self.lock_Rs) >= self.a.lock:
                self.R_lock, self.lock_t = mean_rotation(self.lock_Rs), round(ts - self.t0, 4)
                LOCK.write_text(json.dumps({"R": self.R_lock.tolist()}))
        elif res and self.R_lock is not None:
            if both:  # learn each tag's own-height offset from the two-tag height
                for i in TAGS:
                    one = solve({i: dets[i]}, self.R_lock)
                    w = 0.1 if self.off_n[i] < 30 else 0.02
                    if one:
                        off = one["height_mm"] - h
                        self.off[i] = off if not self.off_n[i] else (1 - w) * self.off[i] + w * off
                        self.off_n[i] += 1
            elif self.off_n[res["ids"][0]] >= 10:
                h = res["height_mm"] - self.off[res["ids"][0]]
        s = 960 / frame.shape[1]
        view = cv2.resize(frame, (960, round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        for i, c in dets.items():
            cv2.polylines(view, [(c * s).round().astype(np.int32)], True, (60, 220, 60), 2)
            org = tuple((c.mean(0) * s).astype(int) + (-7, 8))
            cv2.putText(view, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 160, 255), 2)
        jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        sample = {
            "t": round(ts - self.t0, 4),
            "h": None if h is None else round(h, 4),
            "both": both,
            "ids": res["ids"] if res else [],
            "reproj": round(res["reproj_px"], 3) if res else None,
            "tilt": round(res["tilt_deg"], 3) if res else None,
            "lat": round((time.monotonic() - ts) * 1e3, 1),
        }
        with self.cond:
            self.hist.append((sample["t"], h, both))
            self.seen.append(sorted(dets))
            self.proc_ts.append(time.monotonic())
            self.sample, self.jpeg, self.frame_no = sample, jpeg, self.frame_no + 1
            self.cond.notify_all()

    def window(self, seconds):
        with self.cond:
            now = time.monotonic() - self.t0
            return [r for r in self.hist if r[0] >= now - seconds and r[1] is not None]

    def state(self):
        def rate(ts):
            return 1 / float(np.median(np.diff(list(ts)))) if len(ts) > 3 else None

        n = max(len(self.seen), 1)
        return {
            "status": self.status,
            "error": self.error,
            "camera": "sim" if self.a.sim else self.camera,
            "camera_fps": rate(self.cap_ts),
            "process_fps": rate(self.proc_ts),
            "locked": self.R_lock is not None,
            "lock_progress": len(self.lock_Rs),
            "lock_needed": self.a.lock,
            "variant": self.det.variant,
            "seen": {str(i): sum(i in s for s in self.seen) / n for i in TAGS},
            "cal": self.cal.summary(),
        }

    def average(self):
        """Mean of the frames with both tags over the last --avg seconds. The frames must follow
        the orientation lock, span the window and hold still: a zero or a point from other
        frames is wrong and nothing downstream can tell."""
        if self.R_lock is None:
            raise ValueError("wait for the orientation lock")
        rows = [r for r in self.window(self.a.avg) if r[2] and r[0] > self.lock_t]
        hs, third = np.array([r[1] for r in rows]), len(rows) // 3
        if not third or rows[-1][0] - rows[0][0] < 0.8 * self.a.avg:
            raise ValueError(f"hold both tags in view for {self.a.avg:g} s after the lock")
        moved = abs(float(hs[-third:].mean() - hs[:third].mean()))
        if moved > self.a.steady:
            raise ValueError(f"the height moved {moved:.2f} mm during the average; hold still")
        return {"height_mm": float(hs.mean()), "sd_mm": float(hs.std()), "n": len(hs)}

    def action(self, path, q):
        cal = self.cal
        if path == "/relock":
            self.relock = True
        elif path == "/camera":
            self.camera, self.reopen = int(q["index"][0]), True
        elif path == "/zero":
            cal.h0, cal.points = self.average()["height_mm"], []
        elif path == "/point":
            if cal.h0 is None:
                raise ValueError("set the zero first")
            avg, ref = self.average(), float(q["ref"][0])
            if not MIN_TRAVEL <= abs(ref) < math.inf:  # nan fails both comparisons
                raise ValueError(
                    f"the reference must be a number at least {MIN_TRAVEL:g} mm from 0"
                )
            if abs(avg["height_mm"] - cal.h0) < MIN_TRAVEL:
                raise ValueError(f"move at least {MIN_TRAVEL:g} mm from the zero first")
            cal.points.append(dict(avg, ref_mm=ref))
        elif path == "/delete":
            cal.points.pop(int(q["i"][0]))
        elif path == "/flip":
            cal.sign = -cal.sign
        elif path == "/clear":
            cal.h0, cal.points = None, []
        else:
            raise KeyError(path)
        cal.save()


def make_handler(tr):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, ctype, chunk):
            """Push one chunk per processed frame until the client goes away."""
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with tr.cond:
                        tr.cond.wait_for(lambda: tr.frame_no != last, timeout=1)
                        last = tr.frame_no
                    if tr.sample is not None:
                        self.wfile.write(chunk())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(json.dumps(tr.state()).encode(), "application/json")
            elif path == "/history":
                rows = [[round(t, 4), h, int(b)] for t, h, b in tr.window(60)]
                self._send(json.dumps(rows).encode(), "application/json")
            elif path == "/events":
                self._stream(
                    "text/event-stream", lambda: f"data: {json.dumps(tr.sample)}\n\n".encode()
                )
            elif path == "/stream.mjpg":
                self._stream(
                    "multipart/x-mixed-replace; boundary=frame",
                    lambda: b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + tr.jpeg + b"\r\n",
                )
            else:
                self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                tr.action(u.path, parse_qs(u.query))
                reply = {"ok": True}
            except KeyError:
                return self.send_error(404)
            except (ValueError, IndexError) as e:
                reply = {"error": str(e) or "invalid request"}
            self._send(json.dumps(reply).encode(), "application/json")

    return Handler


def cmd_serve(a):
    tr = Tracker(a)
    for loop in (tr.capture_loop, tr.process_loop):
        threading.Thread(target=loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(tr))
    server.daemon_threads = True
    print(f"dashboard on http://localhost:{a.port}", flush=True)
    server.serve_forever()


def render_plate(height, tilt_deg, offset, yaw_deg, rng):
    """Simulated distorted ELP frame of the plate from above (texture at 12 px/mm), with blur,
    noise and JPEG. It does not model the lens focus."""
    ppm, ext = 12, (-110.0, -70.0, 110.0, 70.0)
    W, H = int((ext[2] - ext[0]) * ppm), int((ext[3] - ext[1]) * ppm)
    noise = cv2.resize(rng.standard_normal((H // 24 + 1, W // 24 + 1)), (W, H))
    tex = np.clip(90 + 25 * noise, 0, 255).astype(np.uint8)

    def px(x, y):
        return round((x - ext[0]) * ppm), round((ext[3] - y) * ppm)

    cv2.rectangle(tex, px(-31.35, 17.0), px(31.35, -17.0), 228, -1)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side = round(2 * HALF * ppm)
    for i, (x, y) in TAGS.items():
        tag = cv2.resize(
            cv2.aruco.generateImageMarker(dic, i, 8, borderBits=1),
            (side, side),
            interpolation=cv2.INTER_NEAREST,
        )
        x0, y0 = px(x - HALF, y + HALF)
        tex[y0 : y0 + side, x0 : x0 + side] = np.where(tag > 127, 228, 30)
    a, y = math.radians(tilt_deg), math.radians(yaw_deg)
    Rt = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    R = Rt @ np.diag([1.0, -1.0, -1.0]) @ Ry  # looking down, image down = plate -y
    t = -R @ np.array([offset[0], offset[1], height])
    Ks = K.copy()  # render 2x supersampled, then area-downsample
    Ks[:2] *= 2
    Ks[:2, 2] += 0.5
    Atex = np.array([[1 / ppm, 0, ext[0]], [0, -1 / ppm, ext[3]], [0, 0, 1.0]])
    big = cv2.warpPerspective(
        tex, Ks @ np.c_[R[:, 0], R[:, 1], t] @ Atex, (SIZE[0] * 2, SIZE[1] * 2), borderValue=90
    )
    und = cv2.resize(big, SIZE, interpolation=cv2.INTER_AREA)
    gx, gy = np.meshgrid(np.arange(SIZE[0], dtype=np.float32), np.arange(SIZE[1], dtype=np.float32))
    u = undistort(np.stack([gx, gy], -1), P=K).reshape(SIZE[1], SIZE[0], 2).astype(np.float32)
    img = cv2.remap(und, u[..., 0], u[..., 1], cv2.INTER_LINEAR, borderValue=90)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    img = np.clip(img + 2.5 * rng.standard_normal(img.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 0), R


def cmd_selftest(a):
    rng, det, worst, failed = np.random.default_rng(1), Detector(), 0.0, False
    print(f"simulated ELP 1280x720 frames, {a.trials} per height; height error in mm")
    print("  height | both tags | free 6-DOF    | locked orientation")
    for hgt in (60, 120, 200, 300, 400, 470):
        e6, el = [], []
        for _ in range(a.trials):
            img, R = render_plate(
                hgt, rng.uniform(0, 1.5), rng.uniform(-5, 5, 2), rng.uniform(-5, 5), rng
            )
            det.prev = None
            d = det(img)
            if len(d) == len(TAGS):
                e6.append(solve(d)["height_mm"] - hgt)
                el.append(solve(d, R)["height_mm"] - hgt)
        failed |= len(el) < a.trials
        worst = max([worst, *np.abs(el)])
        print(
            f"  {hgt:6d} | {len(el):3d}/{a.trials:<5d} | {np.mean(e6):+.3f} ± {np.std(e6):.3f} | "
            f"{np.mean(el):+.3f} ± {np.std(el):.3f}"
            if el
            else f"  {hgt:6d} | no detections"
        )
    ok = not failed and worst < 2.0
    print("PASS" if ok else "FAIL", f"(worst locked error {worst:.2f} mm, limit 2 mm, both tags)")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="live dashboard")
    s.set_defaults(f=cmd_serve)
    s.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    s.add_argument("--fps", type=int, default=60, help="frame rate requested from the camera")
    s.add_argument("--avg", type=float, default=1.0, help="seconds averaged per zero or point")
    s.add_argument(
        "--steady", type=float, default=0.5, help="mm the average may move, start to end"
    )
    s.add_argument("--lock", type=int, default=60, help="two-tag frames for the orientation lock")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--sim", action="store_true", help="simulated frames instead of the camera")
    t = sub.add_parser("selftest", help="check the maths on simulated frames")
    t.set_defaults(f=cmd_selftest)
    t.add_argument("--trials", type=int, default=5)
    a = p.parse_args()
    raise SystemExit(a.f(a))


if __name__ == "__main__":
    main()
