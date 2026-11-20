"""The leg-stand height tracker: orientation lock, calibration capture and reference input."""

import argparse
import importlib.util
import math
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tag_height", ROOT / "testing/leg_stand/tag_height.py"
)
th = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(th)

BLACK = np.zeros((720, 1280, 3), np.uint8)


def plate_pose(height, tilt_deg=0.5, yaw_deg=2.0, offset=(1.0, -2.0)):
    """The pose that render_plate draws. The camera looks down, so the angle is near 180 degrees."""
    a, y = math.radians(tilt_deg), math.radians(yaw_deg)
    Rt = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    R = Rt @ np.diag([1.0, -1.0, -1.0]) @ Ry
    return R, -R @ np.array([offset[0], offset[1], height])


def corners(height, rng, noise_px=0.15):
    """Both tags' corner pixels at this height through the fisheye model. The default noise gives
    the 0.2 mm sd that the README reports at 255 mm."""
    R, t = plate_pose(height)
    out = {}
    for i in th.TAGS:
        obj = th.CORNERS[i].reshape(-1, 1, 3)
        px = cv2.fisheye.projectPoints(obj, cv2.Rodrigues(R)[0], t.reshape(3, 1), th.K, th.D)[0]
        out[i] = px.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
    return out


def angle_deg(A, B):
    return math.degrees(np.linalg.norm(cv2.Rodrigues(A @ B.T)[0]))


class TrackerCase(unittest.TestCase):
    """A tracker whose state files live in a temporary directory and whose clock reads 100 s."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, th, "LOCK", th.LOCK)
        self.addCleanup(setattr, th, "CAL", th.CAL)
        th.LOCK, th.CAL = Path(tmp.name) / "lock.json", Path(tmp.name) / "calibration.json"
        self.tr = th.Tracker(
            argparse.Namespace(camera=0, fps=60, avg=1.0, lock=60, port=0, sim=True, steady=0.5)
        )
        self.tr.t0 = time.monotonic() - 100

    def feed(self, dets):
        self.tr.det = lambda frame: dets
        self.tr.process(BLACK, time.monotonic())
        return self.tr.sample

    def lock(self):
        self.tr.R_lock = plate_pose(300)[0]

    def hold(self, *spans):
        """Two-tag frames at 60 fps that end now: (height, seconds) spans, oldest first."""
        self.tr.hist.clear()
        t = 100 - sum(seconds for _, seconds in spans)
        for height, seconds in spans:
            for k in range(round(seconds * 60)):
                self.tr.hist.append((t + k / 60, float(height), True))
            t += seconds


class OrientationLockTests(TrackerCase):
    def test_mean_rotation_of_tilts_either_side_of_180_degrees(self):
        tilts = (-1.0, -0.5, 0.5, 1.0)
        mean = th.mean_rotation([plate_pose(300, tilt_deg=t)[0] for t in tilts])
        self.assertLess(angle_deg(mean, plate_pose(300, tilt_deg=0.0)[0]), 0.01)

    def test_lock_from_noisy_frames_gives_the_height(self):
        """One frame is about 3 degrees off at this noise, so the mean of 60 is under 1 degree.
        Exact corners after the lock show the height error that the lock alone causes."""
        for seed in range(10):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                self.tr.relock = True
                for _ in range(60):
                    self.feed(corners(380, rng))
                self.assertIsNotNone(self.tr.R_lock)
                self.assertLess(angle_deg(self.tr.R_lock, plate_pose(380)[0]), 2.0)
                for height in (300, 380, 460):
                    exact = self.feed(corners(height, rng, noise_px=0.0))
                    self.assertAlmostEqual(exact["h"], height, delta=0.5)


class CalibrationCaptureTests(TrackerCase):
    def test_zero_is_the_mean_of_a_steady_second(self):
        self.lock()
        self.hold((300, 1.0))
        self.tr.action("/zero", {})
        self.assertAlmostEqual(self.tr.cal.h0, 300.0)

    def test_zero_before_the_lock_is_refused(self):
        self.hold((300, 1.0))
        with self.assertRaises(ValueError):
            self.tr.action("/zero", {})
        self.assertIsNone(self.tr.cal.h0)

    def test_frames_from_before_the_lock_are_not_averaged(self):
        self.lock()
        self.tr.lock_t = 99.7
        self.hold((300, 1.0))
        with self.assertRaises(ValueError):
            self.tr.action("/zero", {})

    def test_zero_soon_after_a_move_is_refused(self):
        self.lock()
        self.hold((300, 0.7), (310, 0.3))
        with self.assertRaises(ValueError):
            self.tr.action("/zero", {})
        self.assertIsNone(self.tr.cal.h0)

    def test_zero_from_a_part_of_the_window_is_refused(self):
        self.lock()
        self.hold((300, 0.3))
        with self.assertRaises(ValueError):
            self.tr.action("/zero", {})


class ReferenceInputTests(TrackerCase):
    def setUp(self):
        super().setUp()
        self.lock()
        self.hold((300, 1.0))
        self.tr.action("/zero", {})

    def test_point_sets_the_scale(self):
        self.hold((250, 1.0))
        self.tr.action("/point", {"ref": ["51"]})
        self.assertAlmostEqual(self.tr.cal.a, -1.02)

    def test_reference_that_is_not_a_finite_number_is_refused(self):
        self.hold((250, 1.0))
        for ref in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                self.tr.action("/point", {"ref": [ref]})
        self.assertEqual(self.tr.cal.points, [])
        self.assertNotIn("NaN", th.CAL.read_text())

    def test_point_at_the_zero_is_refused(self):
        self.hold((300.01, 1.0))
        with self.assertRaises(ValueError):
            self.tr.action("/point", {"ref": ["0"]})
        self.assertEqual(self.tr.cal.a, -1.0)

    def test_zero_reference_away_from_the_zero_is_refused(self):
        self.hold((250, 1.0))
        with self.assertRaises(ValueError):
            self.tr.action("/point", {"ref": ["0"]})
        self.assertEqual(self.tr.cal.a, -1.0)


if __name__ == "__main__":
    unittest.main()
