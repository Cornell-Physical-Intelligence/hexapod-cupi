"""Coverage and failure-oriented tests for isolated Stage 2 review modules."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from experiments.c_length_study.tools.omni_quiet_review import quiet_metrics
from experiments.c_length_study.tools.omni_visual_review import visual_review_specs, review_command, review_reference


class ReviewContracts(unittest.TestCase):
    def quiet_data(self):
        data = {key: np.zeros((40, 1, 2)) for key in ("joint_position_rad", "joint_target_rad", "joint_velocity_rad_s")}
        data["position_world_m"] = np.zeros((40, 1, 3))
        data["quaternion_world_wxyz"] = np.zeros((40, 1, 4))
        data["quaternion_world_wxyz"][..., 0] = 1
        data["computed_torque_nm"] = np.full((40, 1, 2), .5)
        data["applied_torque_nm"] = np.full((40, 1, 2), .5)
        data["terminated"] = np.zeros((40, 1), dtype=bool)
        data["truncated"] = np.zeros((40, 1), dtype=bool)
        return data

    def test_quiet_stand_keeps_failures_before_scored_window(self):
        data = self.quiet_data()
        self.assertTrue(quiet_metrics(data, 0, 10, ["right_knee", "left_hip"], .02)["pass"])
        data["terminated"][2, 0] = True
        result = quiet_metrics(data, 0, 10, ["right_knee", "left_hip"], .02)
        self.assertFalse(result["pass"])
        self.assertEqual(result["terminations"], 1)

    def test_quiet_stand_rejects_drift_motor_motion_and_saturation(self):
        for key, value in (("joint_velocity_rad_s", .1), ("computed_torque_nm", 2.)):
            data = self.quiet_data()
            data[key][:, 0, 0] = value
            self.assertFalse(quiet_metrics(data, 0, 0, ["a", "b"], .02)["pass"])
        data = self.quiet_data()
        data["position_world_m"][:, 0, 0] = np.linspace(0, .02, 40)
        result = quiet_metrics(data, 0, 0, ["a", "b"], .02)
        self.assertIn("max_planar_excursion_m", result["failed_bounds"])
        data = self.quiet_data()
        data["joint_position_rad"][5, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            quiet_metrics(data, 0, 0, ["a", "b"], .02)

    def test_video_covers_all_bearings_two_speeds_turns_and_quiet(self):
        specs = visual_review_specs()
        self.assertEqual(len({spec["name"] for spec in specs}), len(specs))
        translations = [spec for spec in specs if spec["name"].startswith("bearing_")]
        self.assertEqual(len(translations), 32)
        for speed in (.10, .20):
            selected = [spec for spec in translations if abs(math.hypot(*spec["command"][:2])-speed) < 1e-6]
            bearings = sorted(round(math.atan2(spec["command"][1], spec["command"][0]) % (2*math.pi), 5) for spec in selected)
            self.assertEqual(len(set(bearings)), 16)
        self.assertEqual({spec["command"][2] for spec in specs if spec["name"].startswith("turn_")}, {-.4, -.2, .2, .4})
        quiet = next(spec for spec in specs if spec["name"] == "quiet_stand")
        self.assertEqual(quiet["controller"], "command")
        self.assertTrue(all(review_command(quiet, t) == [0., 0., 0.] for t in (0, 1, 10, 29, 31)))

    def test_review_paths_stay_finite_with_independent_heading(self):
        for spec in visual_review_specs():
            poses, commands, requests = review_reference(spec, .02, torch.zeros(1, 3))
            self.assertTrue(torch.isfinite(poses).all())
            self.assertEqual(poses.shape, commands.shape)
            self.assertTrue(torch.allclose(requests[-1], torch.zeros(3)))
            if spec["name"] == "fixed_heading_curve":
                self.assertTrue(torch.allclose(poses[:, 2], torch.zeros(len(poses))))
                self.assertGreater(float(poses[-1, 0]), .2)
                self.assertGreater(float(poses[-1, 1]), .2)


if __name__ == "__main__":
    unittest.main()
