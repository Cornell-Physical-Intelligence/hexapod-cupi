"""CPU-only tests for the Phase 2 timed showcase helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_ROOT = (
    Path(__file__).parents[2] / "packages" / "hexapod_env" / "hexapod_env"
)
MODULE_PATH = PACKAGE_ROOT / "showcase_sequence.py"
SPEC = importlib.util.spec_from_file_location("hexapod_showcase_sequence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
showcase = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = showcase
SPEC.loader.exec_module(showcase)


def _passing_row(
    *, command: tuple[float, float, float]
) -> dict[str, object]:
    vx, vy, yaw = command
    return {
        "mean_command_frame_linear_velocity_mps": [vx, vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, yaw],
        "planar_velocity_rmse_mps": 0.03,
        "yaw_rate_rmse_radps": 0.03,
        "falls": 0,
        "maximum_tilt_degrees": 5.0,
        "torque": {
            "max_per_joint_rms_applied_nm": 1.2,
            "computed_demand_over_rating_fraction": 0.10,
            "maximum_computed_over_rating_burst_s": 0.10,
            "peak_abs_computed_nm": 4.0,
        },
    }


class ShowcaseSequenceTest(unittest.TestCase):
    def test_default_sequence_is_complete_and_uses_navigation_signs(self):
        segments = showcase.default_showcase_segments()
        self.assertEqual(
            [segment.key for segment in segments],
            [
                "stand",
                "forward",
                "strafe_left",
                "strafe_right",
                "turn_left",
                "turn_right",
                "diagonal_forward_left",
                "backward",
            ],
        )
        self.assertEqual(segments[1].command, (0.30, 0.0, 0.0))
        self.assertGreater(segments[2].command[1], 0.0)
        self.assertLess(segments[3].command[1], 0.0)
        self.assertGreater(segments[4].command[2], 0.0)
        self.assertLess(segments[5].command[2], 0.0)
        self.assertLess(segments[-1].command[0], 0.0)

    def test_schedule_has_no_gaps_and_preserves_measured_samples(self):
        schedule = showcase.schedule_segments(
            showcase.default_showcase_segments(),
            policy_dt_s=0.02,
            settle_seconds=0.5,
        )
        self.assertEqual(schedule[0].start_step, 0)
        self.assertEqual(schedule[0].steps, 100)
        self.assertEqual(schedule[1].steps, 150)
        self.assertEqual(schedule[-1].stop_step, 1150)
        self.assertTrue(all(segment.settle_steps == 25 for segment in schedule))
        self.assertTrue(all(segment.measured_steps > 0 for segment in schedule))
        for previous, following in zip(schedule, schedule[1:]):
            self.assertEqual(previous.stop_step, following.start_step)

    def test_schedule_rejects_a_settle_window_that_consumes_segment(self):
        segments = (
            showcase.CommandSegment("short", "SHORT", (0.0, 0.0, 0.0), 0.1),
        )
        with self.assertRaisesRegex(ValueError, "no measured samples"):
            showcase.schedule_segments(
                segments, policy_dt_s=0.02, settle_seconds=0.1
            )

    def test_acceptance_checks_tracking_sign_and_hardware_load(self):
        command = (0.30, 0.0, 0.0)
        accepted = showcase.evaluate_segment_acceptance(
            _passing_row(command=command), command
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["rejection_reasons"], [])

        wrong_direction = _passing_row(command=command)
        wrong_direction["mean_command_frame_linear_velocity_mps"] = [-0.20, 0.0, 0.0]
        wrong_direction["torque"]["peak_abs_computed_nm"] = 6.0
        rejected = showcase.evaluate_segment_acceptance(wrong_direction, command)
        self.assertFalse(rejected["accepted"])
        self.assertTrue(
            any("vx sign is wrong" in reason for reason in rejected["rejection_reasons"])
        )
        self.assertTrue(
            any(
                "peak absolute computed torque" in reason
                for reason in rejected["rejection_reasons"]
            )
        )

    def test_overlay_returns_an_annotated_copy(self):
        source = np.full((720, 1280, 3), 127, dtype=np.uint8)
        annotated = showcase.annotate_showcase_frame(
            source,
            label="DIAGONAL FORWARD LEFT",
            command=(0.25, 0.12, 0.0),
            segment_index=6,
            segment_count=8,
            progress=0.5,
        )
        self.assertEqual(annotated.shape, source.shape)
        self.assertEqual(annotated.dtype, source.dtype)
        self.assertFalse(np.shares_memory(annotated, source))
        self.assertTrue(np.all(source == 127))
        self.assertGreater(np.count_nonzero(annotated != source), 10_000)

        rgba = np.full((360, 640, 4), 255, dtype=np.uint8)
        rgba[:, :, 3] = 73
        rgba_annotated = showcase.annotate_showcase_frame(
            rgba,
            label="TURN LEFT",
            command=(0.0, 0.0, 0.35),
            segment_index=4,
            segment_count=8,
            progress=0.25,
        )
        self.assertTrue(np.all(rgba_annotated[:, :, 3] == 73))


if __name__ == "__main__":
    unittest.main()
