"""The requirements judge measures drift and stopping distance from a control trace."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.evaluation import requirements as judge

DT = judge.CONTROL_DT_S


def trace_from_path(path, commands):
    """A control trace whose root pose follows ``path`` ``(x, y, yaw)`` exactly."""
    n = len(commands)
    pose = np.zeros((n, 1, 7), dtype=np.float32)
    pose[:, 0, 0] = path[:n, 0]
    pose[:, 0, 1] = path[:n, 1]
    pose[:, 0, 5] = np.sin(path[:n, 2] / 2)
    pose[:, 0, 6] = np.cos(path[:n, 2] / 2)
    return {"root_pose_xyzw": pose, "command": np.asarray(commands, dtype=np.float32)[:, None, :]}


class GhostPathTests(unittest.TestCase):
    def test_forward_command_moves_along_body_minus_y_rotated_by_yaw(self):
        start = (1.0, 2.0, math.pi / 2)
        path = judge.ghost_path(start, [[0.05, 0.0, 0.0]] * 100)
        # Navigation forward is body -y; at yaw pi/2 body -y points along world +x.
        self.assertAlmostEqual(path[-1, 0], 1.0 + 0.05 * 2.0, places=9)
        self.assertAlmostEqual(path[-1, 1], 2.0, places=9)

    def test_arc_command_traces_a_circle_of_radius_speed_over_rate(self):
        v, w = 0.04, 0.15
        controls = int(round(2 * math.pi / w / DT))
        path = judge.ghost_path((0.0, 0.0, 0.0), [[v, 0.0, w]] * controls)
        self.assertLess(np.hypot(*path[-1, :2]), 1e-3)
        radius = np.hypot(*(path[:, :2] - path[:, :2].mean(0)).T)
        self.assertAlmostEqual(radius.mean(), v / w, places=3)


class JudgeTests(unittest.TestCase):
    def test_perfect_tracking_has_zero_drift_and_zero_stopping_distance(self):
        commands = [[0.05, 0.0, 0.0]] * 1000 + [[0.0, 0.0, 0.0]] * 200
        path = judge.ghost_path((0.0, 0.0, 0.3), commands)
        trace = trace_from_path(path, commands)
        holds = judge.drift_per_hold(trace)
        self.assertEqual(len(holds), 1)
        self.assertLess(holds[0]["max_gap_m"], 1e-6)
        stop = judge.stopping_distance(trace, settle_s=2.0)
        self.assertEqual(stop["stop_control"], 1000)
        self.assertLess(stop["final_distance_m"], 1e-9)

    def test_a_sideways_slip_shows_as_drift_and_travel_after_the_stop_as_stopping_distance(self):
        commands = [[0.05, 0.0, 0.0]] * 1000 + [[0.0, 0.0, 0.0]] * 200
        path = judge.ghost_path((0.0, 0.0, 0.0), commands)
        path[:, 1] += np.arange(len(path)) * 0.0001  # 0.1 mm of left slip per control
        path[1001:, 0] -= 0.03  # the body moves 3 cm after the stop
        trace = trace_from_path(path, commands)
        hold = judge.drift_per_hold(trace)[0]
        self.assertAlmostEqual(hold["end_gap_m"], 999 * 0.0001, places=6)
        self.assertAlmostEqual(hold["max_gap_m"], hold["end_gap_m"], places=9)
        stop = judge.stopping_distance(trace, settle_s=2.0)
        self.assertAlmostEqual(stop["final_distance_m"], math.hypot(0.03, 199 * 0.0001), places=6)

    def test_judgement_binds_the_requirements_hash_and_fails_on_the_drift_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commands = [[0.05, 0.0, 0.0]] * 1000
            path = judge.ghost_path((0.0, 0.0, 0.0), commands)
            path[:, 1] += np.arange(len(path)) * 0.0002  # 200 mm at the end
            trace = trace_from_path(path, commands)
            batch = root / "batch_000"
            batch.mkdir()
            np.savez(batch / "control_trace.npz", **trace)
            (batch / "report.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "c" * 64,
                        "cases": [
                            {"case_id": "learning:translate_0.05_0deg", "command": [0.05, 0, 0]}
                        ],
                    }
                )
            )
            result = judge.judge(root)
            self.assertEqual(result["checkpoint_sha256"], ["c" * 64])
            self.assertEqual(len(result["requirements_sha256"]), 64)
            self.assertFalse(result["pass"])
            check = result["cases"][0]["checks"]["drift_per_hold_m"]
            self.assertEqual(check["bound"], 0.10)
            self.assertGreater(check["value"], 0.10)

    def test_diagonal_walks_count_as_drift_cases(self):
        rules, _ = judge.load_requirements()
        speed = rules["controller"]["drift_speed_mps"]
        diagonal = [speed * math.cos(math.pi / 4), speed * math.sin(math.pi / 4), 0.0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commands = [diagonal] * 1000
            trace = trace_from_path(judge.ghost_path((0.0, 0.0, 0.0), commands), commands)
            batch = root / "batch_000"
            batch.mkdir()
            np.savez(batch / "control_trace.npz", **trace)
            (batch / "report.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "c" * 64,
                        "cases": [{"case_id": "d", "command": diagonal}],
                    }
                )
            )
            self.assertIn("drift_per_hold_m", judge.judge(root)["cases"][0]["checks"])
