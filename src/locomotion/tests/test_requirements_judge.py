"""The requirements judge: the cases of a set and the four checks it measures from a control trace."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.evaluation import requirements as judge
from locomotion.omniwalk.mdp.commands import TaskConfig, command_bank

DT = judge.CONTROL_DT_S
# c48_long's gait clock is 60 controls, so one stride lasts 1.2 s and a leg swings half a stride.
GAIT_PERIOD_S = 1.2
# Forward reach of a corner leg at its joint limits from the declared stance (7 November 2026 audit
# of the 96 historical cases; the robot model is robot/active_model.json).
REACH_M = 0.0825


def trace_from_path(path, commands, torque=0.0, tilt_rad=0.0):
    """A control trace whose root pose follows ``path`` ``(x, y, yaw)`` exactly."""
    n = len(commands)
    pose = np.zeros((n, 1, 7), dtype=np.float32)
    pose[:, 0, 0] = path[:n, 0]
    pose[:, 0, 1] = path[:n, 1]
    # Yaw about z, then tilt about the body x axis.
    pose[:, 0, 3] = np.sin(tilt_rad / 2) * np.cos(path[:n, 2] / 2)
    pose[:, 0, 4] = np.sin(tilt_rad / 2) * np.sin(path[:n, 2] / 2)
    pose[:, 0, 5] = np.cos(tilt_rad / 2) * np.sin(path[:n, 2] / 2)
    pose[:, 0, 6] = np.cos(tilt_rad / 2) * np.cos(path[:n, 2] / 2)
    return {
        "root_pose_xyzw": pose,
        "command": np.asarray(commands, dtype=np.float32)[:, None, :],
        "applied_torque_abs_max_400hz": np.full((n, 1), torque, dtype=np.float32),
    }


def save_batch(root, index, case, trace):
    batch = root / f"batch_{index:03d}"
    batch.mkdir()
    np.savez(batch / "control_trace.npz", **trace)
    (batch / "report.json").write_text(json.dumps({"checkpoint_sha256": "c" * 64, "cases": [case]}))


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.rules, _ = judge.load_requirements()
        self.cases = judge.cases(self.rules)
        self.fixed = [case for case in self.cases if "command" in case]

    def test_the_set_enumerates_the_stand_every_walk_both_turns_both_arcs_and_the_routes(self):
        ids = [case["case_id"] for case in self.cases]
        routes = self.rules["stop"]["routes_m"]
        self.assertEqual(len(ids), 1 + 2 * 8 + 2 + 2 + len(routes))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids[0], "stand")
        self.assertIn("walk_0.025_135deg", ids)
        self.assertIn("walk_0.050_0deg", ids)
        self.assertEqual(ids[20:21], ["arc_+0.15"])
        self.assertEqual(ids[21:], ["route_" + name for name in routes])
        for case in self.fixed:
            self.assertEqual(case["controls"], 1200)
            self.assertEqual(case["stop_control"], 1000)
        # Every route fits the 60 s native episode at the slowest speed plus the stay and two settles.
        for case in self.cases[21:]:
            self.assertLessEqual(case["controls"], 3000)

    def test_the_moving_cases_are_the_training_command_bank(self):
        moving = [case["command"] for case in self.fixed if any(case["command"])]
        self.assertEqual(moving, command_bank(TaskConfig()))

    def test_every_case_is_reachable_from_the_declared_stance(self):
        for case in self.fixed:
            speed = math.hypot(*case["command"][:2])
            with self.subTest(case=case["case_id"]):
                self.assertLessEqual(speed * GAIT_PERIOD_S / 2, REACH_M)

    def test_command_at_holds_then_commands_zero_and_rejects_outside_controls(self):
        case = self.cases[1]
        self.assertEqual(judge.command_at(case, 0), case["command"])
        self.assertEqual(judge.command_at(case, 999), case["command"])
        self.assertEqual(judge.command_at(case, 1000), [0.0, 0.0, 0.0])
        for control in (-1, 1200):
            with self.assertRaises(ValueError):
                judge.command_at(case, control)
        without_stop = {"case_id": "x", "command": [0.05, 0.0, 0.0], "controls": 4}
        self.assertEqual(judge.command_at({"stop_control": 4, **without_stop}, 3), [0.05, 0.0, 0.0])


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
    def setUp(self):
        self.rules, _ = judge.load_requirements()

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
        result = judge.judge_case(trace, self.rules, 1200)
        self.assertTrue(result["pass"])
        self.assertEqual(
            set(result["checks"]),
            {"drift_per_hold_m", "stopping_distance_m", "tilt_deg", "applied_torque_nm"},
        )

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

    def test_tilt_and_torque_are_judged_against_the_file_limits(self):
        commands = [[0.0, 0.0, 0.0]] * 50
        path = judge.ghost_path((0.0, 0.0, 0.0), commands)
        level = judge.judge_case(trace_from_path(path, commands), self.rules, 50)
        self.assertTrue(level["pass"])
        self.assertAlmostEqual(level["checks"]["tilt_deg"]["value"], 0.0, places=9)
        tilted = judge.judge_case(
            trace_from_path(path, commands, tilt_rad=math.radians(3.0)), self.rules, 50
        )
        self.assertAlmostEqual(tilted["checks"]["tilt_deg"]["value"], 3.0, places=5)
        self.assertEqual(tilted["checks"]["tilt_deg"]["bound"], self.rules["tilt_max_deg"])
        self.assertEqual(tilted["failed_checks"], ["tilt_deg"])
        strained = judge.judge_case(trace_from_path(path, commands, torque=1.7), self.rules, 50)
        self.assertEqual(strained["checks"]["applied_torque_nm"]["bound"], 1.6)
        self.assertEqual(strained["failed_checks"], ["applied_torque_nm"])

    def test_an_incomplete_or_missing_trace_fails(self):
        commands = [[0.05, 0.0, 0.0]] * 500
        trace = trace_from_path(judge.ghost_path((0.0, 0.0, 0.0), commands), commands)
        short = judge.judge_case(trace, self.rules, 1200)
        self.assertEqual(short["failed_checks"], ["complete_trace"])
        self.assertEqual(short["checks"]["complete_trace"]["value"], 500)
        self.assertFalse(judge.judge_case({}, self.rules, 1200)["pass"])

    def test_judgement_binds_the_requirements_hash_and_fails_on_the_drift_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commands = [[0.05, 0.0, 0.0]] * 1000
            path = judge.ghost_path((0.0, 0.0, 0.0), commands)
            path[:, 1] += np.arange(len(path)) * 0.0002  # 200 mm at the end
            case = {"case_id": "walk_0.050_0deg", "command": [0.05, 0, 0], "controls": 1000}
            save_batch(root, 0, case, trace_from_path(path, commands))
            result = judge.judge(root)
            self.assertEqual(result["schema"], judge.SCHEMA)
            self.assertEqual(result["checkpoint_sha256"], ["c" * 64])
            self.assertEqual(result["requirements_sha256"], judge.load_requirements()[1])
            self.assertFalse(result["pass"])
            self.assertEqual(result["failed_cases"], ["walk_0.050_0deg"])
            check = result["cases"][0]["checks"]["drift_per_hold_m"]
            self.assertEqual(check["bound"], 0.10)
            self.assertGreater(check["value"], 0.10)

    def test_every_command_of_the_set_is_a_drift_case_and_a_missing_trace_is_judged(self):
        rules = self.rules
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, case in enumerate(judge.cases(rules)[1:4]):
                commands = [judge.command_at(case, c) for c in range(case["controls"])]
                trace = trace_from_path(judge.ghost_path((0.0, 0.0, 0.0), commands), commands)
                save_batch(root, index, case, trace)
            missing = {"case_id": "turn_+0.20", "command": [0, 0, 0.2], "controls": 1000}
            batch = root / "batch_003"
            batch.mkdir()
            (batch / "report.json").write_text(json.dumps({"cases": [missing]}))
            result = judge.judge(root)
            for row in result["cases"][:3]:
                self.assertIn("drift_per_hold_m", row["checks"])
                self.assertNotIn("complete_trace", row["checks"])
            self.assertEqual(result["cases"][3]["failed_checks"], ["complete_trace"])
            self.assertEqual(result["failed_cases"], ["turn_+0.20"])


class RouteTests(unittest.TestCase):
    def setUp(self):
        rules, _ = judge.load_requirements()
        routes = {"forward_0.5m": [[0.5, 0.0]], "L": [[0.5, 0.0], [0.5, 0.5]]}
        self.rules = {**rules, "stop": {**rules["stop"], "routes_m": routes}}
        self.case = next(c for c in judge.cases(self.rules) if c["case_id"] == "route_forward_0.5m")

    def test_route_cases_come_from_the_file_and_size_their_trials_from_its_numbers(self):
        routes = [case for case in judge.cases(self.rules) if "route_m" in case]
        self.assertEqual([case["case_id"] for case in routes], ["route_forward_0.5m", "route_L"])
        # Path length at the slowest speed, the stay window and two settles.
        self.assertEqual(routes[0]["controls"], round((0.5 / 0.025 + 20.0 + 4.0) / 0.02))
        self.assertEqual(routes[1]["controls"], round((1.0 / 0.025 + 20.0 + 4.0) / 0.02))
        shipped = [c for c in judge.cases(judge.load_requirements()[0]) if "route_m" in c]
        self.assertEqual(len(shipped), 5)

    def test_the_producer_drives_the_follower_from_the_true_pose(self):
        # Body yaw pi/2 points anatomical forward (-y of the body) along world +x.
        facing_x = [0.0, 0.0, 0.1, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        self.assertAlmostEqual(judge.follower_pose(facing_x)[2], 0.0, places=9)
        self.assertEqual(judge.route_goals(self.case, (0.0, 0.0, 0.0)), [(0.5, 0.0)])
        goal = judge.route_goals(self.case, (1.0, 2.0, math.pi / 2))[0]
        self.assertAlmostEqual(goal[0], 1.0, places=9)
        self.assertAlmostEqual(goal[1], 2.5, places=9)
        produce = judge.producer(self.case)
        self.assertEqual(produce(0, facing_x), [0.05, 0.0, 0.0])
        arrived = [0.5, 0.0, 0.1, *facing_x[3:]]
        self.assertEqual(produce(1, arrived), [0.0, 0.0, 0.0])
        fixed = judge.producer(
            {"case_id": "f", "command": [0.0, 0.0, 0.2], "controls": 2, "stop_control": 1}
        )
        self.assertEqual(
            (fixed(0, facing_x), fixed(1, facing_x)), ([0.0, 0.0, 0.2], [0.0, 0.0, 0.0])
        )

    def route_trace(self, rest, zero_controls):
        n = self.case["controls"]
        commands = [[0.05, 0.0, 0.0]] * (n - zero_controls) + [[0.0, 0.0, 0.0]] * zero_controls
        path = np.zeros((n, 3))
        path[:, 2] = math.pi / 2
        path[n - zero_controls :, :2] = rest
        return trace_from_path(path, commands)

    def test_stop_position_requires_arriving_and_staying_within_the_tolerance(self):
        # 900 controls of walking stay under the 20 s drift hold, so only the stop is judged.
        stayed = judge.judge_case(self.route_trace((0.45, 0.02), 1300), self.rules, None, self.case)
        self.assertTrue(stayed["route"]["arrived"])
        check = stayed["checks"]["stop_position_m"]
        self.assertAlmostEqual(check["value"], math.hypot(0.05, 0.02), places=6)
        self.assertEqual(check["bound"], 0.12)
        self.assertTrue(check["pass"])
        self.assertTrue(stayed["pass"])
        short = judge.judge_case(self.route_trace((0.3, 0.0), 1300), self.rules, None, self.case)
        self.assertEqual(short["failed_checks"], ["stop_position_m"])
        self.assertAlmostEqual(short["checks"]["stop_position_m"]["value"], 0.2, places=6)
        restless = judge.judge_case(self.route_trace((0.5, 0.0), 500), self.rules, None, self.case)
        self.assertFalse(restless["route"]["arrived"])
        self.assertIn("stop_position_m", restless["failed_checks"])


if __name__ == "__main__":
    unittest.main()
