"""Acceptance boundary fixtures; failures may not be hidden by a short window."""
import ast
from pathlib import Path
import math
import unittest

import numpy as np

from locomotion import evaluation as ev

ROOT = Path(__file__).resolve().parents[2]


def fixture(n=1000, command=(0., 0., 0.)):
    root = np.zeros((n, 7)); root[:, 2] = .10; root[:, 6] = 1.
    return {
        "root_pose_xyzw": root, "time_s": (np.arange(n)+1)*.02,
        "command": np.tile(command, (n, 1)), "velocity_navigation_mps": np.tile(command[:2]+(0.,), (n, 1)),
        "velocity_world_mps": np.zeros((n, 3)), "gyro_body_rad_s": np.tile((0.,0.,command[2]), (n, 1)),
        **{k: np.zeros((n, 18)) for k in ("joint_position_rad", "joint_velocity_rad_s", "joint_target_rad",
                                          "computed_torque_nm", "applied_torque_nm", "saturation_count_400hz")},
        **{k: np.zeros(n) for k in ("terminated", "truncated", "reset", "applied_torque_abs_max_400hz",
                                    "nonfoot_contact", "nonfoot_contact_count_400hz", "missing_six_toe_count_400hz")},
        "minimum_non_toe_floor_m": np.full(n, .01),
    }


def score(data, profile="quiet_stand", **kwargs):
    return ev.score_recording(data, {"profile": profile, "contact_classification": "exact_distal_points", **kwargs})


class QuietEvidenceTests(unittest.TestCase):
    def test_stage2_long_quiet_preserves32_seconds_and2_second_settle(self):
        result = score(fixture(1600), "stage2_long_quiet")
        self.assertTrue(result["pass"])
        self.assertEqual((result["window_start_control"], result["window_controls"]), (100, 1500))
        self.assertEqual(result["checks"]["quiet_window_controls"]["bound"], 1495)
        self.assertEqual(score(fixture())["window_start_control"], 200)

    def test_canonical_short_stand_cannot_replace_stage2_long_quiet(self):
        result = score(fixture(1000), "stage2_long_quiet")
        self.assertIn("complete_requested_window", result["failed_bounds"])
        self.assertIn("quiet_window_controls", result["failed_bounds"])

    def test_long_quiet_retains_early_failure_and_full_scored_window(self):
        data = fixture(1600)
        data["joint_velocity_rad_s"][100:300, 4] = .1
        result = score(data, "stage2_long_quiet")
        self.assertIn("max_joint_velocity_rms_rad_s", result["failed_bounds"])
        for flag in ("terminated", "truncated", "reset"):
            data = fixture(1600)
            data[flag][20] = 1
            self.assertIn(flag, score(data, "stage2_long_quiet")["failed_bounds"])

    def test_quiet_uses_all_sixteen_seconds(self):
        d = fixture()
        d["joint_velocity_rad_s"][201:301, 4] = .1
        result = score(d)
        self.assertEqual(result["window_controls"], 800)
        self.assertIn("max_joint_velocity_rms_rad_s", result["failed_bounds"])
        self.assertFalse(result["pass"])

    def test_short_video_cannot_pass_standing(self):
        result = score(fixture(600))
        self.assertIn("complete_requested_window", result["failed_bounds"])
        self.assertIn("quiet_window_controls", result["failed_bounds"])

    def test_early_failure_survives_quiet_suffix(self):
        for flag in ("terminated", "truncated", "reset"):
            d = fixture(); d[flag][20] = 1
            result = score(d)
            self.assertIn(flag, result["failed_bounds"])

    def test_missing_terminal_or_reset_channel_is_not_zero(self):
        d = fixture(); del d["reset"]
        result = score(d)
        self.assertIn("reset", result["missing_evidence"])
        self.assertFalse(result["pass"])

    def test_tibia_force_cannot_prove_toe_contact(self):
        result = score(fixture(), contact_classification="tibia_force_only")
        self.assertIn("missing_six_toe_count_400hz", result["missing_evidence"])
        self.assertIn("nonfoot_contact_count_400hz", result["missing_evidence"])

    def test_substep_saturation_cannot_hide_between_endpoints(self):
        d = fixture(); d["saturation_count_400hz"][200:300, 3] = 1
        result = score(d)
        self.assertEqual(result["metrics"]["max_requested_torque_saturation_fraction"], 0.)
        self.assertIn("requested_saturation_400hz", result["failed_bounds"])

    def test_one_support_dropout_is_a_failure(self):
        d = fixture(); d["missing_six_toe_count_400hz"][555] = 1
        self.assertIn("missing_six_toe_count_400hz", score(d)["failed_bounds"])

    def test_nonfinite_and_invalid_counts_are_rejected(self):
        d = fixture(); d["joint_velocity_rad_s"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            score(d)
        d = fixture(); d["saturation_count_400hz"][0, 0] = 9
        with self.assertRaisesRegex(ValueError, "counts"):
            score(d)

    def test_explicit_xyzw_heading(self):
        d = fixture(); angle = np.radians(3.)
        d["root_pose_xyzw"][900:, 5] = np.sin(angle/2)
        d["root_pose_xyzw"][900:, 6] = np.cos(angle/2)
        result = score(d)
        self.assertAlmostEqual(result["metrics"]["max_heading_excursion_deg"], 3.)
        self.assertIn("max_heading_excursion_deg", result["failed_bounds"])

    def test_stop_scores_from_first_zero_plus_two_seconds(self):
        d = fixture(1050); d["command"][:400, 0] = .1
        d["command"][400:420, 0] = .02
        result = score(d, "stop_to_stand")
        self.assertEqual(result["window_start_control"], 520)
        self.assertEqual(result["window_controls"], 530)
        self.assertTrue(result["pass"])

    def test_stop_with_late_zero_cannot_select_last_quiet_seconds(self):
        d = fixture(1050); d["command"][:500, 0] = .1
        result = score(d, "stop_to_stand")
        self.assertEqual(result["window_start_control"], 600)
        self.assertIn("quiet_window_controls", result["failed_bounds"])

    def test_stop_cannot_relabel_a_different_preceding_direction(self):
        d = fixture(1050); d["command"][:400, 0] = .1
        result = score(d, "stop_to_stand", command=[0., .1, 0.])
        self.assertIn("preceding_command", result["failed_bounds"])

    def test_per_replica_failures_not_averaged(self):
        d = fixture()
        batch = {k: np.stack((v, v), axis=1) for k, v in d.items()}
        batch["joint_velocity_rad_s"][200:, 1, 2] = .031
        self.assertTrue(score(batch, env_index=0)["pass"])
        self.assertFalse(score(batch, env_index=1)["pass"])

    def test_frozen_native_names_and_per_joint_cap_aggregation(self):
        d = fixture()
        d["linear_velocity_nav"] = d.pop("velocity_navigation_mps")
        d["angular_velocity_body"] = d.pop("gyro_body_rad_s")
        d["linear_velocity_body"] = np.zeros((1000, 3)); d.pop("velocity_world_mps")
        d["applied_torque_abs_max_400hz"] = np.zeros((1000, 18))
        d["applied_torque_abs_max_400hz"][500, 17] = 1.61
        result = score(d)
        self.assertIn("applied_cap_all_substeps", result["failed_bounds"])
        self.assertIn("velocity_world_mps", result["derived_telemetry"])

    def test_world_vertical_rotates_actual_body_velocity(self):
        d = fixture(command=(.1,0.,0.))
        angle = np.radians(4.)
        d["root_pose_xyzw"][:,3] = np.sin(angle/2)
        d["root_pose_xyzw"][:,6] = np.cos(angle/2)
        d.pop("velocity_world_mps")
        d["linear_velocity_body"] = np.tile((0.,-1.,0.), (1000,1))
        result = score(d, "omni_static", command=[.1,0.,0.])
        self.assertAlmostEqual(result["metrics"]["vertical_velocity_rms_mps"], np.sin(angle))
        self.assertIn("vertical_velocity_rms_mps", result["failed_bounds"])


class MotionEvidenceTests(unittest.TestCase):
    def test_reverse_direction_not_absolute_speed(self):
        d = fixture(command=(-.1, 0., 0.))
        self.assertTrue(score(d, "omni_static", command=[-.1,0.,0.])["pass"])
        d["velocity_navigation_mps"][:, 0] = .1
        self.assertIn("planar_error_mps", score(d, "omni_static", command=[-.1,0.,0.])["failed_bounds"])

    def test_original_static_command_matrix_and_transitions_unchanged(self):
        source = ast.parse((ROOT/"locomotion/tests/fixtures/omni_flat_math.py").read_text())
        nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in
                 ("evaluation_scenarios", "transition_sequence", "trajectory_command")]
        namespace = {"math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "historical_functions", "exec"), namespace)
        self.assertEqual(ev.evaluation_scenarios(), namespace["evaluation_scenarios"]())
        self.assertEqual(len(ev.evaluation_scenarios()), 77)
        self.assertEqual(ev.transition_sequence(), namespace["transition_sequence"]())
        for name, duration, command in ev.transition_sequence():
            self.assertEqual(ev.trajectory_command(name, duration*.75, duration, command),
                             namespace["trajectory_command"](name, duration*.75, duration, command))

    def test_original_quiet_bounds_unchanged(self):
        source = ast.parse((ROOT/"locomotion/tests/fixtures/omni_quiet_review.py").read_text())
        assignment = next(n for n in source.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "QUIET_GATES" for t in n.targets))
        self.assertEqual(ev.QUIET_GATES, ast.literal_eval(assignment.value))

    def test_historical_forward_thresholds_and_targets_unchanged(self):
        source = ast.parse((ROOT/"locomotion/tests/fixtures/grade_stage2c_stable_forward.py").read_text())
        tables = {}
        for node in source.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in {"THRESHOLDS", "STABILITY_COMPONENTS", "STAND_STABILITY_COMPONENTS", "TAIL_STABILITY_COMPONENTS"}:
                    tables[node.target.id] = ast.literal_eval(node.value)
        for ours, theirs in ((ev.MOVING_RMS, "STABILITY_COMPONENTS"),
                             (ev.STAND_RMS, "STAND_STABILITY_COMPONENTS"), (ev.TAIL, "TAIL_STABILITY_COMPONENTS")):
            self.assertEqual(ours, {row[0]: row[3] for row in tables[theirs]})
        self.assertEqual(ev.STAGE2C_TORQUE, {key: tables["THRESHOLDS"][key] for key in ev.STAGE2C_TORQUE})

    def test_formal_forward_rejects_speed_and_strict_deck_motion(self):
        d = fixture(500, (.3,0.,0.)); d["velocity_navigation_mps"][:,0] = .239
        d["velocity_world_mps"][:,2] = .071
        result = score(d, "stage2c_formal", command=[.3,0.,0.], seed=60, target_slew_rad=.040)
        self.assertEqual(result["window_controls"], 475)
        self.assertIn("forward_fraction_minimum", result["failed_bounds"])
        self.assertIn("vertical_velocity_rms_mps", result["failed_bounds"])
        self.assertFalse(result["stage2_complete"])

    def test_full_transition_episode_retains_failure(self):
        commands = np.array([ev.trajectory_command(name, i*.02, duration, c)
                             for name, duration, c in ev.transition_sequence()
                             for i in range(round(duration/.02))])
        d = fixture(len(commands)); d["command"] = commands
        d["requested_command"] = commands.copy()
        d["velocity_navigation_mps"][:,:2] = commands[:,:2]
        d["gyro_body_rad_s"][:,2] = commands[:,2]
        self.assertTrue(ev.score_transitions(d, {})["pass"])
        d["reset"][40] = 1
        self.assertIn("reset", ev.score_transitions(d, {})["failed_bounds"])

    def test_suite_does_not_promote_partial_or_duplicate_evidence(self):
        result = ev.summarize_suite([{"case_id":"forward", "pass":True}], required_cases=["forward", "reverse"])
        self.assertEqual(result["missing_cases"], ["reverse"])
        self.assertIn("matching_batch_standing", result["missing_qualification_evidence"])
        self.assertFalse(result["stage2_complete"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ev.summarize_suite([{"case_id":"x"}, {"case_id":"x"}], required_cases=["x"])


if __name__ == "__main__":
    unittest.main()
