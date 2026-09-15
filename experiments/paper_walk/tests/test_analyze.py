"""Independent CPU diagnostic fixtures, never native physics evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.paper_walk import analyze as an
from experiments.paper_walk import learner


def channels():
    count, steps = 10, 80
    root = np.zeros((count, 1, 7)); root[..., 2] = .1; root[..., 6] = 1
    control = {"time_s": (np.arange(count)+1)*.02, "root_pose_xyzw": root,
        "command": np.tile([-.1, 0., 0.], (count, 1, 1)),
        "velocity_navigation_mps": np.tile([.1, 0., 0.], (count, 1, 1)),
        "gyro_body_rad_s": np.zeros((count, 1, 3)),
        "joint_position_rad": np.zeros((count, 1, 18)),
        "joint_velocity_rad_s": np.zeros((count, 1, 18)),
        "joint_target_rad": np.zeros((count, 1, 18)),
        "terminated": np.zeros((count, 1), bool), "truncated": np.zeros((count, 1), bool),
        "reset": np.zeros((count, 1), bool),
        "estimated_velocity_navigation_mps": np.full((count, 1, 3), .03),
        "critic_observation": np.zeros((count, 1, 234))}
    control["critic_observation"][..., -3:] = .03
    native = {"sequence": np.arange(steps), "substep_index": np.arange(steps)%8,
        "explicit_counter": np.arange(steps)+100,
        "computed_torque_nm": np.zeros((steps, 1, 18)), "applied_torque_nm": np.zeros((steps, 1, 18)),
        "distal_contact": np.ones((steps, 1, 6), bool), "nonfoot_contact": np.zeros((steps, 1), bool),
        "toe_xyz_world_m": np.zeros((steps, 1, 6, 3)),
        "joint_position_rad": np.zeros((steps, 1, 18)), "joint_velocity_rad_s": np.zeros((steps, 1, 18))}
    native["computed_torque_nm"][1:4, 0, 7] = 2.
    native["applied_torque_nm"][1:4, 0, 7] = 1.6
    native["toe_xyz_world_m"][:, 0, 0, 0] = np.arange(steps)*.001
    native["distal_contact"][5:8, 0, 2] = False
    return control, native


class AnalysisTests(unittest.TestCase):
    def test_fft_separates_real_high_frequency_from_constant_signal(self):
        time = np.arange(500)*.02
        result = an.spectrum(np.sin(2*np.pi*8*time), .02)
        self.assertAlmostEqual(result["dominant_frequency_hz"], 8.)
        self.assertGreater(result["power_above_5hz_fraction"], .99)
        self.assertIsNone(an.spectrum(np.ones(100), .02)["dominant_frequency_hz"])
        self.assertEqual(an.spectrum(np.ones(3), .02)["status"], "missing")

    def test_signed_tracking_and_intersample_torque_events(self):
        control, native = channels()
        result = an.analyze_trace(control, native)
        self.assertAlmostEqual(result["tracking"]["planar_error_rmse_mps"], .2)
        self.assertAlmostEqual(result["tracking"]["mean_signed_direction_speed_mps"], -.1)
        torque = result["torque_400hz"]
        self.assertAlmostEqual(torque["requested_saturation_fraction"][7], 3/80)
        self.assertAlmostEqual(torque["requested_saturation_longest_burst_s"][7], .0075)
        self.assertAlmostEqual(torque["first_requested_saturation_s"], .005)
        self.assertAlmostEqual(torque["applied_rms_nm"][7], 1.6*np.sqrt(3/80))

    def test_actual_support_slip_proxy_and_estimator_measurement_point(self):
        control, native = channels()
        result = an.analyze_trace(control, native)
        self.assertAlmostEqual(result["support_400hz"]["toe_contact_fraction"][2], 77/80)
        self.assertEqual(result["support_400hz"]["support_count_histogram"][5], 3)
        self.assertAlmostEqual(result["contact_reference_motion"]["mean_planar_speed_mps"][0], .4)
        self.assertEqual(result["estimator"]["root_origin_velocity_rmse_mps"], [0., 0., 0.])
        del native["distal_contact"]
        result = an.analyze_trace(control, native)
        self.assertIn("distal_contact", result["missing_metrics"])
        self.assertNotIn("support_400hz", result)

    def test_original_verdicts_preserved_and_hash_changes_rejected(self):
        control, native = channels()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"trial"; directory.mkdir()
            capture = directory/"native400hz"; capture.mkdir()
            np.savez_compressed(directory/"control_trace.npz", **control)
            np.savez_compressed(capture/"substeps_000.npz", **native)
            (capture/"capture.json").write_text(json.dumps({"steps": 80, "substep_files": ["substeps_000.npz"],
                "joint_names": ["fixture_"+str(x) for x in range(18)],
                "files": {"substeps_000.npz": an.sha(capture/"substeps_000.npz")}}))
            verdicts = [{"case_id": "fixture", "pass": False, "failed_bounds": ["tracking"]}]
            original = {"controls": 10, "results": verdicts, "cases": [{"case_id": "fixture"}],
                "failure": "preserved fixture failure", "acquisition_complete": False,
                "files": {"control_trace.npz": an.sha(directory/"control_trace.npz")}}
            (directory/"report.json").write_text(json.dumps(original))
            result = an.analyze(directory, Path(temporary)/"analysis")
            self.assertEqual(result["recordings"][0]["original_results"], verdicts)
            self.assertFalse(result["stage2_verdict_changed"])
            self.assertTrue((Path(temporary)/"analysis/analysis.md").is_file())
            with (capture/"substeps_000.npz").open("ab") as stream: stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                an.load_recording(directory)

    def test_zero_control_failure_is_missing_not_zero_performance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"trial"; directory.mkdir()
            (directory/"report.json").write_text(json.dumps({"controls": 0, "files": {}, "failure": "blank camera",
                "cases": [{"case_id": "failed_before_control"}], "results": [{"pass": False}]}))
            result = an.analyze(directory, Path(temporary)/"analysis")
            row = result["recordings"][0]["replicas"][0]
            self.assertEqual(row["status"], "no_control_samples")
            self.assertNotIn("tracking", row)
            self.assertEqual(result["recordings"][0]["original_failure"], "blank camera")

    def test_actor_reconstruction_uses_recorded231_inputs_and_checkpoint_hash(self):
        config = learner.Config(num_envs=1, actor_hidden=(8,), memory_hidden=(8,), estimator_hidden=(8,), critic_hidden=(8,), discriminator_hidden=(8,))
        model = learner.ActorCritic(config).eval()
        obs = torch.randn(7, 1, 231)
        with torch.inference_mode(): actions = model.actor(obs[:, 0])[0].numpy()[:, None]
        controls = {"policy_observation": obs.numpy(), "policy_action": actions.copy()}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/"policy.pt"
            torch.save({"schema": learner.SCHEMA, "config": asdict(config), "model": model.state_dict(), "learner_sha256": an.sha(learner.__file__)}, checkpoint)
            digest = an.sha(checkpoint)
            result = an.reconstruct_actor(controls, checkpoint, digest)
            self.assertTrue(result["within_cpu_float_tolerance"])
            controls["policy_action"] += .1
            self.assertFalse(an.reconstruct_actor(controls, checkpoint, digest)["within_cpu_float_tolerance"])
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                an.reconstruct_actor(controls, checkpoint, "0"*64)

    def test_comparison_requires_matching_full_identity(self):
        current = {"comparison_identity": None, "recordings": []}
        self.assertEqual(an.comparison(current, deepcopy(current))["status"], "incomparable")
        current["comparison_identity"] = [{"physics": "test binding", "cases": ["test case"]}]
        previous = deepcopy(current)
        self.assertEqual(an.comparison(current, previous)["status"], "matched_identity")
        previous["comparison_identity"][0]["physics"] = "another model"
        self.assertEqual(an.comparison(current, previous)["status"], "incomparable")


if __name__ == "__main__":
    unittest.main()
