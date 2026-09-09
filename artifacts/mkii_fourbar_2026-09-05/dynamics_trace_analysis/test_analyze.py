"""Synthetic file-integrity, named-order and pre/post event regression cases."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze import HINGE_FIELDS, MOTOR_FIELDS, TREE_FIELDS, analyze, digest


class Fixture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.joints = ["lf_tibia_pitch", "lf_tibia_lever_pivot", "lf_femur_pitch"]
        self.motors = ["lf_tibia_lever_pivot", "lf_femur_pitch"]
        self.columns = []
        for fields, names in ((TREE_FIELDS, self.joints[::-1]), (MOTOR_FIELDS, self.motors[::-1]),
                              ((*HINGE_FIELDS, "foot_force_w"), ["lf_"+x for x in "xyz"]),
                              (("body_link_linear_velocity_w", "body_link_angular_velocity_w"),
                               [body+"_"+axis for body in ("lf_tibia", "lf_tibia_pushrod") for axis in "xyz"])):
            self.columns += [key+"/"+name for key in fields for name in names]
        self.values = np.zeros((4, 2, len(self.columns)), dtype=np.float32)
        self.dt = .002
        for sample in range(4):
            for env in range(2):
                for joint in self.joints:
                    q = sample*.01
                    qd = -12.5 if (sample, env, joint) == (2, 1, self.motors[0]) else 1.
                    for key, value in (("pre_q", q), ("direct_pre_q", q), ("post_q", q+2*self.dt),
                                       ("pre_qd", qd), ("direct_pre_qd", qd), ("post_qd", 3.)):
                        self.set(sample, env, key, joint, value)
                for motor in self.motors:
                    qd = self.get(sample, env, "pre_qd", motor)
                    p, d, ff = .6, -.6*qd, .1
                    for key, value in (("target", sample*.01+.02), ("processed_target", sample*.01+.02),
                        ("p_term", p), ("d_term", d), ("feedforward", ff), ("demand", p+d+ff),
                        ("applied", min(5.5, p+d+ff)), ("instantaneous_limit", 5.5), ("headroom", .5)):
                        self.set(sample, env, key, motor, value)
                self.set(sample, env, "foot_force_w", "lf_z", 10.)
                self.set(sample, env, "body_link_linear_velocity_w", "lf_tibia_x", .1)
        self.set(1, 0, "hinge_gap_local", "lf_x", .001)
        self.set(1, 0, "hinge_gap_local", "lf_y", .002)
        self.set(2, 1, "direct_pre_q", self.motors[0], .0202)
        self.set(2, 1, "direct_pre_qd", self.motors[0], -12.)
        self.report = {"schema": "hexapod.fourbar_diagnostic.v1", "num_envs": 2,
            "joint_names": self.joints, "active_motor_names": self.motors,
            "numerical_recipe": {"physics_dt_s": self.dt, "decimation": 1},
            "diagnostic_complete": True, "trace_samples": 4, "force_writes": 4,
            "physics_substeps": 4, "steps_completed": 2, "driven_steps_completed": 2,
            "runtime_manifest": {"observed_motor_model_joint_names": self.motors[::-1]},
            "motor_readback": {"armature": [[.0008, .0007]], "stiffness": [[30., 30.]], "damping": [[.6, .6]]},
            "solver_readback": {"armatures": [[0., .0007, .0008]], "stiffnesses": [[0., 0., 0.]], "dampings": [[0., 0., 0.]]},
            "trace_files": []}
        self.write_files()

    def set(self, sample, env, key, name, value):
        self.values[sample, env, self.columns.index(key+"/"+name)] = value

    def get(self, sample, env, key, name):
        return self.values[sample, env, self.columns.index(key+"/"+name)]

    def write_files(self, final_samples=2):
        self.report["trace_files"] = []
        for index, (first, count) in enumerate(((0, 2), (2, final_samples))):
            file = self.directory/f"trace_{index:03d}.npz"
            values = self.values[first:first+count]
            np.savez_compressed(file, values=values, columns=np.asarray(self.columns))
            self.report["trace_files"].append({"file": file.name, "sha256": digest(file),
                "shape": list(values.shape), "first_physics_sample": first,
                "segment": {"phase": "standing" if index == 0 else "driven", "steps": 2,
                            "motors": [] if index == 0 else [self.motors[0]], "offset_rad": 0. if index == 0 else .04}})
        self.save()

    def save(self):
        self.path = self.directory/"report.json"
        self.path.write_text(json.dumps(self.report))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_named_peaks_windows_and_pre_post_velocity(self):
        result = analyze(self.fixture.path, window=1)
        demand = result["events"]["demand"]
        self.assertEqual(demand["peak"]["name"], "lf_tibia_lever_pivot")
        self.assertEqual((demand["peak"]["physics_sample"], demand["peak"]["environment"]), (2, 1))
        self.assertAlmostEqual(demand["peak"]["max_abs"], 8.2, places=5)
        self.assertEqual([x["physics_sample"] for x in demand["window"]["rows"]], [1, 2, 3])
        self.assertEqual([x["segment_index"] for x in demand["window"]["rows"]], [0, 1, 1])
        self.assertAlmostEqual(result["global_metrics"]["finite_difference_dq_dt"]["max_abs"], 2., places=5)
        self.assertAlmostEqual(result["global_metrics"]["post_qd_minus_dq_dt"]["max_abs"], 1., places=5)
        self.assertAlmostEqual(result["global_metrics"]["cached_minus_direct_pre_qd"]["max_abs"], .5)
        self.assertAlmostEqual(result["global_metrics"]["cached_minus_direct_pre_q"]["max_abs"], .0002, places=7)
        self.assertLess(result["global_metrics"]["p_plus_d_plus_ff_minus_demand"]["max_abs"], 1e-6)
        closure = result["events"]["hinge_gap_local_norm"]
        self.assertEqual((closure["peak"]["physics_sample"], closure["peak"]["environment"]), (1, 0))
        self.assertAlmostEqual(closure["peak"]["max_abs"], np.sqrt(5)*.001, places=8)
        self.assertEqual(closure["named_state"]["foot_force_w"]["lf_z"], 10.)
        self.assertEqual(closure["anatomical_context"]["foot_net_forces_w_by_body"]["lf_tibia"], [0., 0., 10.])
        # Reversed trace columns and readback motor order must not alter identity.
        armature = result["readbacks"]["motor_readback"]["values_by_name"]["armature"]
        self.assertEqual(armature["lf_tibia_lever_pivot"], [.0007])
        self.assertEqual(armature["lf_femur_pitch"], [.0008])
        json.dumps(result, allow_nan=False)

    def test_tampered_bytes_fail_before_numeric_analysis(self):
        with (Path(self.temp.name)/"trace_000.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            analyze(self.fixture.path)

    def test_missing_or_overlapping_ranges_rejected(self):
        for first in (1, 3):
            with self.subTest(first=first):
                self.fixture.report["trace_files"][1]["first_physics_sample"] = first
                self.fixture.save()
                with self.assertRaisesRegex(ValueError, "gap, overlap"):
                    analyze(self.fixture.path)

    def test_completed_missing_samples_rejected_but_partial_labeled(self):
        self.fixture.write_files(final_samples=1)
        for key in ("trace_samples", "force_writes", "physics_substeps"):
            self.fixture.report[key] = 3
        self.fixture.report["driven_steps_completed"] = 1
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "completed segment"):
            analyze(self.fixture.path)
        self.fixture.report["diagnostic_complete"] = False
        self.fixture.save()
        result = analyze(self.fixture.path, 1)
        self.assertFalse(result["source_diagnostic_complete"])
        self.assertEqual(result["trace_samples"], 3)

    def test_duplicate_missing_and_nonfinite_columns_rejected(self):
        original = list(self.fixture.columns)
        for problem in ("duplicate", "missing", "nonfinite"):
            with self.subTest(problem=problem):
                self.fixture.columns = list(original)
                self.fixture.values[:] = np.nan if problem == "nonfinite" else 0.
                if problem == "duplicate":
                    self.fixture.columns[1] = self.fixture.columns[0]
                elif problem == "missing":
                    self.fixture.columns[1] = "pre_q/unknown_joint"
                self.fixture.write_files()
                with self.assertRaises(ValueError):
                    analyze(self.fixture.path)

    def test_pd_reconstruction_error_is_reported_not_hidden(self):
        value = self.fixture.get(3, 1, "demand", self.fixture.motors[0])
        self.fixture.set(3, 1, "demand", self.fixture.motors[0], value+1.)
        self.fixture.write_files()
        result = analyze(self.fixture.path, 0)
        self.assertAlmostEqual(result["global_metrics"]["p_plus_d_plus_ff_minus_demand"]["max_abs"], 1., places=6)
        self.assertEqual(sum(s["p_d_ff_reconstruction_outside_float32_rounding_count"] for s in result["segments"]), 1)

    def test_unrecorded_model_order_is_not_guessed(self):
        self.fixture.report["runtime_manifest"] = {}
        self.fixture.save()
        result = analyze(self.fixture.path, 0)
        self.assertFalse(result["readbacks"]["motor_readback"]["available_by_name"])
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
