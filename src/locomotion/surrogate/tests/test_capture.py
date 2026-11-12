"""Check live capture, reset motor snapshots and independent evidence rejection."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from locomotion import randomization as rnd
from locomotion.evaluation import requirements, runner
from locomotion.robots.hexapod import KD
from locomotion.surrogate.capture import SurrogateCaptureSource
from locomotion.surrogate.env import SurrogateConfig, SurrogateEnv


class CaptureTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def env(self, **options):
        env = SurrogateEnv(SurrogateConfig(num_envs=1), threads=0, **options)
        self.addCleanup(env.close)
        return env

    def run_batch(self, env, name="run", source=SurrogateCaptureSource, controls=2):
        return runner.run_batch(
            env,
            lambda obs: torch.full((1, 18), 0.2),
            [{"case_id": "sample", "command": [0.0, 0.0, 0.0], "controls": controls}],
            self.root / name,
            env.geometry_extrema_path,
            checkpoint_sha256="test",
            source_sha256="test",
            capture_source=source,
        )

    def test_nominal_and_fixed_strength_capture_complete_with_bound_source_metadata(self):
        for name, condition in (
            ("nominal", None),
            ("strength08", rnd.cell_draw("table2_v1_grid", "mu1_payload0.000kg_strength0.8")),
        ):
            with self.subTest(name=name):
                env = self.env(condition=condition)
                self.assertFalse(any(hasattr(env, key) for key in ("sim", "contact", "sensor_map")))
                report = self.run_batch(env, name, controls=20)
                self.assertIsNone(report["failure"])
                self.assertEqual(report["recorded_physics_steps"], 160)
                directory = self.root / name / "native400hz"
                source = json.loads((directory / "source.json").read_text())
                receipt = json.loads((directory / "capture.json").read_text())
                self.assertIn("no native patch completeness claim", source["contacts"])
                self.assertEqual(receipt["contact_classification"], "per_geom_netforce_positions")
                self.assertEqual(
                    report["contact_classification"], receipt["contact_classification"]
                )
                self.assertEqual(report["force_metrics"]["status"], "available")
                np.testing.assert_array_equal(
                    np.asarray(source["motor_snapshot"]["strength"], np.float32),
                    np.full((1, 18), 1.0 if condition is None else 0.8, np.float32),
                )
                for filename, digest in receipt["files"].items():
                    self.assertEqual(
                        hashlib.sha256((directory / filename).read_bytes()).hexdigest(), digest
                    )
                with np.load(directory / "substeps_000.npz") as raw:
                    np.testing.assert_array_equal(
                        raw["native_input_pre_nm"], raw["applied_torque_nm"]
                    )
                    self.assertEqual(raw["native_input_pre_nm"].dtype, np.dtype("float32"))
                    self.assertEqual(raw["distal_force_world_n"].dtype, np.dtype("float64"))
                    self.assertLessEqual(float(abs(raw["applied_torque_nm"]).max()), 1.60001)

    def test_run_cases_snapshots_randomized_coefficients_after_its_reset(self):
        env = self.env(randomization="table2_v1")
        before = env._kp.clone()
        rules, _ = requirements.load_requirements()
        rules["hold_max_s"] = 0.02
        rules["stop"]["settle_s"] = 0.01
        path = self.root / "requirements.json"
        path.write_text(json.dumps({"sets": {"short": rules}}))
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"capture integration fixture")
        snapshots = []

        def source(reset_env):
            snapshots.append(SurrogateCaptureSource(reset_env))
            return snapshots[-1]

        judgement = runner.run_cases(
            env,
            lambda obs: torch.zeros(1, 18),
            self.root / "cases",
            env.geometry_extrema_path,
            checkpoint,
            requirements_path=path,
            requirements_set="short",
            case_ids=["stand"],
            capture_source=source,
        )
        self.assertIsNone(judgement["failure"])
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(torch.equal(before, env._kp))
        np.testing.assert_array_equal(snapshots[0].kp, env._kp.numpy())
        receipt = json.loads((self.root / "cases/batch_000/native400hz/capture.json").read_text())
        self.assertEqual(receipt["steps"], 16)

    def test_oracle_keeps_high_speed_strength_and_gain_arithmetic_independent(self):
        env = SimpleNamespace(
            num_envs=2,
            randomized=True,
            _kp=torch.tensor([[13.2], [10.8]], dtype=torch.float32),
            _kd=torch.tensor(KD, dtype=torch.float32)[None].repeat(2, 1) * 0.9,
            _strength=torch.full((2, 18), 1.2),
            native_readback={
                "surrogate": {"engine": "test"},
                "randomization": rnd.identity("none"),
            },
        )
        source = SurrogateCaptureSource(env)
        q = torch.linspace(-0.2, 0.2, 36).reshape(2, 18)
        target = -q
        rpm = torch.tensor([0, 70, 275, 340, 450, 460, 477, 479, 480] * 2, dtype=torch.float32)
        dq = (rpm * (2 * np.pi / 60))[None].repeat(2, 1)
        dq[1] *= -1
        expected = rnd.motor_force(q, dq, target, env._kd, env._kp, env._strength)
        # A later mutation must not silently redefine this recorded trial's motor law.
        env._kp.fill_(1.0)
        env._kd.fill_(1.0)
        env._strength.fill_(0.8)
        actual = source.servo(q.numpy(), dq.numpy(), target.numpy())
        for value, oracle in zip(expected, actual):
            np.testing.assert_array_equal(value.numpy(), oracle)
            self.assertEqual(oracle.dtype, np.dtype("float32"))
        self.assertEqual(actual[2][0, 0], np.float32(1.6))
        self.assertAlmostEqual(float(actual[2][0, 6]), 0.6, places=5)
        self.assertEqual(actual[2][0, 8], 0.0)

    def test_changed_physics_input_preserves_a_failed_prefix(self):
        env = self.env()
        advance = env._advance

        def wrong_input(applied, substep):
            result = advance(applied, substep)
            env._control[0, 0, 0] += 0.125
            return result

        with patch.object(env, "_advance", side_effect=wrong_input):
            report = self.run_batch(env)
        self.assertIn("Actual native motor input differs", report["failure"])
        self.assertEqual(report["recorded_physics_steps"], 1)
        self.assertFalse(report["acquisition_complete"])
        raw = self.root / "run/native400hz"
        self.assertTrue((raw / "failed_partial_step.json").is_file())
        with np.load(raw / "substeps_000.npz") as data:
            self.assertNotEqual(
                data["native_input_pre_nm"][0, 0, 0], data["applied_torque_nm"][0, 0, 0]
            )

    def test_motor_coefficients_changed_after_snapshot_are_rejected(self):
        env = self.env(condition=rnd.cell_draw("table2_v1_grid", "mu1_payload0.000kg_strength0.8"))

        def source(reset_env):
            result = SurrogateCaptureSource(reset_env)
            reset_env._strength.fill_(0.7)
            return result

        report = self.run_batch(env, source=source)
        self.assertIn("Canonical motor recurrence differs: effort_ceiling_nm", report["failure"])
        self.assertEqual(report["recorded_physics_steps"], 1)

    def test_each_corrupted_motor_output_is_rejected_by_the_independent_oracle(self):
        env = self.env(condition=rnd.cell_draw("table2_v1_grid", "mu1_payload0.000kg_strength0.8"))
        motor = rnd.motor_force
        for index, field in enumerate(
            ("computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm")
        ):
            with self.subTest(field=field):

                def corrupt(*args):
                    result = list(motor(*args))
                    result[index] = result[index] + 0.01
                    return result

                with patch("locomotion.surrogate.env.rnd.motor_force", side_effect=corrupt):
                    report = self.run_batch(env, name=field)
                self.assertIn("Canonical motor recurrence differs: " + field, report["failure"])
                self.assertEqual(report["recorded_physics_steps"], 1)


if __name__ == "__main__":
    unittest.main()
