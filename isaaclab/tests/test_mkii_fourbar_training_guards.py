"""CPU failure controls for admission, PPO state and every-substep guards."""
from __future__ import annotations

import copy
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from tools.mkii_training_contract import identity, require_admission, require_checkpoint, write_json, digest, TASK_ID

spec = importlib.util.spec_from_file_location("fourbar_trainer_guard_tests", ROOT / "isaaclab/train_mkii_fourbar.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class SourceAdmissionTests(unittest.TestCase):
    def make_source(self, root):
        paths = {
            "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda": "#usda1.0",
            "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf": "<robot/>",
            "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json": "{}",
            "packages/hexapod_core/hexapod_core/rs05_v2.json": '{"peak":5.5}',
        }
        for name, value in paths.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)

    def test_motor_json_change_invalidates_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            original = identity(root)
            report_path = root / "admission.json"
            write_json(report_path, {"pass": True, "simulation_training_admission": True,
                "task_id": TASK_ID, "errors": [], "contract": original, "num_envs": 32, "steps_completed": 1000})
            require_admission(report_path, original)
            (root / "packages/hexapod_core/hexapod_core/rs05_v2.json").write_text('{"peak":55}')
            changed = identity(root)
            self.assertNotEqual(original, changed)
            with self.assertRaises(ValueError): require_admission(report_path, changed)

    def test_admission_counts_must_be_real_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            good = {"pass": True, "simulation_training_admission": True, "task_id": TASK_ID,
                    "errors": [], "contract": {}, "num_envs": 32, "steps_completed": 1000}
            for key, bad in [("num_envs", "32"), ("steps_completed", 1000.0), ("num_envs", float("inf"))]:
                path.write_text(json.dumps(dict(good, **{key: bad})))
                with self.assertRaises(ValueError): require_admission(path, {})

    def test_checkpoint_sidecar_must_match_bytes_contract_and_next_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"example checkpoint")
            sidecar = checkpoint.with_suffix(".pt.json")
            write_json(sidecar, {"contract": {"id": 1}, "checkpoint_sha256": digest(checkpoint), "next_iteration": 10})
            self.assertEqual(require_checkpoint(checkpoint, {"id": 1})["next_iteration"], 10)
            with self.assertRaises(ValueError): require_checkpoint(checkpoint, {"id": 2})
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(ValueError): require_checkpoint(checkpoint, {"id": 1})


class LearningStateTests(unittest.TestCase):
    def state(self):
        return {"actor_state_dict": {"weight": torch.ones(2), "normalizer": torch.ones(2)},
                "critic_state_dict": {"weight": torch.ones(2)},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(5.), "exp_avg": torch.zeros(2)}},
                                         "param_groups": [{"lr": .0001, "params": [0]}]}}

    def test_digest_catches_critic_moments_and_normalizer_changes(self):
        original = self.state()
        expected = trainer.finite_state_digest(original)
        for group, field in [("critic_state_dict", "weight"), ("actor_state_dict", "normalizer")]:
            changed = copy.deepcopy(original)
            changed[group][field][0] = 2
            self.assertNotEqual(expected, trainer.finite_state_digest(changed))
        changed = copy.deepcopy(original)
        changed["optimizer_state_dict"]["state"][0]["exp_avg"][0] = .1
        self.assertNotEqual(expected, trainer.finite_state_digest(changed))

    def test_nonfinite_buffers_and_optimizer_moments_fail(self):
        for group, field in [("actor_state_dict", "normalizer"), ("critic_state_dict", "weight")]:
            changed = self.state()
            changed[group][field][0] = float("nan")
            with self.assertRaises(ValueError): trainer.finite_state_digest(changed)
        changed = self.state()
        changed["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("inf")
        with self.assertRaises(ValueError): trainer.finite_state_digest(changed)
        with self.assertRaises(ValueError): trainer.finite_state_digest({"learning_rate": float("nan")})

    def test_digest_handles_scalar_and_bfloat_buffers_and_structure(self):
        a = trainer.finite_state_digest({"value": torch.tensor(1., dtype=torch.bfloat16), "steps": torch.tensor(3)})
        self.assertEqual(a, trainer.finite_state_digest({"steps": torch.tensor(3), "value": torch.tensor(1., dtype=torch.bfloat16)}))
        self.assertNotEqual(trainer.finite_state_digest([1, 2]), trainer.finite_state_digest((1, 2)))

    def test_load_restores_scheduler_scalar_from_optimizer(self):
        algorithm = types.SimpleNamespace(learning_rate=.001,
            optimizer=types.SimpleNamespace(param_groups=[{"lr": .00013}, {"lr": .00013}]))
        trainer.restore_adaptive_learning_rate(algorithm)
        self.assertEqual(algorithm.learning_rate, .00013)
        for bad in ([], [{"lr": float("nan")}], [{"lr": 0}], [{"lr": .1}, {"lr": .2}]):
            algorithm.optimizer.param_groups = bad
            with self.assertRaises(ValueError): trainer.restore_adaptive_learning_rate(algorithm)

    def test_actor_changed_means_parameters_not_only_normalization(self):
        policy = torch.nn.Linear(2, 1)
        policy.register_buffer("running_mean", torch.zeros(2))
        runner = types.SimpleNamespace(alg=types.SimpleNamespace(get_policy=lambda: policy))
        original = trainer.parameters_digest(runner)
        policy.running_mean += 1
        self.assertEqual(original, trainer.parameters_digest(runner))
        with torch.no_grad(): policy.weight += 1
        self.assertNotEqual(original, trainer.parameters_digest(runner))


class AdmittedRuntimeTests(unittest.TestCase):
    def manifest(self):
        return {"schema": "hexapod.physical_fourbar_runtime.v1", "task_id": TASK_ID,
                "model_id": "mkii_fourbar_v3", "usd_root_sha256": "a"*64,
                "motor_contract": {"model_id": "RS05-provisional-v2", "physics_dt_s": .00125,
                                   "implementation_sha256": {"runtime.py": "b"*64}},
                "resolved_motor_configuration": {"armature": .0007, "effort_limit_sim": 5.5},
                "observed_tree_joint_names": ["lf_coxa_yaw", "lf_femur_pitch", "lf_tibia_lever_pivot"],
                "observed_motor_model_joint_names": ["lf_femur_pitch", "lf_coxa_yaw", "lf_tibia_lever_pivot"],
                "resolved_simulation": {"solver_position_iterations": 64, "solver_velocity_iterations": 1,
                                        "reset_motor_jitter_rad": 0., "self_collision_enabled": False},
                "future_asset_identity": {"closure_type": "revolute", "dependencies": {"base.usda": "c"*64}}}

    def test_exact_manifest_accepts_only_object_key_reordering(self):
        actual = self.manifest()
        admitted = json.loads(json.dumps(dict(reversed(list(actual.items())))))
        result = trainer.compare_admitted_runtime(admitted, actual)
        self.assertTrue(result["pass"])
        self.assertEqual(result["admitted_sha256"], result["actual_sha256"])
        self.assertEqual(result["excluded_fields"], [])
        self.assertEqual(result["mismatch_paths"], [])
        self.assertEqual(result["errors"], [])

    def test_asset_motor_solver_and_joint_order_changes_all_fail(self):
        baseline = self.manifest()
        cases = [(("usd_root_sha256",), "d"*64), (("model_id",), "mkii_fourbar_v4"),
                 (("motor_contract", "model_id"), "other"),
                 (("motor_contract", "implementation_sha256", "runtime.py"), "e"*64),
                 (("resolved_motor_configuration", "armature"), .007),
                 (("resolved_motor_configuration", "effort_limit_sim"), 55.),
                 (("resolved_simulation", "solver_position_iterations"), 128),
                 (("resolved_simulation", "reset_motor_jitter_rad"), .01),
                 (("observed_tree_joint_names",), list(reversed(baseline["observed_tree_joint_names"]))),
                 (("observed_motor_model_joint_names",), list(reversed(baseline["observed_motor_model_joint_names"]))),
                 (("future_asset_identity", "closure_type"), "planar_d6"),
                 (("future_asset_identity", "dependencies", "base.usda"), "f"*64)]
        for keys, value in cases:
            actual = copy.deepcopy(baseline)
            parent = actual
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            with self.subTest(keys=keys):
                result = trainer.compare_admitted_runtime(baseline, actual)
                self.assertFalse(result["pass"])
                self.assertNotEqual(result["admitted_sha256"], result["actual_sha256"])
                self.assertTrue(any(path.startswith("/" + "/".join(keys)) for path in result["mismatch_paths"]))

    def test_missing_extra_nonfinite_and_type_coercions_fail(self):
        baseline = self.manifest()
        for side in ("admitted", "actual"):
            for change in (None, {}, [], dict(baseline, extra_assumption="new")):
                pair = {"admitted": baseline, "actual": baseline}
                pair[side] = change
                with self.subTest(side=side, change=change):
                    self.assertFalse(trainer.compare_admitted_runtime(**pair)["pass"])
        for value in (True, 1., float("nan"), float("inf"), (1,)):
            actual = copy.deepcopy(baseline)
            actual["resolved_simulation"]["solver_velocity_iterations"] = value
            with self.subTest(value=value):
                self.assertFalse(trainer.compare_admitted_runtime(baseline, actual)["pass"])
        actual = copy.deepcopy(baseline)
        del actual["future_asset_identity"]
        self.assertIn("/future_asset_identity", trainer.compare_admitted_runtime(baseline, actual)["mismatch_paths"])

    def pre_wrapper_block(self):
        """Execute the actual trainer statements from gym.make through wrapper creation."""
        module = ast.parse((ROOT / "isaaclab/train_mkii_fourbar.py").read_text())
        for node in ast.walk(module):
            if not isinstance(node, ast.Try):
                continue
            start = next((i for i, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "env" for target in statement.targets)
                and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "make"), None)
            stop = next((i for i, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "wrapped" for target in statement.targets)), None)
            if start is not None and stop is not None:
                return compile(ast.Module(body=node.body[start:stop+1], type_ignores=[]), "trainer_pre_wrapper", "exec")
        self.fail("Trainer gym.make/wrapper initialization block not found")

    def test_actual_training_path_rejects_before_wrapper_and_records_comparison(self):
        baseline = self.manifest()
        code = self.pre_wrapper_block()
        for matching in (True, False):
            actual = copy.deepcopy(baseline)
            if not matching:
                actual["usd_root_sha256"] = "f"*64
            wrapper = Mock()
            scope = {"gym": types.SimpleNamespace(make=lambda *a, **k: types.SimpleNamespace(
                        unwrapped=types.SimpleNamespace(runtime_manifest=actual))),
                     "TASK_ID": TASK_ID, "cfg": object(), "report": {},
                     "admission": {"runtime_manifest": baseline}, "compare_admitted_runtime": trainer.compare_admitted_runtime,
                     "torch": types.SimpleNamespace(__version__="test"),
                     "agent_cfg": types.SimpleNamespace(clip_actions=None), "RslRlVecEnvWrapper": wrapper}
            if matching:
                exec(code, scope)
                wrapper.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, "differs from admitted nominal"):
                    exec(code, scope)
                wrapper.assert_not_called()
            self.assertEqual(scope["report"]["admitted_runtime_comparison"]["pass"], matching)
            self.assertEqual(scope["report"]["runtime_manifest"], actual)


class FakeScene:
    def __init__(self): self.state = 0
    def update(self, dt): self.state += 1


class FakeMetrics:
    def __init__(self, raw):
        self.raw, self.pending, self.windows, self.seen = raw, [], {}, []
        self.window = "learning"
        self.bad = None

    def capture(self):
        self.seen.append(self.raw.scene.state)
        self.pending.append(self.raw.scene.state)

    def drain(self):
        if len(self.pending) != trainer.DECIMATION: raise ValueError("Missing sample")
        self.pending.clear()
        self.windows[self.window] = {"max_closure_point_m": 0., "max_closure_axis_chord": 0.,
            "max_envelope_excess_nm": 0., "invalid_samples": 0,
            "min_height_m": -1., "nonfoot_contact_env_substeps": 100}
        if self.bad: self.windows[self.window][self.bad[0]] = self.bad[1]


class PhysicsGuardTests(unittest.TestCase):
    def raw(self):
        return types.SimpleNamespace(scene=FakeScene(), _physics_handles_decimation=False,
            common_step_counter=0, cfg=types.SimpleNamespace(sim=types.SimpleNamespace(dt=trainer.PHYSICS_DT_S), decimation=trainer.DECIMATION),
            _body_contact_sensors={"body": types.SimpleNamespace(cfg=types.SimpleNamespace(update_period=trainer.PHYSICS_DT_S))})

    def test_measures_after_update_all_substeps_and_allows_falling(self):
        raw = self.raw()
        metrics = FakeMetrics(raw)
        with trainer.PhysicalTrainingGuard(raw, metrics) as guard:
            for control_step in range(2):
                raw.common_step_counter = control_step
                for _ in range(trainer.DECIMATION): raw.scene.update(dt=trainer.PHYSICS_DT_S)
            guard.require_coverage(2)
        self.assertEqual(metrics.seen, list(range(1, 2*trainer.DECIMATION+1)))
        self.assertNotIn("update", vars(raw.scene))

    def test_rejects_loop_error_axis_motor_envelope_and_invalid_sample(self):
        for key, bad in [("max_closure_point_m", .000101), ("max_closure_axis_chord", math.radians(.101)),
                         ("max_envelope_excess_nm", .000011), ("invalid_samples", 1)]:
            raw = self.raw()
            metrics = FakeMetrics(raw)
            metrics.bad = key, bad
            with self.assertRaises(ValueError):
                with trainer.PhysicalTrainingGuard(raw, metrics):
                    for _ in range(trainer.DECIMATION): raw.scene.update(trainer.PHYSICS_DT_S)
            self.assertNotIn("update", vars(raw.scene))

    def test_partial_extra_wrong_dt_and_hidden_substeps_rejected(self):
        raw = self.raw()
        with trainer.PhysicalTrainingGuard(raw, FakeMetrics(raw)) as guard:
            for _ in range(trainer.DECIMATION-1): raw.scene.update(trainer.PHYSICS_DT_S)
            with self.assertRaises(ValueError): guard.require_coverage(1)
            raw.common_step_counter = 1
            with self.assertRaises(ValueError): raw.scene.update(trainer.PHYSICS_DT_S)
        raw = self.raw()
        with trainer.PhysicalTrainingGuard(raw, FakeMetrics(raw)):
            for _ in range(trainer.DECIMATION): raw.scene.update(trainer.PHYSICS_DT_S)
            with self.assertRaises(ValueError): raw.scene.update(trainer.PHYSICS_DT_S)
            with self.assertRaises(ValueError): raw.scene.update(.02)
        raw = self.raw()
        raw._physics_handles_decimation = True
        with self.assertRaises(ValueError): trainer.PhysicalTrainingGuard(raw, FakeMetrics(raw)).__enter__()

    def test_restores_preexisting_instance_override(self):
        raw = self.raw()
        replacement = lambda dt: None
        raw.scene.update = replacement
        with trainer.PhysicalTrainingGuard(raw, FakeMetrics(raw)): pass
        self.assertIs(raw.scene.update, replacement)


if __name__ == "__main__": unittest.main()
