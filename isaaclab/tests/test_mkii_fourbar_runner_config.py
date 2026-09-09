"""RSL 5 constructor regression using the captured, installed Lab adapter."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts/mkii_fourbar_2026-09-06/rsl501_startup_review"
spec = importlib.util.spec_from_file_location("fourbar_runner_config_test", ROOT / "isaaclab/train_mkii_fourbar.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def recorded_source(suffix):
    records = json.loads((EVIDENCE / "sources.json").read_text())["sources"]
    record, = [r for r in records if r["path"].endswith(suffix)]
    data = (EVIDENCE / record["copy"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise AssertionError("Installed source evidence hash changed")
    return data.decode()


class RecordedConfig:
    """The actual emitted model dictionaries, with configclass's mutable attributes."""
    def __init__(self):
        model = {"class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "elu",
                 "obs_normalization": True, "distribution_cfg": None, "stochastic": {},
                 "init_noise_std": {}, "noise_std_type": "scalar", "state_dependent_std": False}
        self.actor = types.SimpleNamespace(**copy.deepcopy(model))
        self.critic = types.SimpleNamespace(**copy.deepcopy(model))
        self.actor.distribution_cfg = {"class_name": "GaussianDistribution", "init_std": .15, "std_type": "scalar"}

    def to_dict(self):
        return {"actor": copy.deepcopy(vars(self.actor)), "critic": copy.deepcopy(vars(self.critic)),
                "num_steps_per_env": 24, "check_for_nan": False}


class RunnerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the complete installed adapter, not a reimplementation of its removal rules.
        cls.sdk_namespace = {"__name__": "captured_isaaclab_rsl_utils",
            "version": types.SimpleNamespace(parse=lambda text: tuple(int(v) for v in text.split(".")))}
        adapter = ast.parse(recorded_source("isaaclab_rl/isaaclab_rl/rsl_rl/utils.py"))
        # Packaging is an SDK dependency, absent from the stdlib/Torch test environment.
        # Substitute only parsing of these fixed numeric versions; all adapter bodies are exact.
        adapter.body = [n for n in adapter.body if not (isinstance(n, ast.ImportFrom) and n.module == "packaging")]
        exec(compile(adapter, "captured_isaaclab_rsl_utils", "exec"), cls.sdk_namespace)
        tree = ast.parse(recorded_source("rsl_rl/models/mlp_model.py"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLPModel")
        constructor = copy.deepcopy(next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
        constructor.body = [ast.Pass()]
        namespace = {}
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                constructor], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), "installed_mlp_signature", "exec"), namespace)
        cls.signature = inspect.signature(namespace["__init__"])

    def sdk(self):
        module = types.ModuleType("isaaclab_rl.rsl_rl")
        module.handle_deprecated_rsl_rl_cfg = self.sdk_namespace["handle_deprecated_rsl_rl_cfg"]
        module.RslRlMLPModelCfg = type("RslRlMLPModelCfg", (), {})
        return module

    def test_actual_adapter_removes_rejected_keywords_preserving_policy_and_normalizers(self):
        cfg = RecordedConfig()
        before = cfg.to_dict()
        self.assertIsNone(self.signature.parameters.get("kwargs"))
        for name in ("actor", "critic"):
            kwargs = {k: v for k, v in before[name].items() if k != "class_name"}
            with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'stochastic'"):
                self.signature.bind(None, None, None, None, None, **kwargs)
        # Restore only this module key; never remove unrelated native imports from sys.modules.
        prior = sys.modules.get("isaaclab_rl.rsl_rl")
        sys.modules["isaaclab_rl.rsl_rl"] = self.sdk()
        try:
            result = trainer.runner_config_dict(cfg, "5.0.1")
        finally:
            if prior is None:
                sys.modules.pop("isaaclab_rl.rsl_rl", None)
            else:
                sys.modules["isaaclab_rl.rsl_rl"] = prior
        for name in ("actor", "critic"):
            kwargs = {k: v for k, v in result[name].items() if k != "class_name"}
            self.signature.bind(None, None, None, None, None, **kwargs)
            for field in ("distribution_cfg", "obs_normalization", "hidden_dims", "activation"):
                self.assertEqual(result[name][field], before[name][field])
            self.assertEqual(result[name], vars(getattr(cfg, name)))
        self.assertTrue(result["check_for_nan"])
        self.assertEqual(result["num_steps_per_env"], 24)

    def test_unsupported_version_fails_before_sdk_import_or_config_mutation(self):
        for version in ("4.0.0", "5.0.0", "5.0.2", None):
            cfg = RecordedConfig()
            before = cfg.to_dict()
            with self.assertRaisesRegex(ValueError, "5.0.1"):
                trainer.runner_config_dict(cfg, version)
            self.assertEqual(cfg.to_dict(), before)

    def test_main_uses_checked_version_adapter_before_constructing_runner(self):
        tree = ast.parse((ROOT / "isaaclab/train_mkii_fourbar.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "agent_dict" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        assignment = assignments[0]
        self.assertEqual(ast.unparse(assignment.value), "runner_config_dict(agent_cfg, report['rsl_rl_version'])")
        runner = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "GuardedRunner")
        self.assertLess(assignment.lineno, runner.lineno)


class ReloadBufferTests(unittest.TestCase):
    def algorithm(self):
        actor = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Identity())
        actor[1].register_buffer("normal", torch.tensor([2., 3.], dtype=torch.float64))
        with torch.inference_mode():
            actor[1].register_buffer("rollout", torch.tensor([.2, .3], dtype=torch.float64))
        critic = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
        return types.SimpleNamespace(actor=actor, critic=critic, rnd=None, optimizer=optimizer)

    def test_real_inference_buffer_reload_failure_is_repaired_without_state_change(self):
        algorithm = self.algorithm()
        module = algorithm.actor[1]
        before = trainer.finite_state_digest(algorithm.actor.state_dict())
        state = {name: value.clone() for name, value in algorithm.actor.state_dict().items()}
        normal = module.normal
        rollout = module.rollout
        parameter_ids = [id(p) for group in algorithm.optimizer.param_groups for p in group["params"]]
        optimizer_before = trainer.finite_state_digest(algorithm.optimizer.state_dict())
        with self.assertRaisesRegex(RuntimeError, "Inplace update to inference tensor"):
            algorithm.actor.load_state_dict(state, strict=True)
        self.assertEqual(trainer.prepare_algorithm_buffers_for_load(algorithm), ["actor.1.rollout"])
        self.assertIs(module.normal, normal)
        self.assertIsNot(module.rollout, rollout)
        self.assertFalse(module.rollout.is_inference())
        self.assertEqual(module.rollout.dtype, rollout.dtype)
        self.assertEqual(module.rollout.device, rollout.device)
        self.assertEqual(module.rollout.shape, rollout.shape)
        self.assertEqual(trainer.finite_state_digest(algorithm.actor.state_dict()), before)
        self.assertEqual(trainer.finite_state_digest(algorithm.optimizer.state_dict()), optimizer_before)
        self.assertEqual([id(p) for group in algorithm.optimizer.param_groups for p in group["params"]], parameter_ids)
        algorithm.actor.load_state_dict(state, strict=True)
        self.assertEqual(trainer.prepare_algorithm_buffers_for_load(algorithm), [])

    def test_clones_are_normal_even_if_helper_called_with_inference_enabled(self):
        algorithm = self.algorithm()
        with torch.inference_mode():
            trainer.prepare_algorithm_buffers_for_load(algorithm)
        self.assertFalse(algorithm.actor[1].rollout.is_inference())
        # Actual loading and continued training remain outside inference mode.
        (algorithm.actor(torch.ones(1, 2)).sum() + algorithm.critic(torch.ones(1, 2)).sum()).backward()
        algorithm.optimizer.step()
        self.assertTrue(all(not tensor.is_inference() for state in algorithm.optimizer.state.values()
                            for tensor in state.values() if isinstance(tensor, torch.Tensor)))


if __name__ == "__main__":
    unittest.main()
