"""CPU contract tests for the recovery model-only RSL-RL load wrapper."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "train_model_only_resume.py"
SPEC = importlib.util.spec_from_file_location("train_model_only_resume", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
model_only = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_only)


class FakeDistribution:
    def __init__(self):
        self.std_param = torch.nn.Parameter(torch.full((18,), 0.10))

    def update(self, _mlp_output):
        return self.std_param.detach().clone()


class FakeModel:
    def __init__(self, stochastic: bool):
        self.obs_normalizer = SimpleNamespace(until=None)
        self.distribution = FakeDistribution() if stochastic else None


class FakeAlgorithm:
    def __init__(self):
        self._raw_actor = FakeModel(stochastic=True)
        self._raw_critic = FakeModel(stochastic=False)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.Adam([parameter], lr=5e-5)

    def update(self):
        self._raw_actor.distribution.std_param.data.fill_(0.50)
        return {"surrogate": 1.0}


class FakeRunner:
    def __init__(self):
        self.alg = FakeAlgorithm()
        self.current_learning_iteration = 0


class ModelOnlyResumeTest(unittest.TestCase):
    def test_selective_load_ignores_optimizer_and_iteration_and_bounds_std(self):
        captured = {}

        def original_load(
            runner,
            path,
            load_cfg=None,
            strict=True,
            map_location=None,
        ):
            captured.update(
                path=path,
                load_cfg=load_cfg,
                strict=strict,
                map_location=map_location,
            )
            runner.alg._raw_actor.distribution.std_param.data.fill_(0.309)
            if load_cfg.get("iteration"):
                runner.current_learning_iteration = 200
            if load_cfg.get("optimizer"):
                raise AssertionError("optimizer must not be loaded")
            return {"source": "test"}

        runner = FakeRunner()
        wrapped = model_only.make_model_only_load(original_load, 0.06, 0.14)
        infos = wrapped(runner, "model.pt")
        self.assertEqual(infos, {"source": "test"})
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(
            captured["load_cfg"],
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        self.assertEqual(len(runner.alg.optimizer.state), 0)
        self.assertEqual(runner.alg._raw_actor.obs_normalizer.until, 0)
        self.assertEqual(runner.alg._raw_critic.obs_normalizer.until, 0)
        torch.testing.assert_close(
            runner.alg._raw_actor.distribution.std_param,
            torch.full((18,), 0.14),
        )

        runner.alg._raw_actor.distribution.std_param.data.fill_(0.50)
        effective_std = runner.alg._raw_actor.distribution.update(None)
        torch.testing.assert_close(effective_std, torch.full((18,), 0.14))

        result = runner.alg.update()
        self.assertEqual(result, {"surrogate": 1.0})
        torch.testing.assert_close(
            runner.alg._raw_actor.distribution.std_param,
            torch.full((18,), 0.14),
        )

    def test_invalid_std_bounds_are_rejected(self):
        runner = FakeRunner()
        with self.assertRaises(ValueError):
            model_only.configure_loaded_runner(runner, 0.2, 0.1)

    def test_wrapper_rejects_nested_load_policy(self):
        def original_load(
            runner,
            path,
            load_cfg=None,
            strict=True,
            map_location=None,
        ):
            return None

        wrapped = model_only.make_model_only_load(original_load, 0.06, 0.14)
        with self.assertRaises(RuntimeError):
            wrapped(FakeRunner(), "model.pt", load_cfg={"actor": True})

    def test_stock_train_sibling_directory_is_temporarily_importable(self):
        train_script = Path("/tmp/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py")
        original_argv = sys.argv
        original_sys_path = list(sys.path)

        def assert_context(path, run_name):
            self.assertEqual(path, str(train_script))
            self.assertEqual(run_name, "__main__")
            self.assertEqual(sys.path[0], str(train_script.parent))
            self.assertEqual(sys.argv, [str(train_script), "--task", "Recovery-v0"])

        with mock.patch.object(model_only.runpy, "run_path", side_effect=assert_context):
            model_only.run_training_script(train_script, ["--task", "Recovery-v0"])
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(sys.path, original_sys_path)

    def test_import_hook_defers_partial_runners_module_and_fails_closed(self):
        partial_module = ModuleType("rsl_rl.runners")
        sentinel = object()
        previous = sys.modules.get("rsl_rl.runners", sentinel)
        sys.modules["rsl_rl.runners"] = partial_module

        def nested_import(name, globals=None, locals=None, fromlist=(), level=0):
            return partial_module

        hook = model_only.make_recovery_import_hook(nested_import, 0.06, 0.14)
        try:
            # A nested import sees the partially initialized module and must
            # defer rather than raising the remote-smoke failure.
            self.assertIs(hook("rsl_rl.algorithms", fromlist=("PPO",)), partial_module)
            self.assertFalse(hasattr(partial_module, "OnPolicyRunner"))

            # At the stock trainer's actual import boundary, remaining
            # unpatched is unsafe and must fail before runner.load is possible.
            with self.assertRaisesRegex(RuntimeError, "import boundary"):
                hook("rsl_rl.runners", fromlist=("OnPolicyRunner",))
        finally:
            if previous is sentinel:
                del sys.modules["rsl_rl.runners"]
            else:
                sys.modules["rsl_rl.runners"] = previous


if __name__ == "__main__":
    unittest.main()
