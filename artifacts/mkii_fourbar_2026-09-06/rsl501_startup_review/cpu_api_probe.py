#!/usr/bin/env python3
"""Exercise the actual RSL 5.0.1 API on a synthetic CPU environment, never Isaac.

Run with RSL 5.0.1 and its CPU dependencies on PYTHONPATH. Installed Spark SDK
sources are hash checked and loaded without its native package initializers.
The resulting report is API evidence only and cannot admit physical training.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]


def source(suffix):
    records = json.loads((EVIDENCE / "sources.json").read_text())["sources"]
    record, = [r for r in records if r["path"].endswith(suffix)]
    path = EVIDENCE / record["copy"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Captured source hash mismatch: {suffix}")
    return path, data.decode()


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def load_source(name, suffix, *, transform=None, namespace=None):
    path, text = source(suffix)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    if namespace:
        module.__dict__.update(namespace)
    tree = ast.parse(text)
    if transform:
        tree = transform(tree)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_lab_config_sources():
    """Only config utilities, with unavailable native array conversion disabled."""
    for name in ("isaaclab", "isaaclab.utils", "isaaclab_rl"):
        package(name)
    rsl = package("isaaclab_rl.rsl_rl")
    load_source("isaaclab.utils.string", "isaaclab/isaaclab/utils/string.py")

    class UnusedArrayConversions(dict):
        def __getitem__(self, key):
            raise AssertionError("Native array conversion is outside this CPU API probe")

    def remove_array_import(tree):
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ImportFrom) and node.module == "array")]
        return tree

    load_source("isaaclab.utils.dict", "isaaclab/isaaclab/utils/dict.py",
                transform=remove_array_import,
                namespace={"TENSOR_TYPES": UnusedArrayConversions(),
                           "TENSOR_TYPE_CONVERSIONS": UnusedArrayConversions()})
    load_source("isaaclab.utils.configclass", "isaaclab/isaaclab/utils/configclass.py")
    for stem in ("rnd_cfg", "symmetry_cfg", "rl_cfg", "utils"):
        module = load_source(f"isaaclab_rl.rsl_rl.{stem}", f"isaaclab_rl/isaaclab_rl/rsl_rl/{stem}.py")
        for name, value in vars(module).items():
            if name.startswith("RslRl") or name == "handle_deprecated_rsl_rl_cfg":
                setattr(rsl, name, value)
    return rsl


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite API probe evidence")
    import torch
    import gymnasium as gym
    import rsl_rl
    from rsl_rl.runners import OnPolicyRunner
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if importlib.metadata.version("rsl-rl-lib") != "5.0.1":
        raise ValueError("This probe covers exactly RSL-RL 5.0.1")
    matches = {}
    for suffix in ("runners/on_policy_runner.py", "algorithms/ppo.py", "utils/logger.py",
                   "models/mlp_model.py", "utils/utils.py"):
        _, captured = source("rsl_rl/" + suffix)
        actual = (Path(rsl_rl.__file__).parent / suffix).read_bytes()
        matches[suffix] = hashlib.sha256(actual).hexdigest()
        if actual != captured.encode():
            raise ValueError(f"RSL package differs from captured installed Spark bytes: {suffix}")

    lab_rsl = load_lab_config_sources()
    trainer = load_file("mkii_cpu_api_trainer", ROOT / "isaaclab/train_mkii_fourbar.py")
    cfg_module = load_file("mkii_cpu_api_cfg", ROOT / "packages/hexapod_env/hexapod_env/ppo_cfg.py")

    # The real wrapper is exercised; only its type-check base is a synthetic CPU
    # class. None of DirectRLEnv, Kit, PhysX, CUDA or the robot task is imported.
    env_types = package("isaaclab.envs")
    class SyntheticDirectEnv:
        pass
    env_types.DirectRLEnv = SyntheticDirectEnv
    env_types.ManagerBasedEnv = type("UnusedManagerBasedEnv", (), {})
    env_types.ManagerBasedRLEnv = type("UnusedManagerBasedRLEnv", (), {})
    wrapper_module = load_source("isaaclab_rl.rsl_rl.vecenv_wrapper",
                                "isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py")

    class ToyEnv(SyntheticDirectEnv):
        num_envs, num_actions, max_episode_length = 8, 18, 19
        device = "cpu"
        def __init__(self):
            self.cfg = types.SimpleNamespace(is_finite_horizon=False)
            self.single_action_space = gym.spaces.Box(-1., 1., shape=(18,))
            self.episode_length_buf = torch.zeros(8, dtype=torch.long)
            self.state = torch.zeros(8, 18)
            self.steps = 0
            self.reset()
        @property
        def unwrapped(self):
            return self
        def _get_observations(self):
            return {"policy": torch.cat((self.state, self.state.square(),
                                        torch.ones(8, 27) * .1), dim=-1)}
        def reset(self):
            self.episode_length_buf.zero_()
            self.state = torch.randn(8, 18) * .01
            return self._get_observations(), {}
        def step(self, actions):
            assert actions.device.type == "cpu"
            self.steps += 1
            self.episode_length_buf += 1
            self.state = .95 * self.state + .05 * actions
            rewards = 1. - self.state.square().mean(-1) - .1 * actions.square().mean(-1)
            terminated = torch.zeros(8, dtype=torch.bool)
            truncated = self.episode_length_buf >= self.max_episode_length
            self.episode_length_buf[truncated] = 0
            self.state[truncated] = 0.
            return self._get_observations(), rewards, terminated, truncated, {
                "log": {"Episode_Reward/synthetic_only": rewards.mean()}}

    def configuration():
        cfg = cfg_module.HexapodPPORunnerCfg()
        cfg.device, cfg.logger, cfg.save_interval = "cpu", "tensorboard", 10
        return cfg

    raw = configuration().to_dict()
    unadapted_error = None
    try:
        OnPolicyRunner(wrapper_module.RslRlVecEnvWrapper(ToyEnv()), copy.deepcopy(raw), device="cpu")
    except TypeError as error:
        unadapted_error = str(error)
    if not unadapted_error or "unexpected keyword argument 'stochastic'" not in unadapted_error:
        raise AssertionError(f"Did not reproduce the actual deprecated-key failure: {unadapted_error}")
    adapted = trainer.runner_config_dict(configuration(), "5.0.1")
    expected_distribution = {"class_name": "GaussianDistribution", "init_std": .15, "std_type": "scalar"}
    assert adapted["actor"]["distribution_cfg"] == expected_distribution
    assert adapted["critic"]["distribution_cfg"] is None
    assert all(adapted[key]["obs_normalization"] for key in ("actor", "critic"))

    # Execute the exact nested production checkpoint subclass; no duplicated
    # implementation of save/load/next-iteration semantics in this fixture.
    tree = ast.parse((ROOT / "isaaclab/train_mkii_fourbar.py").read_text())
    guarded, = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GuardedRunner"]
    namespace = dict(vars(trainer), OnPolicyRunner=OnPolicyRunner,
                     contract={"api_probe_only": True, "physical_admission": False})
    exec(compile(ast.Module(body=[guarded], type_ignores=[]), "production_GuardedRunner", "exec"), namespace)
    GuardedRunner = namespace["GuardedRunner"]
    history = []
    def instrument(runner, expected_start):
        original_log = getattr(runner, "_api_original_log", runner.logger.log)
        runner._api_original_log = original_log
        def checked_log(**values):
            assert values["start_it"] == expected_start
            assert {"value", "surrogate", "entropy"}.issubset(values["loss_dict"])
            assert all(torch.isfinite(torch.tensor(v)) for v in values["loss_dict"].values())
            trainer.algorithm_state_digest(runner)
            original_log(**values)
            history.append({"iteration": values["it"], "start_iteration": values["start_it"],
                            "losses": values["loss_dict"], "collect_seconds": values["collect_time"],
                            "learn_seconds": values["learn_time"]})
        runner.logger.log = checked_log

    with tempfile.TemporaryDirectory(prefix="hexapod_rsl501_cpu_api_") as directory:
        output = Path(directory)
        wrapped = wrapper_module.RslRlVecEnvWrapper(ToyEnv())
        runner = GuardedRunner(wrapped, copy.deepcopy(adapted), log_dir=str(output / "scratch"), device="cpu")
        before = trainer.parameters_digest(runner)
        instrument(runner, 0)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 1 and wrapped.env.steps == 48
        after = trainer.parameters_digest(runner)
        assert before != after
        checkpoint = output / "synthetic_api_checkpoint.pt"
        runner.save(str(checkpoint))
        before_roundtrip = trainer.algorithm_state_digest(runner)
        checkpoint_sha256 = trainer.digest(checkpoint)
        sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text())
        assert sidecar["next_iteration"] == 2 and sidecar["checkpoint_sha256"] == checkpoint_sha256
        infos = runner.load(str(checkpoint), strict=True, map_location="cpu")
        assert infos["next_iteration"] == 2
        assert trainer.algorithm_state_digest(runner) == before_roundtrip
        assert trainer.parameters_digest(runner) == after
        policy = runner.get_inference_policy(device="cpu")
        with torch.inference_mode():
            observations = wrapped.get_observations()
            actions = policy(observations)
            assert actions.shape == (8, 18) and torch.isfinite(actions).all()
            assert torch.equal(actions, policy(observations))
        runner.current_learning_iteration = infos["next_iteration"]
        instrument(runner, 2)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 2 and wrapped.env.steps == 72
        same_runner_after = trainer.parameters_digest(runner)
        assert same_runner_after != after
        runner.save(str(output / "synthetic_resumed_checkpoint.pt"))
        resume_checkpoint = output / "synthetic_resumed_checkpoint.pt"
        resumed_state = trainer.algorithm_state_digest(runner)
        fresh = GuardedRunner(wrapper_module.RslRlVecEnvWrapper(ToyEnv()), copy.deepcopy(adapted),
                              log_dir=str(output / "resume"), device="cpu")
        resume_infos = fresh.load(str(resume_checkpoint), strict=True, map_location="cpu")
        assert trainer.algorithm_state_digest(fresh) == resumed_state
        fresh.current_learning_iteration = resume_infos["next_iteration"]
        instrument(fresh, 3)
        fresh.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        assert fresh.current_learning_iteration == 3 and fresh.env.env.steps == 24
        assert trainer.parameters_digest(fresh) != same_runner_after
        assert [item["iteration"] for item in history] == [0, 1, 2, 3]
        full_state = fresh.alg.save()
        assert full_state["optimizer_state_dict"]["state"]
        normalization_keys = [name for name in full_state["actor_state_dict"] if "normaliz" in name]
        assert normalization_keys

    report = {"schema": "hexapod.rsl501_cpu_api_probe.v1", "pass": True,
              "physical_robot_ppo_started": False, "physical_admission": False,
              "isaac_sim_or_cuda_imported": any(name == "pxr" or name.startswith("omni.") for name in sys.modules)
                  or torch.cuda.is_initialized(),
              "device": "cpu", "environment": "synthetic linear recurrence; no robot or contacts",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trainer_sha256": hashlib.sha256((ROOT / "isaaclab/train_mkii_fourbar.py").read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256((ROOT / "packages/hexapod_env/hexapod_env/ppo_cfg.py").read_bytes()).hexdigest(),
              "sdk_sources_manifest_sha256": hashlib.sha256((EVIDENCE / "sources.json").read_bytes()).hexdigest(),
              "torch_version": torch.__version__,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("rsl-rl-lib", "tensordict", "gymnasium", "tensorboard", "packaging")},
              "rsl_byte_matches_to_spark": matches,
              "unadapted_constructor_error": unadapted_error,
              "raw_config": raw, "adapted_config": adapted,
              "policy_before_sha256": before, "policy_after_sha256": after,
              "synthetic_checkpoint_sha256": checkpoint_sha256,
              "checkpoint_roundtrip_pass": True, "same_runner_resume_pass": True,
              "fresh_runner_resume_pass": True,
              "normalization_state_keys": normalization_keys,
              "logger_history": history,
              "limitations": ["Synthetic CPU environment is not a physical robot PPO run or admission.",
                  "Captured config/wrapper source is loaded without Isaac package initializers.",
                  "Unused native array conversion imports are replaced with fail-on-use dictionaries.",
                  "DirectRLEnv type check uses a synthetic class; Isaac/PhysX lifecycle is not exercised.",
                  "The local dependencies are recorded, not claimed identical to the Spark dependency stack.",
                  "Temporary synthetic checkpoints and event logs are removed after successful checks."]}
    assert report["isaac_sim_or_cuda_imported"] is False
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU_API_PROBE_PASS", args.report)


if __name__ == "__main__":
    main()
