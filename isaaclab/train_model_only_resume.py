#!/usr/bin/env python3
"""Run Isaac Lab RSL-RL training with a strict model-only checkpoint load.

The stock Isaac Lab entrypoint resumes optimizer moments, learning rate, and
iteration whenever ``--resume`` is supplied. This wrapper intercepts that one
load and requests only actor/critic tensors. It also freezes the transformed
observation normalizers and enforces a bounded scalar Gaussian action std after
every PPO update.

The import hook intentionally waits for Isaac Lab's training script to import
RSL-RL itself, preserving AppLauncher/import ordering.
"""

from __future__ import annotations

import argparse
import builtins
import inspect
import math
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


DEFAULT_TRAIN_SCRIPT = Path(
    os.environ.get(
        "ISAACLAB_RSL_RL_TRAIN_SCRIPT",
        "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
    )
)


def _scalar_std_parameter(distribution: Any) -> Any:
    parameter = getattr(distribution, "std_param", None)
    if parameter is None:
        raise RuntimeError(
            "Recovery training requires a scalar Gaussian distribution with std_param"
        )
    return parameter


def configure_loaded_runner(runner: Any, std_min: float, std_max: float) -> None:
    """Freeze RMS state and install effective plus parameter-level std bounds."""

    if not (math.isfinite(std_min) and math.isfinite(std_max)):
        raise ValueError("std bounds must be finite")
    if std_min <= 0.0 or std_min > std_max:
        raise ValueError(f"Invalid std bounds [{std_min}, {std_max}]")

    algorithm = runner.alg
    actor = getattr(algorithm, "_raw_actor", getattr(algorithm, "actor", None))
    critic = getattr(algorithm, "_raw_critic", getattr(algorithm, "critic", None))
    if actor is None or critic is None:
        raise RuntimeError("Unable to locate loaded RSL-RL actor and critic")

    for name, model in (("actor", actor), ("critic", critic)):
        normalizer = getattr(model, "obs_normalizer", None)
        if normalizer is None or not hasattr(normalizer, "until"):
            raise RuntimeError(f"{name} does not expose an empirical observation normalizer")
        # RSL-RL's update() returns immediately once count >= until. Zero is a
        # persistent freeze even after alg.train_mode() recursively switches
        # modules back into training mode.
        normalizer.until = 0

    distribution = getattr(actor, "distribution", None)
    if distribution is None:
        raise RuntimeError("Loaded actor has no stochastic output distribution")
    import torch

    std_parameter = _scalar_std_parameter(distribution)
    with torch.no_grad():
        std_parameter.clamp_(std_min, std_max)

    # RSL-RL 5.0.1 does not expose std_range. Clamp before every Normal is
    # constructed so an optimizer step between PPO minibatches cannot create a
    # negative or out-of-contract scalar std.
    original_distribution_update = distribution.update

    def bounded_distribution_update(*args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            _scalar_std_parameter(distribution).clamp_(std_min, std_max)
        return original_distribution_update(*args, **kwargs)

    distribution.update = bounded_distribution_update

    original_update = algorithm.update

    def bounded_update(*args: Any, **kwargs: Any) -> Any:
        result = original_update(*args, **kwargs)
        with torch.no_grad():
            _scalar_std_parameter(distribution).clamp_(std_min, std_max)
        return result

    algorithm.update = bounded_update


def make_model_only_load(
    original_load: Callable[..., Any], std_min: float, std_max: float
) -> Callable[..., Any]:
    """Wrap ``OnPolicyRunner.load`` with explicit actor/critic-only semantics."""

    signature = inspect.signature(original_load)
    if "load_cfg" not in signature.parameters:
        raise RuntimeError(
            "Installed RSL-RL OnPolicyRunner.load lacks selective load_cfg support"
        )

    def model_only_load(
        runner: Any,
        path: str,
        load_cfg: dict[str, bool] | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> Any:
        if load_cfg is not None:
            raise RuntimeError(
                "Recovery wrapper owns load_cfg; a second selective-load policy was supplied"
            )
        initial_iteration = runner.current_learning_iteration
        optimizer = runner.alg.optimizer
        initial_optimizer_state_count = len(optimizer.state)
        infos = original_load(
            runner,
            path,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=strict,
            map_location=map_location,
        )
        if runner.current_learning_iteration != initial_iteration:
            raise RuntimeError("Model-only load unexpectedly changed the learning iteration")
        if len(optimizer.state) != initial_optimizer_state_count:
            raise RuntimeError("Model-only load unexpectedly changed fresh optimizer state")
        configure_loaded_runner(runner, std_min, std_max)
        print(
            "[RECOVERY] Loaded actor+critic only; optimizer/iteration ignored; "
            f"observation RMS frozen; scalar action std bounded to [{std_min}, {std_max}]"
        )
        return infos

    return model_only_load


def patch_runners_module(module: ModuleType, std_min: float, std_max: float) -> bool:
    """Patch a fully initialized runners module; tolerate nested partial imports."""

    runner_class = getattr(module, "OnPolicyRunner", None)
    if runner_class is None:
        return False
    if getattr(runner_class, "_hexapod_model_only_patched", False):
        return True
    runner_class.load = make_model_only_load(runner_class.load, std_min, std_max)
    runner_class._hexapod_model_only_patched = True
    return True


def make_recovery_import_hook(
    original_import: Callable[..., Any], std_min: float, std_max: float
) -> Callable[..., Any]:
    """Build the deferred import hook around an injected import function."""

    def import_with_recovery_patch(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        module = original_import(name, globals, locals, fromlist, level)
        runners_module = sys.modules.get("rsl_rl.runners")
        patched = False
        if runners_module is not None:
            patched = patch_runners_module(runners_module, std_min, std_max)
        # The stock trainer imports OnPolicyRunner at this exact boundary. A
        # missing patch here must abort before an unsafe full-state load can be
        # reached. During nested imports the module may legitimately exist
        # without having defined/exported OnPolicyRunner yet, so those calls
        # merely defer patching.
        requests_runner = (
            name == "rsl_rl.runners"
            and bool(fromlist)
            and "OnPolicyRunner" in fromlist
        )
        if requests_runner and not patched:
            raise RuntimeError(
                "Recovery import hook reached the OnPolicyRunner import boundary "
                "without installing the model-only load patch"
            )
        return module

    return import_with_recovery_patch


def install_deferred_rsl_patch(std_min: float, std_max: float) -> Callable[..., Any]:
    """Patch RSL-RL at its first import and return the original import function."""

    original_import = builtins.__import__

    builtins.__import__ = make_recovery_import_hook(
        original_import, std_min, std_max
    )
    return original_import


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--std-min", type=float, default=0.06)
    parser.add_argument("--std-max", type=float, default=0.14)
    return parser.parse_known_args(argv)


def run_training_script(train_script: Path, train_args: list[str]) -> None:
    """Execute the stock entrypoint with its sibling imports resolvable."""

    original_argv = sys.argv
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(train_script.parent))
        sys.argv = [str(train_script), *train_args]
        runpy.run_path(str(train_script), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def main() -> None:
    args, train_args = parse_wrapper_args(sys.argv[1:])
    train_script = args.train_script.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if "--resume" not in train_args:
        raise ValueError("Model-only recovery launch requires the downstream --resume flag")
    if "--task" not in train_args and not any(value.startswith("--task=") for value in train_args):
        raise ValueError("Model-only recovery launch requires an explicit --task")

    original_import = install_deferred_rsl_patch(args.std_min, args.std_max)
    try:
        run_training_script(train_script, train_args)
    finally:
        builtins.__import__ = original_import


if __name__ == "__main__":
    main()
