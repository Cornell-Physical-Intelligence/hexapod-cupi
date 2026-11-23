"""Judge the stored optimized cycles on the walking requirement cases in the surrogate.

    uv run python -m locomotion.surrogate.cycle --bank bank.npz --output /absolute/fresh/evaluation

``locomotion.amp.cycle`` builds the bank and holds the controller. ``--learning-rate`` above zero
turns on its joint-tracking correction. ``--action-delay`` and ``--condition`` are the stress
options of ``locomotion.surrogate.evaluate``. The surrogate gives design evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch

import locomotion.evaluation.runner as evaluation
import locomotion.robots.randomization as rnd
from locomotion.amp import cycle, replay_native
from locomotion.amp.features import extract_features
from locomotion.surrogate.capture import SurrogateCaptureSource
from locomotion.surrogate.env import (
    DEFAULT_GEOMETRY,
    DEFAULT_MODEL,
    DEFAULT_STANCE,
    SurrogateConfig,
    SurrogateEnv,
    repository_state,
    sha,
)
from locomotion.surrogate.evaluate import STRESS_FIELDS, source_identity, stress_condition

# The newest of the five 42-value frames holds the joint offsets from neutral; the command follows the frames.
JOINT_OFFSET, COMMAND = slice(174, 192), slice(210, 213)


def policy_for(controller, env, delay=None):
    """The case runner's policy: restart the controller with each case and send actions ``delay`` controls late."""
    controller_delay = getattr(controller, "delay", None)
    if delay is None:
        delay = 0 if controller_delay is None else controller_delay
    if type(delay) is not int or delay < 0:
        raise ValueError("Action delay counts whole controls from zero")
    if controller_delay is not None and delay != controller_delay:
        raise ValueError("Policy delay differs from controller delay")
    late = deque()

    def policy(observation):
        if int(env.episode_steps[0]) == 0:
            controller.reset()
            late.clear()
        row = observation[0].numpy()
        late.append(controller(row[COMMAND], row[JOINT_OFFSET] + controller.neutral))
        target = late.popleft() if len(late) > delay else controller.neutral
        action = (target - controller.neutral) / env.cfg.action_scale_rad
        return torch.as_tensor(action, dtype=torch.float32)[None]

    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory.")
    parser.add_argument(
        "--cases", default="all", help="all, or comma-separated requirement case ids."
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--action-delay", type=int, default=0)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=", ".join(STRESS_FIELDS),
    )
    args = parser.parse_args(argv)
    if args.action_delay < 0:
        parser.error("--action-delay counts whole controls from zero")
    if not np.isfinite(args.learning_rate) or args.learning_rate < 0:
        parser.error("--learning-rate must be finite and nonnegative")
    condition, stress = stress_condition(args.condition, rnd.nominal_draw())
    bank_data = args.bank.read_bytes()
    bank_sha = hashlib.sha256(bank_data).hexdigest()
    identity = source_identity()
    identity["source_files"].update(
        {"amp/cycle.py": sha(cycle.__file__), "amp/replay_native.py": sha(replay_native.__file__)}
    )
    with np.load(io.BytesIO(bank_data), allow_pickle=False) as bank:
        arrays = [bank[name] for name in ("commands", "targets", "references")]
        provenance = {
            name: bank[name].tolist()
            for name in ("trace_sha256", "admission_sha256", "model_sha256")
            if name in bank
        }
    if provenance and provenance.get("model_sha256") != identity["model_sha256"]:
        parser.error("Bank model differs from the surrogate's approved model")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = SurrogateEnv(
        SurrogateConfig(
            num_envs=1, seed=args.seed, episode_seconds=60.0, record_motion_features=True
        ),
        None,
        DEFAULT_MODEL,
        DEFAULT_GEOMETRY,
        args.output / "native",
        reference_metadata=json.loads(DEFAULT_STANCE.read_text()),
        threads=0,
        condition=condition,
    )
    try:
        controller = cycle.CycleController(
            *arrays,
            env.neutral.numpy(),
            learning_rate=args.learning_rate,
            action_delay=args.action_delay,
        )
        # The runner hashes the same bytes the controller loaded, even if the input path changes.
        with tempfile.NamedTemporaryFile() as snapshot:
            snapshot.write(bank_data)
            snapshot.flush()
            judgement = evaluation.run_cases(
                env,
                policy_for(controller, env),
                args.output / "evaluation",
                env.geometry_extrema_path,
                snapshot.name,
                case_ids=None if args.cases == "all" else args.cases.split(","),
                seed=args.seed,
                progress=lambda value: print(json.dumps(value), flush=True),
                motion_features=extract_features,
                capture_source=SurrogateCaptureSource,
            )
    finally:
        env.close()
    summary = {
        "schema": "hexapod_surrogate_cycle_evaluation_v1",
        "scope": "CPU MuJoCo surrogate; design evidence. Native runs are the only acceptance evidence.",
        "bank": str(args.bank),
        "bank_sha256": bank_sha,
        "bank_declared_provenance": provenance or None,
        "learning_rate": args.learning_rate,
        "action_delay_controls": args.action_delay,
        "condition": None if condition is None else {"stress": stress, "draw": condition.as_json()},
        "seed": args.seed,
        "repository": repository_state(),
        "identity": identity,
        "requirements_sha256": judgement["requirements_sha256"],
        "judged_cases": judgement["judged_cases"],
        "failed_cases": judgement["failed_cases"],
        "acquisition_failure": judgement["failure"],
        "pass": judgement["pass"],
    }
    (args.output / "cycle_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("judged_cases", "failed_cases", "pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
