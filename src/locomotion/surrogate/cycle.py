"""Judge the stored optimized cycles on the walking requirement cases in the surrogate.

    uv run python -m locomotion.surrogate.cycle --bank bank.npz --output /absolute/fresh/evaluation

``locomotion.amp.cycle`` builds the bank and holds the controller. ``--learning-rate`` above zero
turns on its joint-tracking correction. ``--action-delay`` and ``--condition`` are the stress
options of ``locomotion.surrogate.evaluate``. The surrogate gives design evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import locomotion.evaluation.runner as evaluation
import locomotion.randomization as rnd
from locomotion.amp.cycle import CycleController
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
from locomotion.surrogate.evaluate import STRESS_FIELDS, stress_condition

# The newest of the five 42-value frames holds the joint offsets from neutral; the command follows the frames.
JOINT_OFFSET, COMMAND = slice(174, 192), slice(210, 213)


def policy_for(controller, env, delay=0):
    """The case runner's policy: restart the controller with each case and send actions ``delay`` controls late."""
    late = []

    def policy(observation):
        if int(env.episode_steps[0]) == 0:
            controller.reset()
            late[:] = [controller.neutral] * delay
        row = observation[0].numpy()
        late.append(controller(row[COMMAND], row[JOINT_OFFSET] + controller.neutral))
        action = (late.pop(0) - controller.neutral) / env.cfg.action_scale_rad
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
    condition, stress = stress_condition(args.condition, rnd.nominal_draw())
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
    with np.load(args.bank, allow_pickle=False) as bank:
        controller = CycleController(
            bank["commands"],
            bank["targets"],
            bank["references"],
            env.neutral.numpy(),
            learning_rate=args.learning_rate,
        )
    judgement = evaluation.run_cases(
        env,
        policy_for(controller, env, args.action_delay),
        args.output / "evaluation",
        env.geometry_extrema_path,
        args.bank,
        case_ids=None if args.cases == "all" else args.cases.split(","),
        seed=args.seed,
        progress=lambda value: print(json.dumps(value), flush=True),
        motion_features=extract_features,
        capture_source=SurrogateCaptureSource,
    )
    env.close()
    summary = {
        "schema": "hexapod_surrogate_cycle_evaluation_v1",
        "scope": "CPU MuJoCo surrogate; design evidence. Native runs are the only acceptance evidence.",
        "bank": str(args.bank),
        "bank_sha256": sha(args.bank),
        "learning_rate": args.learning_rate,
        "action_delay_controls": args.action_delay,
        "condition": None if condition is None else {"stress": stress, "draw": condition.as_json()},
        "seed": args.seed,
        "repository": repository_state(),
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
