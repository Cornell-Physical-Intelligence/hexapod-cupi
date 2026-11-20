"""Diagnostic: the 96 required cases with the leg-stand fitted dynamics applied in the surrogate.

NOT a judge result. The judges run at canonical dynamics; this asks the separate question "how
does a checkpoint behave in the simulator the leg stand says is closer to the real leg". The
repeating job: one checkpoint + one set of fitted flags -> the unchanged scorer's verdicts under
those dynamics. Inputs: checkpoints (name=path) and the ranges recorded in
records/leg_stand_20261109/actuator_params.json (the same flags fit1 trained with). Check: the
repository's evaluator, scorer, bounds and capture checks, all untouched; this file only injects
the 96 cases (full96_suite) and wraps ``SurrogateEnv`` construction, as
records/curriculum_20261109/adapter/judges/sim2real_robust_eval.py does. Record: per checkpoint a
folder with the evaluator's own surrogate_summary.json plus fitted_draws.json (ranges, the draw
each episode received, checkpoint sha256).

Matching: every checkpoint runs the same seed, and the environment draws per (seed, row, episode),
so case k sees the same drawn world for every checkpoint.

Usage (from the hexapod-cupi checkout):
  uv run python fitted_eval.py OUT_ROOT name=checkpoint.pt [name=ckpt ...]
        [--axes delay,kd,toe_friction,joint_friction,armature] [--match TEXT ...] [--seed N]
        [--motor-check canonical|drawn] [--predictor predictor.pt [--predict-controls drawn|N]]

--axes none is the contact-only control (touchdown model off, no draws). The evaluator's capture
checks that every torque sample follows the canonical motor law, so a damping (kd) draw stops it
at the first case. --motor-check drawn makes that one comparison use the damping the episode
actually drew (this process only; the repository's check is not edited, and every bound and
threshold stays as it is). Results with it are doubly diagnostic and are labelled in the record.
The runs of 2026-11-10 before these options existed used the canonical check and no "none".

--predictor puts a delay_predictor.py model in front of the policy: the actor's first layer reads
the predictor's output (the inputs N controls ahead) in place of the inputs themselves. N is the
delay the episode drew (--predict-controls drawn, the default: the predictor is told the true
delay) or a fixed number. The observation, the scorer and every bound are untouched; the control
is the same axes and seed without --predictor (the draws are identical). Omitted, nothing changes.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full96_suite  # noqa: E402  (injects the 96 cases and the 90 s episode timeout)
import locomotion.surrogate.env as surrogate_env  # noqa: E402
from locomotion.surrogate import evaluate  # noqa: E402

FIT = {
    "delay": {"action_delay_controls": (1, 2)},
    "kd": {"kd_scale": (0.13, 0.37)},
    "toe_friction": {"friction_scale": (0.45, 0.6)},
    "joint_friction": {"joint_friction_nm": ((0.0335, 0.0622), (0.1387, 0.2576), (0.018, 0.0335))},
    "armature": {"armature_kg_m2": ((0.0, 0.0001), (0.0, 0.00058), (0.0, 0.00049))},
}
OriginalEnv = surrogate_env.SurrogateEnv
ACTIVE = {"randomization": None, "draws": [], "runner": None, "canonical_kd": None}
PREDICT = {"model": None, "controls": None, "env": None, "applied": Counter()}


class FittedEnv(OriginalEnv):
    """The stock surrogate with the fitted ranges drawn per episode and the touchdown model off."""

    def __init__(self, *args, **kwargs):
        contact = kwargs.get("contact") or surrogate_env.ContactModel()
        kwargs["contact"] = replace(contact, touchdown_overshoot=False)
        kwargs["randomization"] = ACTIVE["randomization"]
        if ACTIVE["runner"] is not None:
            ACTIVE["runner"].KD = ACTIVE["canonical_kd"]
        super().__init__(*args, **kwargs)
        PREDICT["env"] = self

    def _randomize(self, rows):
        super()._randomize(rows)
        ACTIVE["draws"].append(
            {key: float(value[0]) for key, value in (self.randomization_draws or {}).items()}
        )
        if ACTIVE["runner"] is not None:
            # --motor-check drawn: the evaluator's motor-law recurrence check (evaluation/runner.py)
            # compares against the damping this episode actually drew, not the canonical one.
            # The evaluator runs one robot, so row 0 is the robot under test.
            ACTIVE["runner"].KD = self._kd_rows[0].detach().cpu().numpy().copy()


surrogate_env.SurrogateEnv = FittedEnv
OriginalPolicy = evaluate.DeterministicPolicy


class PredictedInputs(OriginalPolicy):
    """The stock evaluation policy; with --predictor its actor reads the predicted inputs."""

    def __init__(self, actor, env, *args, **kwargs):
        super().__init__(actor, env, *args, **kwargs)
        if PREDICT["model"] is None:
            return

        def ahead(module, inputs):
            controls = PREDICT["controls"]
            if controls is None:  # the delay this episode drew (the evaluator runs one robot)
                steps = PREDICT["env"]._delay_steps
                controls = 0 if steps is None else int(steps[0])
            PREDICT["applied"][controls] += 1
            if not controls:
                return None
            return (PREDICT["model"](inputs[0], controls=controls),) + tuple(inputs[1:])

        actor.mlp.register_forward_pre_hook(ahead)


evaluate.DeterministicPolicy = PredictedInputs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(axes_text, motor_check):
    """Set this process's drawn ranges and motor check; returns (axis names, ranges)."""
    axes = [name for name in axes_text.split(",") if name and name != "none"]
    ranges = {key: value for name in axes for key, value in FIT[name].items()}
    # "--axes none" is the contact-only control: touchdown model off, no randomization at all.
    ACTIVE["randomization"] = surrogate_env.DynamicsRandomization(**ranges) if ranges else None
    if motor_check == "drawn":
        import locomotion.evaluation.runner as native_runner

        ACTIVE["runner"], ACTIVE["canonical_kd"] = native_runner, native_runner.KD
    return axes, ranges


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoints", nargs="+", metavar="name=checkpoint")
    parser.add_argument("--axes", default=",".join(FIT))
    parser.add_argument("--match", nargs="*", default=None, help="keep case ids containing any")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument(
        "--motor-check",
        choices=["canonical", "drawn"],
        default="canonical",
        help="drawn: the motor-law check uses each episode's drawn damping (needed for kd)",
    )
    parser.add_argument("--predictor", type=Path, default=None, help="delay_predictor.py model")
    parser.add_argument(
        "--predict-controls",
        default="drawn",
        help="controls ahead the predictor is asked for: drawn (the episode's delay) or a number",
    )
    args = parser.parse_args(argv)
    if args.predictor is not None:
        import torch

        import delay_predictor

        PREDICT["model"] = delay_predictor.build(torch.load(args.predictor, weights_only=False))
        PREDICT["controls"] = None if args.predict_controls == "drawn" else int(args.predict_controls)
    axes, ranges = configure(args.axes, args.motor_check)
    ids = [case["case_id"] for case in full96_suite.required()]
    if args.match:
        ids = [i for i in ids if any(text in i for text in args.match)]
    args.root.mkdir(parents=True, exist_ok=True)
    for item in args.checkpoints:
        name, checkpoint = item.split("=", 1)
        directory = args.root / name
        ACTIVE["draws"] = []
        PREDICT["applied"] = Counter()
        evaluate.main(
            ["--checkpoint", checkpoint, "--output", str(directory), "--cases", ",".join(ids),
             "--seed", str(args.seed), "--video", "none", "--no-contact-consensus"]
        )  # fmt: skip
        cases = json.loads((directory / "surrogate_summary.json").read_text())["cases"]
        record = {
            "label": "fitted-dynamics diagnostic, not a judge result",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "seed": args.seed,
            "axes": axes,
            "ranges": ranges,
            "touchdown_overshoot": False,
            "motor_check": args.motor_check,
            "cases": len(cases),
            "passes": sum(bool(case["pass"]) for case in cases),
            "numeric_screen_passes": sum(bool(case.get("passed_numeric_screen")) for case in cases),
            "batch_failures": sum(bool(case.get("batch_failure")) for case in cases),
            "failed_bounds": Counter(b for case in cases for b in case.get("failed_bounds") or []),
            "draws": ACTIVE["draws"],
        }
        if args.predictor is not None:
            record["label"] += "; the policy reads the delay predictor's output, not its inputs"
            record["predictor"] = str(args.predictor)
            record["predictor_sha256"] = sha256(args.predictor)
            record["predict_controls"] = args.predict_controls
            record["controls_ahead_applied"] = dict(PREDICT["applied"])
        (directory / "fitted_draws.json").write_text(json.dumps(record, indent=1) + "\n")
        print(
            f"FITTED {name}: pass {record['passes']}/{len(cases)}, numeric screen "
            f"{record['numeric_screen_passes']}, batch failures {record['batch_failures']}, "
            f"draws {len(record['draws'])}, failed bounds {dict(record['failed_bounds'])}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
