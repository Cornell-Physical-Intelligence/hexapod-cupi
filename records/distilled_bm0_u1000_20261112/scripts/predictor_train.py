"""Surrogate PPO with a frozen delay predictor in front of the actor (the repository trainer, untouched).

The repeating job: one init checkpoint + one frozen delay_predictor.py model -> a policy trained
while reading the predictor's output (its inputs N controls ahead) instead of its inputs. Inputs:
the predictor file, N, and the trainer's own arguments (--initialize-from is required: the hook is
installed where the trainer loads the init). Check: none here; the result is judged by
fitted_eval.py --predictor (unchanged scorer) against the same recipe trained without a predictor
and against the frozen init with the same predictor. Record: the trainer's own output folder plus
<output>/predictor_in_loop.json (predictor sha256, N, this file's sha256, actor calls hooked).

What changes and what does not: only the actor's first layer input, in the rollout and in the PPO
update alike (the stored observations pass through the same hook, so the update's log-probabilities
are those of the policy that acted). The critic reads the observations as they are. The predictor's
weights are frozen and are not saved in the checkpoint: a checkpoint from this script is a policy
that expects the predictor in front of it (evaluate with fitted_eval.py --predictor P
--predict-controls N). N is fixed for every robot: the stored observations do not carry each
episode's drawn delay, so "told the drawn delay" is not available to the update.

Usage (from the hexapod-cupi checkout):
  uv run python predictor_train.py PREDICTOR.pt N [trainer arguments ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delay_predictor  # noqa: E402
import locomotion.rl.ppo as ppo  # noqa: E402
from locomotion.surrogate import train  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    predictor_path, controls, rest = Path(argv[0]), int(argv[1]), argv[2:]
    if controls < 1 or "--initialize-from" not in rest or "--output" not in rest:
        raise SystemExit("needs N >= 1 and the trainer's --initialize-from and --output")
    output = Path(rest[rest.index("--output") + 1])
    model = delay_predictor.build(torch.load(predictor_path, weights_only=False))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    seen = {"installed": 0, "actor_calls": 0, "rows": 0}
    original = ppo.initialize_learner

    def initialize(algorithm, *args, **kwargs):
        result = original(algorithm, *args, **kwargs)

        def ahead(module, inputs):
            x = inputs[0]
            with torch.no_grad():
                predicted = model(x.reshape(-1, x.shape[-1]), controls=controls).reshape(x.shape)
            seen["actor_calls"] += 1
            seen["rows"] += predicted.shape[0] if predicted.dim() > 1 else 1
            return (predicted,) + tuple(inputs[1:])

        algorithm.actor.mlp.register_forward_pre_hook(ahead)
        seen["installed"] += 1
        return result

    ppo.initialize_learner = initialize
    status = train.main(rest)
    unchanged = all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    record = {
        "label": "the actor read a frozen delay predictor's output during training; diagnostic arm",
        "predictor": str(predictor_path),
        "predictor_sha256": sha256(predictor_path),
        "controls_ahead": controls,
        "script_sha256": sha256(__file__),
        "delay_predictor_sha256": sha256(delay_predictor.__file__),
        "hook_installed": seen["installed"],
        "actor_calls_hooked": seen["actor_calls"],
        "rows_predicted": seen["rows"],
        "predictor_weights_unchanged": unchanged,
        "trainer_status": status,
    }
    (output / "predictor_in_loop.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"PREDICTOR IN LOOP: {json.dumps(record)}", flush=True)
    if seen["installed"] != 1 or not unchanged:
        raise SystemExit("the predictor hook was not installed exactly once, or its weights moved")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
