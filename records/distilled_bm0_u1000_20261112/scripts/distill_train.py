"""DAgger distillation on the surrogate: a student with the actor's own inputs and outputs (278 -> 18)
learns to act as "frozen predictor in front of a frozen teacher actor" does, with no predictor.

The repeating job: one teacher checkpoint + one frozen delay_predictor.py model + N -> a student
checkpoint in the trainer's own format that runs alone (evaluate with fitted_eval.py, no
--predictor). Inputs: the predictor file, N, and the repository trainer's own arguments
(--initialize-from is both the teacher and the student's start; the trainer, env, wrapper,
randomization and checkpoint code are untouched). Check: fitted_eval.py on the student alone
against the teacher with the predictor in front (same axes, seeds, draws), plus the normal
simulator judge for the unchanged gates. Record: the trainer's output folder plus
<output>/distillation.json.

What replaces PPO (only algorithm.act and algorithm.update are wrapped):
- Rollouts: the student acts (its sampled action, the trainer's own deviation). For a fraction
  BETA of the robots, re-chosen each update, the teacher acts instead (its mean plus the same
  deviation), so early rollouts reach the states the teacher reaches; BETA falls linearly from
  BETA0 to 0 over BETA_UPDATES updates, after which the student drives alone (DAgger).
- Labels: on every stored state, the teacher's deterministic action (tanh mean) with the
  predictor N controls ahead in front of the teacher's first layer.
- Update: EPOCHS passes of mean-squared error between the student's deterministic action and the
  label, Adam at LR over the student actor's MLP only. The critic is fitted to the rollout's
  returns with its own Adam (kept usable for a later PPO stage); PPO's optimizer is not stepped.
No anchor, no PPO loss. The predictor and teacher weights are frozen and checked unchanged.

Usage (from the hexapod-cupi checkout):
  uv run python distill_train.py PREDICTOR.pt N [--beta0 B] [--beta-updates U] [--epochs E]
        [--lr LR] [trainer arguments ...]
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delay_predictor  # noqa: E402
import locomotion.rl.ppo as ppo  # noqa: E402
from locomotion.surrogate import train  # noqa: E402

OWN = {"--beta0": 0.5, "--beta-updates": 200, "--epochs": 5, "--lr": 5e-4}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take_own(rest):
    values = dict(OWN)
    for flag in OWN:
        if flag in rest:
            at = rest.index(flag)
            values[flag] = type(OWN[flag])(rest[at + 1])
            rest = rest[:at] + rest[at + 2 :]
    return values, rest


def main(argv):
    predictor_path, controls = Path(argv[0]), int(argv[1])
    own, rest = take_own(argv[2:])
    if controls < 1 or "--initialize-from" not in rest or "--output" not in rest:
        raise SystemExit("needs N >= 1 and the trainer's --initialize-from and --output")
    if "--anchor-weight" in rest:
        raise SystemExit("no anchor: it would pull the student back toward the teacher's weights")
    output = Path(rest[rest.index("--output") + 1])
    teacher_path = Path(rest[rest.index("--initialize-from") + 1])
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    predictor = delay_predictor.build(torch.load(predictor_path, weights_only=False))
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    predictor_before = {k: v.clone() for k, v in predictor.state_dict().items()}
    log = {"updates": [], "installed": 0}
    held = {}
    original = ppo.initialize_learner

    def initialize(algorithm, *args, **kwargs):
        result = original(algorithm, *args, **kwargs)
        teacher = copy.deepcopy(algorithm.actor).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        held["teacher_before"] = {k: v.clone() for k, v in teacher.state_dict().items()}

        def ahead(module, inputs):
            x = inputs[0]
            return (predictor(x.reshape(-1, x.shape[-1]), controls=controls).reshape(x.shape),)

        teacher.mlp.register_forward_pre_hook(ahead)
        student = algorithm.actor
        actor_optimizer = torch.optim.Adam(student.mlp.parameters(), lr=own["--lr"])
        critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(seed + 7919)
        state = {"update": 0, "mask": None}
        stock_act = algorithm.act

        def beta():
            if own["--beta-updates"] <= 0:
                return 0.0
            return max(0.0, own["--beta0"] * (1 - state["update"] / own["--beta-updates"]))

        def act(obs):
            actions = stock_act(obs)
            mask = state["mask"]
            if mask is not None and mask.any():
                with torch.no_grad():
                    label = teacher(obs)
                    noise = torch.randn(label.shape, generator=generator) * student.output_std
                    actions = actions.clone()
                    actions[mask] = (label + noise)[mask]
                algorithm.transition.actions = actions.detach()
            return actions

        def update():
            storage = algorithm.storage
            observations = storage.observations.flatten(0, 1)
            returns = storage.returns.flatten(0, 1)
            rows = returns.shape[0]
            with torch.no_grad():
                labels = teacher(observations)
                before = (student(observations) - labels).pow(2).mean().item()
            batches = 6
            actor_loss = critic_loss = 0.0
            for _ in range(own["--epochs"]):
                order = torch.randperm(rows, generator=generator)
                for chunk in order.chunk(batches):
                    batch = observations[chunk]
                    loss = (student(batch) - labels[chunk]).pow(2).mean()
                    actor_optimizer.zero_grad()
                    loss.backward()
                    actor_optimizer.step()
                    actor_loss += loss.item()
                    value = algorithm.critic(batch).squeeze(-1)
                    vloss = (value - returns[chunk].squeeze(-1)).pow(2).mean()
                    critic_optimizer.zero_grad()
                    vloss.backward()
                    critic_optimizer.step()
                    critic_loss += vloss.item()
            steps = own["--epochs"] * batches
            with torch.no_grad():
                after = (student(observations) - labels).pow(2).mean().item()
            mask = state["mask"]
            log["updates"].append(
                {
                    "update": state["update"],
                    "beta": round(beta(), 4),
                    "teacher_driven_robots": 0 if mask is None else int(mask.sum()),
                    "mse_before": before,
                    "mse_after": after,
                    "label_abs_mean": labels.abs().mean().item(),
                }
            )
            print(
                f"DISTILL update {state['update']} beta {beta():.3f} mse {before:.5f} -> {after:.5f}",
                flush=True,
            )
            storage.clear()
            state["update"] += 1
            n = algorithm.storage.observations.shape[1]
            b = beta()
            state["mask"] = (torch.rand(n, generator=generator) < b) if b > 0 else None
            return {"value": critic_loss / steps, "surrogate": actor_loss / steps, "entropy": 0.0}

        n = algorithm.storage.observations.shape[1] if algorithm.storage is not None else None
        held["setup"] = (state, beta, n)
        algorithm.act = act
        algorithm.update = update
        held["teacher"] = teacher
        log["installed"] += 1
        return result

    ppo.initialize_learner = initialize
    # The first rollout's mask: the storage exists only once the runner is built, so it is set on
    # the first act call instead.
    status_holder = {}

    stock_initialize = ppo.initialize_learner

    def initialize_with_first_mask(algorithm, *args, **kwargs):
        result = stock_initialize(algorithm, *args, **kwargs)
        state, beta, _ = held["setup"]
        installed_act = algorithm.act

        def first_act(obs):
            if state["mask"] is None and state["update"] == 0 and beta() > 0:
                generator0 = torch.Generator().manual_seed(seed + 104729)
                state["mask"] = torch.rand(obs.batch_size[0], generator=generator0) < beta()
            algorithm.act = installed_act
            return installed_act(obs)

        algorithm.act = first_act
        return result

    ppo.initialize_learner = initialize_with_first_mask
    status_holder["status"] = train.main(rest)
    teacher_unchanged = all(
        torch.equal(held["teacher_before"][k], v) for k, v in held["teacher"].state_dict().items()
    )
    predictor_unchanged = all(
        torch.equal(predictor_before[k], v) for k, v in predictor.state_dict().items()
    )
    record = {
        "label": "DAgger distillation of teacher + delay predictor into a student without one; diagnostic arm",
        "teacher": str(teacher_path),
        "teacher_sha256": sha256(teacher_path),
        "predictor": str(predictor_path),
        "predictor_sha256": sha256(predictor_path),
        "controls_ahead": controls,
        "options": {k.lstrip("-"): v for k, v in own.items()},
        "script_sha256": sha256(__file__),
        "delay_predictor_sha256": sha256(delay_predictor.__file__),
        "installed": log["installed"],
        "teacher_weights_unchanged": teacher_unchanged,
        "predictor_weights_unchanged": predictor_unchanged,
        "trainer_status": status_holder["status"],
        "updates": log["updates"],
    }
    (output / "distillation.json").write_text(json.dumps(record, indent=1) + "\n")
    last = log["updates"][-1] if log["updates"] else {}
    print(f"DISTILLATION DONE: installed {log['installed']}, last {json.dumps(last)}", flush=True)
    if log["installed"] != 1 or not teacher_unchanged or not predictor_unchanged:
        raise SystemExit("the hooks were not installed exactly once, or frozen weights moved")
    return status_holder["status"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
