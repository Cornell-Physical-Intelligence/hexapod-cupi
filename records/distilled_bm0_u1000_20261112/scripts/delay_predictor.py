"""A second network beside the policy: the policy's 278 inputs now -> the same 278 inputs one
control (20 ms) later. Apply it once per control of transport delay (the leg-stand fit: 1 or 2).

The repeating job: one checkpoint -> a predictor of its inputs. Inputs: the checkpoint and the
evaluator's 96 required cases minus the 35 of the infeasible 0.20 m/s tier (decisions.md,
2026-11-01 scope ruling: 16 translate_0.20, 16 tight_arc, formal__0.20, formal__0.30,
stop__forward_fast), run on the NORMAL simulator (no delay, no fitted dynamics) with the
repository's evaluator untouched; a hook on the actor's first layer keeps every input it reads.
Check: prediction error on cases held out of training (five folds by case, each case held out
once) against two controls on the same held-out rows: no prediction (the input as it is, which
is what the policy acts on under a delay today) and a linear least-squares predictor. Record:
OUT/inputs.npz + collect.json (the data, the checkpoint and data hashes), OUT/predictor.pt (the
model trained on all kept cases) and OUT/report.json (held-out errors per fold and pooled).

Model: the policy's hidden layers (256, 256, 128, ELU). The four older history frames of the next
input are exact copies of frames in the current one and the command is held, so those 171 values
are copied, not learned; the network outputs the change of the other 107 (newest frame, previous
action, gait clock, odometry ghost, footfall values). Trained on one- and two-control rollouts.
Prediction error is not a scorer result; no judge, gate or bound is read or changed here.

Usage (from the hexapod-cupi checkout):
  uv run python delay_predictor.py collect OUT name=checkpoint.pt [--seed 20261011]
        [--axes kd,toe_friction,joint_friction,armature [--motor-check drawn]]
  uv run python delay_predictor.py merge   OUT COLLECTED [COLLECTED ...]
  uv run python delay_predictor.py train   OUT [--epochs 120] [--folds 5]
  uv run python delay_predictor.py report  OUT
Load: m = build(torch.load("predictor.pt")); m(inputs, controls=d) -> the inputs d controls on.

collect --axes records the same cases in fitted_eval.py's simulator (touchdown model off, the
named leg-stand ranges drawn per episode) in place of the normal one; leave the delay axis out,
because a predictor that works makes the delayed loop behave like the undelayed one. merge joins
collections of one checkpoint and one simulator (different seeds, so different dynamics draws);
folds stay by case, so every seed's run of a held-out case is held out together. Without --axes
and merge nothing changes (the first predictor of 2026-11-10 was made that way).
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fitted_report import kind  # noqa: E402

EXCLUDED = ("translate_0.20", "tight_arc", "formal__0.20", "formal__0.30", "stop__forward_fast")
WIDTH, FRAME, FRAMES = 278, 42, 5
HIDDEN = (256, 256, 128)
# Index ranges inside the 278 inputs (rl/wrapper.py ACTOR_SCALES, then clock and destination inputs).
COMMAND, ACTION, CLOCK, GHOST, FOOTFALL = (210, 213), (213, 231), (231, 233), (233, 242), (242, 278)
# name -> (offset in the newest frame, width, input scale, unit after dividing by the scale)
FRAME_GROUPS = {
    "body angular velocity": (0, 3, 2.0, "rad/s"),
    "gravity direction": (3, 3, 1.0, "-"),
    "joint position": (6, 18, 1 / 0.35, "rad"),
    "joint velocity": (24, 18, 0.5, "rad/s"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excluded(case_id):
    return any(text in case_id for text in EXCLUDED)


# ---------------------------------------------------------------- collect


def collect(root, item, seed, axes=None, motor_check="canonical"):
    sys.path.insert(0, str(C))
    import full96_suite  # injects the 96 cases and the 90 s episode timeout
    from locomotion.surrogate import evaluate

    fitted, simulator = None, "normal (no delay, no fitted dynamics)"
    if axes is not None:
        import fitted_eval  # wraps the simulator: touchdown model off, fitted ranges drawn per episode

        fitted = {"module": fitted_eval, "motor_check": motor_check}
        fitted["axes"], fitted["ranges"] = fitted_eval.configure(axes, motor_check)
        fitted_eval.ACTIVE["draws"] = []
        simulator = (
            f"fitted ({', '.join(fitted['axes']) or 'contact only'} drawn per episode; "
            f"touchdown model off; motor check {motor_check})"
        )
    name, checkpoint = item.split("=", 1)
    log = {"inputs": [], "steps": []}

    class Recording(evaluate.DeterministicPolicy):
        def __init__(self, actor, env, *args, **kwargs):
            super().__init__(actor, env, *args, **kwargs)

            def keep(module, inputs):
                log["inputs"].append(inputs[0][0].detach().to(torch.float32).cpu().numpy().copy())
                log["steps"].append(int(env.episode_steps[0]))

            actor.mlp.register_forward_pre_hook(keep)

    evaluate.DeterministicPolicy = Recording
    every = [case["case_id"] for case in full96_suite.required()]
    ids = [i for i in every if not excluded(i)]
    assert len(every) == 96 and len(ids) == 61, (len(every), len(ids))
    root.mkdir(parents=True, exist_ok=False)
    evaluate.main(
        ["--checkpoint", checkpoint, "--output", str(root / "run"), "--cases", ",".join(ids),
         "--seed", str(seed), "--video", "none", "--no-contact-consensus"]
    )  # fmt: skip
    inputs, steps = np.stack(log["inputs"]), np.asarray(log["steps"])
    episode = np.cumsum(steps == 0) - 1
    results = json.loads((root / "run" / "evaluation" / "summary.json").read_text())["results"]
    order = [row["case_id"] for row in results]
    lengths = np.bincount(episode)
    contiguous = all(
        np.array_equal(steps[episode == e], np.arange(lengths[e])) for e in range(len(lengths))
    )
    record = {
        "label": f"policy inputs on the {simulator.split(' (')[0]} simulator; data for the delay predictor",
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": seed,
        "simulator": simulator,
        "excluded_cases": [i for i in every if excluded(i)],
        "cases": ids,
        "scorer_case_order": order,
        "episodes": int(len(lengths)),
        "rows": int(len(inputs)),
        "controls_per_episode": lengths.tolist(),
        "steps_contiguous": bool(contiguous),
        "scorer_pass": {row["case_id"]: bool(row["pass"]) for row in results},
    }
    if fitted is not None:
        record.update(
            axes=fitted["axes"], ranges=fitted["ranges"], motor_check=motor_check,
            touchdown_overshoot=False, draws=fitted["module"].ACTIVE["draws"],
            ended_early=sum(bool(row.get("batch_failure")) for row in results),
        )  # fmt: skip
    assert inputs.shape[1] == WIDTH, inputs.shape
    assert len(lengths) == len(ids) and contiguous, (len(lengths), len(ids), contiguous)
    np.savez_compressed(root / "inputs.npz", inputs=inputs, episode=episode, step=steps)
    record["inputs_sha256"] = sha256(root / "inputs.npz")
    (root / "collect.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"COLLECTED {name}: {len(inputs)} inputs in {len(lengths)} episodes "
        f"({lengths.min()} to {lengths.max()} controls), scorer pass "
        f"{sum(record['scorer_pass'].values())}/{len(results)}",
        flush=True,
    )


def merge(root, sources):
    """Join collections of one checkpoint and one simulator; episodes keep their case ids."""
    parts, inputs, episode, step, episode_case = [], [], [], [], []
    for source in sources:
        record = json.loads((source / "collect.json").read_text())
        assert record["inputs_sha256"] == sha256(source / "inputs.npz"), source
        data = np.load(source / "inputs.npz")
        inputs.append(data["inputs"])
        episode.append(data["episode"] + len(episode_case))
        step.append(data["step"])
        episode_case += record.get("episode_case", record["cases"])
        parts.append(
            {"folder": str(source), "scorer_passes": sum(record["scorer_pass"].values()),
             **{key: record.get(key) for key in ("seed", "episodes", "rows", "ended_early", "inputs_sha256")}}
        )  # fmt: skip
        if len(parts) == 1:
            first = record
        same = ("checkpoint_sha256", "simulator", "cases", "excluded_cases")
        assert all(record[key] == first[key] for key in same), source
    assert len({part["seed"] for part in parts}) == len(parts), "one collection per seed"
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        root / "inputs.npz", inputs=np.concatenate(inputs), episode=np.concatenate(episode), step=np.concatenate(step)
    )
    merged = {
        **{key: first[key] for key in ("name", "checkpoint", "checkpoint_sha256", "simulator", "excluded_cases", "cases")},
        "label": first["label"] + f" ({len(parts)} collections merged)",
        "seed": [part["seed"] for part in parts],
        "episode_case": episode_case,
        "episodes": len(episode_case),
        "rows": int(sum(len(block) for block in inputs)),
        "parts": parts,
        "inputs_sha256": sha256(root / "inputs.npz"),
    }
    for key in ("axes", "ranges", "motor_check", "touchdown_overshoot"):
        if key in first:
            merged[key] = first[key]
    (root / "collect.json").write_text(json.dumps(merged, indent=1) + "\n")
    print(f"MERGED {len(parts)} collections: {merged['rows']} inputs in {merged['episodes']} episodes", flush=True)


# ---------------------------------------------------------------- model


def structure(x0, x1):
    """Which next inputs are copies of current ones. Returns (source index, learned index, record)."""
    old, new = FRAME * (FRAMES - 1), FRAME * FRAMES
    first = float(np.abs(x1[:, FRAME:new] - x0[:, :old]).max())  # newest frame in slot 0
    last = float(np.abs(x1[:, :old] - x0[:, FRAME:new]).max())  # newest frame in the last slot
    source = np.arange(WIDTH)
    if first <= 1e-6:
        source[FRAME:new] = np.arange(old)
        newest = 0
    elif last <= 1e-6:
        source[:old] = np.arange(FRAME, new)
        newest = old
    else:
        raise SystemExit(f"history frames do not shift exactly: {first} / {last}")
    learned = np.concatenate(
        [np.arange(newest, newest + FRAME)]
        + [np.arange(*span) for span in (ACTION, CLOCK, GHOST, FOOTFALL)]
    )
    return source, learned, {"newest_frame_offset": newest, "shift_mismatch": min(first, last)}


class DelayPredictor(nn.Module):
    """inputs now -> inputs ``controls`` controls later (each control one application of step)."""

    def __init__(self, source, learned, linear=False):
        super().__init__()
        self.register_buffer("source", torch.as_tensor(source, dtype=torch.long))
        self.register_buffer("learned", torch.as_tensor(learned, dtype=torch.long))
        for name, size in (("mean", WIDTH), ("scale", WIDTH), ("delta_mean", len(learned)), ("delta_scale", len(learned))):
            self.register_buffer(name, torch.zeros(size))
        if linear:
            self.net = nn.Linear(WIDTH, len(learned))
        else:
            layers, width = [], WIDTH
            for hidden in HIDDEN:
                layers += [nn.Linear(width, hidden), nn.ELU()]
                width = hidden
            self.net = nn.Sequential(*layers, nn.Linear(width, len(learned)))

    def standardize(self, x0, x1):
        self.mean.copy_(x0.mean(0))
        deviation = x0.std(0)
        self.scale.copy_(torch.where(deviation > 1e-6, deviation, torch.ones_like(deviation)))
        delta = (x1 - x0)[:, self.learned]
        self.delta_mean.copy_(delta.mean(0))
        deviation = delta.std(0)
        # An input that never changes in training keeps a zero scale: its predicted change is 0.
        self.delta_scale.copy_(torch.where(deviation > 1e-8, deviation, torch.zeros_like(deviation)))

    def step(self, x):
        change = self.net((x - self.mean) / self.scale) * self.delta_scale + self.delta_mean
        return x[:, self.source].index_add(1, self.learned, change)

    def forward(self, x, controls=1):
        for _ in range(controls):
            x = self.step(x)
        return x


def build(saved):
    model = DelayPredictor(saved["source"], saved["learned"])
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def fit(model, X, rows, held, epochs, seed, batch=512):
    """Huber loss on the standardized one- and two-control errors of the learned inputs."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model.standardize(X[rows], X[rows + 1])
    inverse = torch.where(model.delta_scale > 0, 1 / model.delta_scale.clamp_min(1e-12), torch.zeros(()))

    def huber(error):
        return nn.functional.smooth_l1_loss(error, torch.zeros_like(error))

    def loss_of(index):
        first = model.step(X[index])
        second = model.step(first)
        return huber((first - X[index + 1])[:, model.learned] * inverse) + huber(
            (second - X[index + 2])[:, model.learned] * inverse
        )

    steps = epochs * ((len(rows) + batch - 1) // batch)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rows[torch.randperm(len(rows), generator=generator)]
        total = 0.0
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            loss = loss_of(index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss.detach()) * len(index)
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                outside = float(loss_of(held)) if held is not None and len(held) else None
            curve.append({"epoch": epoch, "train": total / len(order), "held_out": outside})
            print(f"   epoch {epoch:>4}: train {total / len(order):.4f} held-out {outside}", flush=True)
    return curve


def linear_control(source, learned, X, rows):
    """Least-squares predictor of the same learned changes from the same inputs (ridge 1e-3)."""
    model = DelayPredictor(source, learned, linear=True)
    with torch.no_grad():
        model.standardize(X[rows], X[rows + 1])
        a = ((X[rows] - model.mean) / model.scale).double()
        inverse = torch.where(model.delta_scale > 0, 1 / model.delta_scale.clamp_min(1e-12), torch.zeros(()))
        b = (((X[rows + 1] - X[rows])[:, model.learned] - model.delta_mean) * inverse).double()
        gram = a.T @ a + 1e-3 * len(rows) * torch.eye(WIDTH, dtype=torch.double)
        weight = torch.linalg.solve(gram, a.T @ b)
        model.net.weight.copy_(weight.T.float())
        model.net.bias.zero_()
    return model.eval()


def groups(newest):
    table = {
        name: (np.arange(newest + offset, newest + offset + width), scale, unit)
        for name, (offset, width, scale, unit) in FRAME_GROUPS.items()
    }
    table["previous action"] = (np.arange(*ACTION), 1.0, "input")
    table["gait clock"] = (np.arange(*CLOCK), 1.0, "-")
    table["odometry ghost"] = (np.arange(*GHOST), 1.0, "input")
    table["footfall values"] = (np.arange(*FOOTFALL), 1.0, "input")
    table["all 278 inputs"] = (np.arange(WIDTH), 1.0, "input")
    return table


def squared_errors(models, X, rows, newest):
    """Sum of squared error and count per (controls ahead, group, predictor), in physical units."""
    out = {}
    with torch.no_grad():
        for controls in (1, 2):
            truth = X[rows + controls]
            predicted = {"no prediction": X[rows], **{k: m(X[rows], controls) for k, m in models.items()}}
            for group, (index, scale, _) in groups(newest).items():
                for name, value in predicted.items():
                    error = ((value - truth)[:, index] / scale).double()
                    out[f"{controls}|{group}|{name}"] = [float((error**2).sum()), int(error.numel())]
    return out


def rmse(cell):
    return (cell[0] / cell[1]) ** 0.5 if cell[1] else None


# ---------------------------------------------------------------- train


def train(root, epochs, folds):
    torch.set_num_threads(6)
    data = np.load(root / "inputs.npz")
    collected = json.loads((root / "collect.json").read_text())
    X = torch.as_tensor(data["inputs"])
    episode = data["episode"]
    cases = collected.get("episode_case", collected["cases"])  # the case each episode ran
    # Rows whose next two controls lie in the same episode.
    rows_all = np.flatnonzero((episode[:-2] == episode[2:]))
    source, learned, shape = structure(data["inputs"][rows_all], data["inputs"][rows_all + 1])
    # Folds by case, spread within each command type so every fold holds each type where it can.
    # A merged collection runs each case once per seed; all of a case's episodes share its fold.
    ranked = sorted(set(cases), key=lambda case: (kind(case), case))
    fold_of_case = {case: position % folds for position, case in enumerate(ranked)}
    fold_of = np.array([fold_of_case[case] for case in cases])
    report = {
        "label": "delay predictor; held-out prediction error, not a scorer result",
        "simulator": collected["simulator"],
        "checkpoint_sha256": collected["checkpoint_sha256"],
        "inputs_sha256": collected["inputs_sha256"],
        "rows": int(len(X)), "training_rows": int(len(rows_all)), "cases": len(ranked),
        "episodes": len(cases), "collection_seed": collected["seed"],
        "structure": {**shape, "copied_inputs": int(WIDTH - len(learned)), "learned_inputs": int(len(learned))},
        "recipe": {"hidden": HIDDEN, "activation": "elu", "epochs": epochs, "batch": 512,
                   "optimizer": "AdamW 1e-3 cosine to 1e-5, weight decay 1e-5",
                   "loss": "Huber on standardized one- and two-control errors of the learned inputs"},
        "folds": [],
    }  # fmt: skip
    pooled = {}
    for fold in range(folds):
        held_cases = np.flatnonzero(fold_of == fold)
        inside = torch.as_tensor(rows_all[~np.isin(episode[rows_all], held_cases)])
        outside = torch.as_tensor(rows_all[np.isin(episode[rows_all], held_cases)])
        print(f"fold {fold}: train {len(inside)} rows, held out {len(outside)} rows in {len(held_cases)} cases", flush=True)
        model = DelayPredictor(source, learned)
        curve = fit(model, X, inside, outside, epochs, seed=20261110 + fold)
        models = {"network": model.eval(), "linear": linear_control(source, learned, X, inside)}
        errors = squared_errors(models, X, outside, shape["newest_frame_offset"])
        for key, cell in errors.items():
            total = pooled.setdefault(key, [0.0, 0])
            total[0] += cell[0]
            total[1] += cell[1]
        report["folds"].append(
            {"fold": fold, "held_out_cases": list(dict.fromkeys(cases[e] for e in held_cases)), "train_rows": int(len(inside)),
             "held_out_rows": int(len(outside)), "curve": curve, "rmse": {k: rmse(v) for k, v in errors.items()}}
        )  # fmt: skip
    report["pooled_held_out_rmse"] = {k: rmse(v) for k, v in pooled.items()}
    print("final model: all kept cases", flush=True)
    everything = torch.as_tensor(rows_all)
    final = DelayPredictor(source, learned)
    report["final_curve"] = fit(final, X, everything, None, epochs, seed=20261110 + folds)
    final.eval()
    inside_errors = squared_errors({"network": final}, X, everything, shape["newest_frame_offset"])
    report["final_in_sample_rmse"] = {k: rmse(v) for k, v in inside_errors.items()}
    report["parameters"] = int(sum(p.numel() for p in final.net.parameters()))
    torch.save(
        {"state_dict": final.state_dict(), "source": source, "learned": learned, "hidden": HIDDEN,
         "checkpoint_sha256": collected["checkpoint_sha256"], "inputs_sha256": collected["inputs_sha256"],
         "recipe": report["recipe"], "seed": 20261110 + folds},
        root / "predictor.pt",
    )  # fmt: skip
    report["predictor_sha256"] = sha256(root / "predictor.pt")
    (root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    show(root)


# ---------------------------------------------------------------- report


def show(root):
    report = json.loads((root / "report.json").read_text())
    table = groups(report["structure"]["newest_frame_offset"])
    pooled, folds = report["pooled_held_out_rmse"], report["folds"]
    variance = np.load(root / "inputs.npz")["inputs"].astype(np.float64).var(0)

    def ratios(controls, group, control):
        values = [
            f["rmse"][f"{controls}|{group}|network"] / f["rmse"][f"{controls}|{group}|{control}"]
            for f in folds
            if f["rmse"][f"{controls}|{group}|{control}"]
        ]
        if not values:
            return f'{"-":>8}{"-":>6}{"-":>6}'
        return f"{statistics.median(values):>8.2f}{min(values):>6.2f}{max(values):>6.2f}"

    print(
        f'\nDelay predictor, {report["simulator"]} simulator; {report["parameters"]} parameters; '
        f'{report["training_rows"]} rows from {report["cases"]} cases; {len(folds)} folds by case, each case held out once'
    )
    print(
        "Held-out root-mean-square error, pooled over the folds. size = standard deviation of the input over all\n"
        "recorded rows. Ratios = network / control per fold (median, min, max); below 1 means the network is closer."
    )
    head = (
        f'{"input group":<23}{"unit":>6}{"size":>9}{"no prediction":>15}{"linear":>10}{"network":>10}'
        f'   {"/ no prediction":>20}   {"/ linear":>20}'
    )
    for controls in (1, 2):
        print(f"\n{controls} control{'s' if controls > 1 else ''} ahead ({20 * controls} ms)\n{head}")
        for group, (index, scale, unit) in table.items():
            cells = [pooled[f"{controls}|{group}|{name}"] for name in ("no prediction", "linear", "network")]
            size = float(np.sqrt(variance[index].mean())) / scale
            print(
                f"{group:<23}{unit:>6}{size:>9.4f}{cells[0]:>15.5f}{cells[1]:>10.5f}{cells[2]:>10.5f}"
                f'   {ratios(controls, group, "no prediction")}   {ratios(controls, group, "linear")}'
            )
    last = [f["curve"][-1]["held_out"] for f in folds]
    best = [min(c["held_out"] for c in f["curve"]) for f in folds]
    train_last = [f["curve"][-1]["train"] for f in folds]
    print(f"\nloss at the last epoch per fold, training / held-out (lowest held-out during training): "
          + ", ".join(f"{t:.3f} / {a:.3f} ({b:.3f})" for t, a, b in zip(train_last, last, best)))
    inside = report["final_in_sample_rmse"]
    print(
        "final model (trained on all cases; in-sample, not held out), all 278 inputs: "
        f'{inside["1|all 278 inputs|network"]:.5f} at 1 control, {inside["2|all 278 inputs|network"]:.5f} at 2'
    )
    held = {f["fold"]: sorted({kind(c) for c in f["held_out_cases"]}) for f in folds}
    print("command types held out per fold: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in held.items()))
    print(f'predictor.pt sha256 {report["predictor_sha256"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "merge", "train", "report"])
    parser.add_argument("root", type=Path)
    parser.add_argument("items", nargs="*", metavar="name=checkpoint | COLLECTED")
    parser.add_argument("--seed", type=int, default=20261011, help="the judge runs' evaluation seed")
    parser.add_argument("--axes", default=None, help="collect in fitted_eval.py's simulator with these axes")
    parser.add_argument("--motor-check", choices=["canonical", "drawn"], default="canonical")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.root, args.items[0], args.seed, args.axes, args.motor_check)
    elif args.mode == "merge":
        merge(args.root, [Path(item) for item in args.items])
    elif args.mode == "train":
        train(args.root, args.epochs, args.folds)
    else:
        show(args.root)


if __name__ == "__main__":
    main()
