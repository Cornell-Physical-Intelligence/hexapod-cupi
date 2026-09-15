"""Read-only evidence analysis; writes only a fresh report beside this script."""
import collections
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parents[2]
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path):
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def load_run(number):
    base = A / f"results_train_{number:03d}"
    metrics = [json.loads(line) for line in bind(base / "standing/learner/metrics.jsonl").read_text().splitlines()]
    tasks = []
    for line in bind(base / "logs/standing.log").read_text().splitlines():
        if '"interval_metrics"' not in line:
            continue
        try:
            item = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
        if item.get("schema") == "canonical_paper_walking_task_v1":
            tasks.append(item)
    assert len(metrics) == len(tasks) == (200 if number == 6 else 20)
    for index, task in enumerate(tasks):
        assert task["controls_completed"] == (index + 1) * 24
        assert task["interval_metrics"]["environment_controls"] == 3072
    for name in ("standing/state.json", "cleanup.json", "standing/task/latest_status.json"):
        bind(base / name)
    return metrics, tasks


def stats(values):
    assert values and all(math.isfinite(x) for x in values)
    return dict(mean=sum(values)/len(values), minimum=min(values), maximum=max(values), first=values[0], last=values[-1])


def aggregate(metrics, tasks):
    intervals = [task["interval_metrics"] for task in tasks]
    count = sum(x["environment_controls"] for x in intervals)
    moving = sum(x["moving_command_rows"] for x in intervals)
    zero = sum(x["zero_command_rows"] for x in intervals)
    names = ("kl_final", "kl_max", "normalizer_policy_kl", "value_loss", "explained_variance", "estimator_loss", "reward", "task_reward", "style_reward", "action_std", "raw_action_clip_fraction", "raw_action_abs_mean", "clip_fraction", "policy_score", "expert_score")
    result = {"updates": [metrics[0]["update"], metrics[-1]["update"]], "update_count": len(metrics), "environment_controls": count, "learner": {k: stats([m[k] for m in metrics]) for k in names}}
    result.update(moving_command_rows=moving, moving_projection_mps=sum(x["moving_signed_command_direction_speed_mps"]*x["moving_command_rows"] for x in intervals if x["moving_command_rows"])/moving,
                  requested_saturation_fraction=sum(x["requested_saturation_fraction"]*x["environment_controls"] for x in intervals)/count,
                  terminations=sum(x["terminations"] for x in intervals), truncations=sum(x["truncations"] for x in intervals),
                  termination_reasons={k: sum(x["termination_reasons"]["rows"][k] for x in intervals) for k in ("height", "tilt", "joint_limit", "timeout")},
                  nonfoot_event_rows=sum(x["nonfoot_event_rows"] for x in intervals),
                  model_steps=sum(x["ppo_batches"] for x in metrics), discriminator_batches=sum(x["discriminator_batches"] for x in metrics))
    result["reward_components"] = {k: {"sum": sum(x["reward_components"][k]["sum"] for x in intervals), "mean": sum(x["reward_components"][k]["sum"] for x in intervals)/count, "absolute_sum": sum(x["reward_components"][k]["absolute_sum"] for x in intervals)} for k in intervals[0]["reward_components"]}
    result["command_classes"] = {}
    for name in intervals[0]["command_classes"]:
        rows = [x["command_classes"][name] for x in intervals]
        n = sum(x["environment_controls"] for x in rows)
        nm = sum(x["moving_command_rows"] for x in rows)
        result["command_classes"][name] = {"environment_controls": n, "moving_rows": nm, "moving_projection_mps": sum(x["moving_signed_command_direction_speed_mps"]*x["moving_command_rows"] for x in rows if x["moving_command_rows"])/nm if nm else None, "task_reward_mean": sum(x["task_reward_mean"]*x["environment_controls"] for x in rows if x["environment_controls"])/n if n else None}
    result["zero_rows"] = zero
    result["zero_joint_rate_rms_rad_s"] = [math.sqrt(sum(x["zero_hold_joint_rate_rms_rad_s"][j]**2*x["zero_command_rows"] for x in intervals if x["zero_command_rows"])/zero) for j in range(18)]
    result["zero_worst_joint_rate_rms_rad_s"] = max(result["zero_joint_rate_rms_rad_s"])
    slew = [x["actual_target_slew"] for x in intervals]
    eligible = sum(x["environment_controls"] for x in slew)
    result["target_slew"] = {"eligible_rows": eligible, "excluded_first_or_reset_rows": sum(x["excluded_first_or_reset_rows"] for x in slew), "by_joint": {j: {"occupancy_fraction": sum(x["by_joint"][j]["at_limit_rows"] for x in slew)/eligible, "beyond_limit_rows": sum(x["by_joint"][j]["beyond_limit_rows"] for x in slew)} for j in slew[0]["by_joint"]}}
    result["target_slew"]["mean_joint_occupancy"] = sum(x["occupancy_fraction"] for x in result["target_slew"]["by_joint"].values())/18
    return result


def guard(metrics):
    attempts = [a for m in metrics for a in m["model_trial_attempts"]]
    for m in metrics:
        accepted = [a for a in m["model_trial_attempts"] if a["accepted"]]
        assert len(accepted) == m["accepted_model_steps"] == m["ppo_batches"] == len(m["kl_after_model_steps"])
        assert len(m["model_trial_attempts"])-len(accepted) == m["rejected_model_trials"]
        assert m["discriminator_batches"] == 20 and m["normalizer_policy_kl"] == 0
        assert all(a["kl"] <= .02 for a in accepted)
        assert all(a["kl"] > .02 for a in m["model_trial_attempts"] if not a["accepted"])
        for a in m["model_trial_attempts"]:
            assert a["retry"] in range(4) and a["learning_rate"] == .0001*.5**a["retry"]
    return {"proposals": sum(m["model_proposals"] for m in metrics), "trials": len(attempts), "accepted": sum(a["accepted"] for a in attempts), "rejected": sum(not a["accepted"] for a in attempts), "trial_rate_counts": dict(collections.Counter(str(a["learning_rate"]) for a in attempts)), "accepted_rate_counts": dict(collections.Counter(str(a["learning_rate"]) for a in attempts if a["accepted"])), "max_accepted_kl": max(a["kl"] for a in attempts if a["accepted"]), "max_rejected_kl": max(a["kl"] for a in attempts if not a["accepted"]), "zero_accepted_updates": sum(m["accepted_model_steps"] == 0 for m in metrics), "per_update": [{k: m[k] for k in ("update", "model_proposals", "accepted_model_steps", "rejected_model_trials", "kl_final", "kl_max", "learning_rate", "discriminator_batches")} for m in metrics]}


def fixed_observation_probe():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    source = bind(A / "source_018/learner.py")
    bind(A / "source_018/FREEZE_SHA256.json")
    spec = importlib.util.spec_from_file_location("frozen_analysis_learner_018", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = torch.load(bind(ROOT / "artifacts/restart_2026-09-14/paper_ppo_migration_001/run_001/checkpoint_update000200_migrated.pt"), map_location="cpu", weights_only=False)
    new = torch.load(bind(A / "results_train_007/standing/learner/checkpoint_000220.pt"), map_location="cpu", weights_only=False)
    assert old["schema"] == new["schema"] == module.SCHEMA
    assert old["learner_sha256"] == new["learner_sha256"] == sha(source)
    assert old["config"] == new["config"] and old["prior_sha256"] == new["prior_sha256"]
    pairs = []
    for payload in (old, new):
        config = module.Config(**payload["config"])
        model, amp = module.ActorCritic(config), module.MotionPrior(config)
        model.load_state_dict(payload["model"], strict=True)
        amp.load_state_dict(payload["amp"], strict=True)
        pairs.append((model.eval(), amp.eval()))
    normalizer_keys = [k for k in old["model"] if k.startswith("obs_normalizer.")]
    assert normalizer_keys and all(torch.equal(old["model"][k], new["model"][k]) for k in normalizer_keys)
    result = {"scope": "Counterfactual CPU inference on all real checkpoint200 evaluation013 recorded rows; not a checkpoint220 closed-loop rollout. Accumulated old200||new220 KL on off-rollout states is not the per-update safeguard statistic.", "actor_observation_normalizer_bitwise_unchanged": True, "counters_old": old["counters"], "counters_new": new["counters"], "final_optimizer_learning_rates": [g["lr"] for g in new["optimizer"]["param_groups"]], "cases": []}
    for index, case in enumerate(("forward", "quiet", "stop")):
        path = bind(A / f"results_evaluate_013/standing/evaluation/batch_{index:03d}/control_trace.npz")
        with np.load(path, allow_pickle=False) as arrays:
            obs = torch.from_numpy(arrays["policy_observation"].reshape(-1,231).copy())
            critic = torch.from_numpy(arrays["critic_observation"].reshape(-1,234).copy())
            first = torch.from_numpy(arrays["amp_state_before"].reshape(-1,61).copy())
            second = torch.from_numpy(arrays["amp_state_after"].reshape(-1,61).copy())
        with torch.no_grad():
            outputs = []
            for model, amp in pairs:
                dist, velocity = model.distribution(obs)
                outputs.append((dist, velocity, model.value(critic), amp.style_reward(first,second)))
            old_dist, new_dist = outputs[0][0], outputs[1][0]
            om, nm = old_dist.mean.double(), new_dist.mean.double()
            os, ns = old_dist.stddev.double(), new_dist.stddev.double()
            kl = (torch.log(ns/os)+(os.square()+(om-nm).square())/(2*ns.square())-.5).sum(-1)
            result["cases"].append({"case": case, "rows": len(obs), "old200_new220_kl_mean": kl.mean().item(), "old200_new220_kl_max": kl.max().item(), "mean_action_absolute_change": (old_dist.mean-new_dist.mean).abs().mean().item(), "old200": {"raw_mean_clip_fraction": (old_dist.mean.abs()>1).float().mean().item(), "velocity_rmse_mps": (outputs[0][1]-critic[:,-3:]).square().mean().sqrt().item(), "critic_mean": outputs[0][2].mean().item(), "style_mean": outputs[0][3].mean().item()}, "new220": {"raw_mean_clip_fraction": (new_dist.mean.abs()>1).float().mean().item(), "velocity_rmse_mps": (outputs[1][1]-critic[:,-3:]).square().mean().sqrt().item(), "critic_mean": outputs[1][2].mean().item(), "style_mean": outputs[1][3].mean().item()}})
    return result


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    bind(Path(__file__).resolve())
    m6, t6 = load_run(6)
    m7, t7 = load_run(7)
    assert [x["update"] for x in m7] == list(range(201,221))
    report = {"schema": "train007_matched_window_diagnostic_v1", "started_utc": started, "train006_last20": aggregate(m6[-20:], t6[-20:]), "train007_all20": aggregate(m7,t7), "train007_first10": aggregate(m7[:10],t7[:10]), "train007_last10": aggregate(m7[10:],t7[10:]), "rollback_guard": guard(m7), "fixed_observation_probe": fixed_observation_probe(), "allocation_new_transitions": m7[-1]["allocation_new_transitions"], "allocation_elapsed_s": m7[-1]["elapsed_s"], "allocation_new_transitions_per_s": m7[-1]["allocation_new_transitions_per_s"], "limitations": ["Train007 starts from new physical reset and newly drawn commands; train006 last20 is mid-run. These windows are not controlled causal replicas.", "Task metrics use compact training endpoint aggregates, not a full raw Stage2 physical acceptance audit.", "Reward and critic/style losses have changing return/discriminator targets; rising reward does not prove command tracking.", "Sampled raw-action clipping, PPO probability-ratio clipping, requested motor saturation and target-slew occupancy are distinct reported quantities.", "Guard bounds mean old||new actor KL on each collected fixed rollout only; it does not bound physical motion or all states.", "Legacy transitions_per_s is inflated by pre-resume cumulative transitions; use allocation_new_transitions_per_s.", "Evaluation014 is outside this artifact input set; root reports its results separately."]}
    assert all(sha(ROOT/path) == digest for path,digest in INPUTS.items())
    report["input_sha256"] = INPUTS
    report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE/"REPORT.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checks_passed": True, "input_count": len(INPUTS), "report_sha256": sha(HERE/"REPORT.json"), "guard": report["rollback_guard"]},sort_keys=True))


if __name__ == "__main__":
    main()
