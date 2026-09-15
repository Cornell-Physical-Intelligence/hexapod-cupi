"""CPU-only descriptive nearest-demo analysis; no thresholds or native actions."""
from pathlib import Path
import collections
import datetime
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parents[2]
PINS = {}
SAVED = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    PINS[str(path.relative_to(ROOT))] = digest(path)
    return path


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values,.95)), "min": float(values.min()),
            "max": float(values.max())}


def distances(query, ref):
    q, r = query.astype(np.float64), ref.astype(np.float64)
    squared = (q*q).sum(1)[:,None] + (r*r).sum(1)[None,:] - 2*q@r.T
    return np.sqrt(np.maximum(squared,0)/q.shape[1])


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    torch.set_num_threads(1)
    pin(Path(__file__).resolve())
    source = pin(A/"source_017/learner.py")
    for path in (A/"source_017/FREEZE_SHA256.json", ROOT/"artifacts/paper_bc_data_002/ROW_PROVENANCE.jsonl", ROOT/"artifacts/paper_bc_data_002/SELECTION.json", ROOT/"artifacts/paper_bc_data_002/RECONSTRUCTION.json", A/"source_017/env.py", A/"results_evaluate_012/standing/state.json", ROOT/"artifacts/paper_bc_fit_004/REPORT.json"):
        pin(path)
    spec = importlib.util.spec_from_file_location("bc_onset_frozen_017",source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payloads = [torch.load(pin(ROOT/f"artifacts/paper_bc_fit_{n:03d}/candidate_checkpoint_update000000.pt"), map_location="cpu", weights_only=False) for n in (3,4)]
    assert all(p["schema"] == module.SCHEMA and p["learner_sha256"] == digest(source) for p in payloads)
    assert payloads[0]["model"].keys() == payloads[1]["model"].keys()
    assert all(torch.equal(v,payloads[1]["model"][k]) for k,v in payloads[0]["model"].items())
    assert all(p["counters"]["updates"] == p["counters"]["transitions"] == 0 for p in payloads)
    state = json.loads((A/"results_evaluate_012/standing/state.json").read_text())
    assert state["input_checkpoint_sha256"] == PINS["artifacts/paper_bc_fit_003/candidate_checkpoint_update000000.pt"]
    with np.load(pin(ROOT/"artifacts/paper_bc_data_002/bc_dataset.npz"),allow_pickle=False) as arrays:
        data = {k:arrays[k].copy() for k in arrays.files}
    with np.load(pin(A/"results_evaluate_012/standing/evaluation/batch_000/control_trace.npz"),allow_pickle=False) as arrays:
        trace = {k:arrays[k].copy() for k in arrays.files}
    obs, actual = data["observations"], trace["policy_observation"][:,0]
    assert obs.shape == (3760,231) and actual.shape == (1000,231)
    assert np.isfinite(obs).all() and np.isfinite(actual).all()
    expected = np.array([.05,0,0],np.float32)
    assert np.all(actual[:,210:213] == expected)
    model = module.ActorCritic(module.Config(**payloads[0]["config"])).eval()
    model.load_state_dict(payloads[0]["model"],strict=True)
    with torch.no_grad():
        normalized = model.obs_normalizer(torch.from_numpy(obs)).numpy()
        native_normalized = model.obs_normalizer(torch.from_numpy(actual)).numpy()
        demo_action = model.actor(torch.from_numpy(obs))[0].numpy()
        actor_action = model.actor(torch.from_numpy(actual))[0].numpy()
    recorded_action = trace["policy_action"][:,0]
    scale = .35
    groups = {"all":np.arange(231), "q_history":np.array([42*f+i for f in range(5) for i in range(6,24)]),
              "dq_history":np.array([42*f+i for f in range(5) for i in range(24,42)]),
              "gyro_history":np.array([42*f+i for f in range(5) for i in range(3)]),
              "gravity_history":np.array([42*f+i for f in range(5) for i in range(3,6)]),
              "held_target":np.arange(213,231), "command":np.arange(210,213)}
    forward = np.all(data["commands"] == expected,axis=1)
    pools = {"all_demonstrations":np.arange(len(obs)), "forward_all":np.flatnonzero(forward),
             "forward_onset":np.flatnonzero(forward & (data["source_kind"]==1)),
             "forward_steady":np.flatnonzero(forward & (data["source_kind"]==0))}
    assert len(pools["forward_onset"]) == len(pools["forward_steady"]) == 120
    assert set(data["control_index"][pools["forward_onset"]]) == set(range(200,260))
    windows = {"first100":slice(0,100), "later900":slice(100,1000)}
    report = {"schema":"bc_onset_distribution_diagnostic_v1", "started_utc":started,
              "scope":"Descriptive CPU inference on existing cold evaluation012 observations only. No warm trial or causal conclusion.",
              "fit003_fit004_model_and_normalizers_bitwise_equal":True,
              "cpu_actor_vs_recorded_native_max_absolute_action_error":float(np.abs(actor_action-recorded_action).max()),
              "normalization":"Saved fit003 actor mean/variance, sqrt(var+1e-6), actual actor clamp[-10,10]. Distances are RMS per included scalar, never a qualification threshold.",
              "feature_groups":{k:v.tolist() for k,v in groups.items()},
              "pool_composition":{k:{"rows":len(v),"raw_controls_min_max":[int(data['control_index'][v].min()),int(data['control_index'][v].max())],"replicas":np.unique(data['env_index'][v]).tolist()} for k,v in pools.items()},
              "nearest":{}, "training_input_fit":{}, "local_target_spread":{}, "coordinate_ranges":{}}
    for pool, ids in pools.items():
        report["nearest"][pool] = {}
        for name, columns in groups.items():
            dist = distances(native_normalized[:,columns],normalized[ids][:,columns])
            nearest = ids[dist.argmin(1)]
            minimum = dist.min(1)
            SAVED[pool+"__"+name+"__nearest_row"] = nearest
            SAVED[pool+"__"+name+"__distance"] = minimum
            report["nearest"][pool][name] = {w:summary(minimum[sl]) for w,sl in windows.items()}
            if name == "all":
                ref = distances(normalized[ids],normalized[ids])
                same_control = data["control_index"][ids,None] == data["control_index"][ids][None,:]
                ref[same_control] = np.inf
                baseline = ref.min(1)
                report["nearest"][pool]["demo_nearest_different_raw_control_baseline"] = summary(baseline)
                for w,sl in windows.items():
                    n = nearest[sl]
                    report["nearest"][pool][w+"_action_comparison"] = {
                        "recorded_actor_to_nearest_teacher_requested_target_rms_rad":summary(np.sqrt(np.mean(((recorded_action[sl]-data["actions"][n])*scale)**2,axis=1))),
                        "recorded_held_target_to_nearest_teacher_applied_target_rms_rad":summary(np.sqrt(np.mean((trace["joint_target_rad"][sl,0]-data["applied_joint_target_rad"][n])**2,axis=1))),
                        "nearest_source_kind_counts":dict(collections.Counter(str(x) for x in data["source_kind"][n])),
                        "nearest_command_counts":dict(collections.Counter(str(x.tolist()) for x in data["commands"][n]))}
                report["nearest"][pool]["selected_rows"] = [{"evaluation_control":int(t), "dataset_row":int(nearest[t]), "raw_control":int(data["control_index"][nearest[t]]), "replica":int(data["env_index"][nearest[t]]), "source_kind":int(data["source_kind"][nearest[t]]), "distance":float(minimum[t]), "group_rms_at_this_same_neighbor":{g:float(np.sqrt(np.mean((native_normalized[t,c]-normalized[nearest[t],c])**2))) for g,c in groups.items()}, "actual_recorded_actor_action":recorded_action[t].tolist(), "nearest_teacher_action":data["actions"][nearest[t]].tolist(), "previous_actual_action":actual[t,213:].tolist()} for t in (0,1,4,20,50,99,100,200,500,999)]
                if pool == "forward_all":
                    order = np.argsort(dist,axis=1)
                    for k in (1,4,8,16):
                        labels = data["actions"][ids[order[:,:k]]]
                        mean = labels.mean(1)
                        spread = np.sqrt(np.mean((labels-mean[:,None])**2,axis=(1,2)))*scale
                        err = np.sqrt(np.mean((recorded_action-mean)**2,axis=1))*scale
                        report["local_target_spread"][str(k)] = {w:{"neighbor_teacher_target_spread_rms_rad":summary(spread[sl]),"actor_to_neighbor_mean_rms_rad":summary(err[sl])} for w,sl in windows.items()}
    for pool,ids in pools.items():
        error = np.sqrt(np.mean(((demo_action[ids]-data["actions"][ids])*scale)**2,axis=1))
        report["training_input_fit"][pool] = {"actor_to_own_teacher_rms_rad":summary(error), "actor_minus_previous_held_rms_rad":summary(np.sqrt(np.mean(((demo_action[ids]-obs[ids,213:])*scale)**2,axis=1))), "teacher_minus_previous_held_rms_rad":summary(np.sqrt(np.mean(((data["actions"][ids]-obs[ids,213:])*scale)**2,axis=1)))}
    for name,columns in groups.items():
        lo,hi = obs[:,columns].min(0),obs[:,columns].max(0)
        outside = (actual[:,columns]<lo)|(actual[:,columns]>hi)
        report["coordinate_ranges"][name] = {w:{"scalar_fraction_outside_observed_demo_min_max":float(outside[sl].mean()),"rows_with_any_outside":int(outside[sl].any(1).sum())} for w,sl in windows.items()}
    SAVED.update(recorded_actor_action=recorded_action, cpu_actor_action=actor_action, demo_actor_action=demo_action)
    report["limitations"] = ["Nearest neighbor is a descriptive comparator, not the policy's actual retrieval mechanism or a success threshold.", "History frames and neighboring controls are correlated; dataset-to-dataset distances are not independent validation.", "All-command neighbors may have different teacher commands; same-command pools are reported separately.", "Teacher requested-target differences do not represent torque or an executable action replay; actuator clipping/slew and physical state are separate.", "Greater proximity to a neighborhood's mean cannot establish that target averaging causes failed walking.", "Cold evaluation starts from a physical reset; accepted forward-onset controls200-259 follow4s of native neutral settling.", "No warm-start result is read or inferred. Existing full Stage2 gates are unchanged."]
    report["input_sha256"] = PINS
    assert all(digest(ROOT/path) == value for path,value in PINS.items())
    report["inputs_unchanged"] = True
    report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE/"RESULT.json").open("x") as file:
        json.dump(report,file,indent=2,allow_nan=False);file.write("\n")
    with (HERE/"per_row.npz").open("xb") as file:
        np.savez_compressed(file,**SAVED)
    print(json.dumps({"completed":True,"input_count":len(PINS),"cpu_native_action_error":report["cpu_actor_vs_recorded_native_max_absolute_action_error"], "forward_all":report["nearest"]["forward_all"]["all"], "forward_onset":report["nearest"]["forward_onset"]["all"], "training_input_fit":report["training_input_fit"]}))


if __name__ == "__main__":
    main()
