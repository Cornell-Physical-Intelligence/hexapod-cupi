"""Reporting-only detail from already pinned cold observations and predictions."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parents[2]
paths = [HERE/"RESULT.json", HERE/"per_row.npz", Path(__file__).resolve(),
         ROOT/"artifacts/paper_bc_data_002/bc_dataset.npz",
         A/"results_evaluate_012/standing/evaluation/batch_000/control_trace.npz",
         A/"results_evaluate_012/standing/state.json"]
pins = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report = json.loads((HERE/"RESULT.json").read_text())
with np.load(paths[3]) as f:
    data = {k:f[k].copy() for k in f.files}
with np.load(paths[4]) as f:
    obs = f["policy_observation"][:,0].copy()
with np.load(HERE/"per_row.npz") as f:
    saved = {k:f[k].copy() for k in f.files}
state = json.loads(paths[5].read_text())
names = state["identity"]["config"]["joint_names"]
ids = np.flatnonzero(np.all(data["commands"] == np.array([.05,0,0],np.float32),axis=1) & (data["control_index"] == 200))
assert len(ids) == 2
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
initial=[]
for i in ids:
    previous = data["observations"][i,213:]
    teacher = data["actions"][i]
    predicted = saved["demo_actor_action"][i]
    initial.append({"dataset_row":int(i),"replica":int(data["env_index"][i]),
                    "teacher_minus_previous_rms_rad":rms((teacher-previous)*.35),
                    "fitted_actor_minus_previous_rms_rad":rms((predicted-previous)*.35),
                    "fitted_actor_to_own_teacher_rms_rad":rms((predicted-teacher)*.35),
                    "cold0_actor_to_this_teacher_rms_rad":rms((saved["recorded_actor_action"][0]-teacher)*.35),
                    "cold0_current_q_difference_rms_rad":rms(obs[0,174:192]-data["observations"][i,174:192]),
                    "cold0_current_q_difference_max_rad":float(np.abs(obs[0,174:192]-data["observations"][i,174:192]).max())})
later=[]
for j,name in enumerate(names):
    col=174+j
    lo,hi=float(data["observations"][:,col].min()),float(data["observations"][:,col].max())
    values=obs[100:,col]
    later.append({"joint":name,"quantity":"q minus neutral, radians","demo_range":[lo,hi],
                  "cold_later_mean":float(values.mean()),"cold_later_range":[float(values.min()),float(values.max())],
                  "cold_later_fraction_outside_demo_range":float(((values<lo)|(values>hi)).mean())})
baseline=report["nearest"]["forward_all"]["demo_nearest_different_raw_control_baseline"]
distance=saved["forward_all__all__distance"]
result={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "initial_forward_teacher_actions_identical_across_replicas":bool(np.array_equal(data["actions"][ids[0]],data["actions"][ids[1]])),
        "initial_forward_rows":initial,
        "cold0_actor_minus_previous_rms_rad":rms((saved["recorded_actor_action"][0]-obs[0,213:])*.35),
        "cold0_all_raw_features_within_entire_dataset_coordinate_ranges":bool(((obs[0]>=data["observations"].min(0))&(obs[0]<=data["observations"].max(0))).all()),
        "nearest_forward_distance_exceeds_observed_demo_different_control_max_fraction":{"first100":float((distance[:100]>baseline["max"]).mean()),"later900":float((distance[100:]>baseline["max"]).mean())},
        "nearest_distance_comparison_scope":"Empirical descriptive comparison only, not an OOD acceptance threshold; demonstrations share correlated trajectories.",
        "later_current_joint_pose_ranges":later,"input_sha256":pins}
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in pins.items())
with (HERE/"INITIAL_ACTION_DETAIL.json").open("x") as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("input_sha256","later_current_joint_pose_ranges")}))
