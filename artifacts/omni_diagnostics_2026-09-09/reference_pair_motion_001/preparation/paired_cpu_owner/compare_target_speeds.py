"""Separate target-only comparisons from the exact frozen planned-pair oracle.
No timed endpoint is a measured landing or contact-aware controller admission.
"""
from pathlib import Path
import hashlib,json,copy,numpy as np
from planned_target_oracle import evaluate
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((H/'inputs/planned_target_base_plan.json').read_text())
for category in ('source_files','inputs'):
 for f,h in plan[category].items():assert sha(H/f)==h,f
with np.load(H/'inputs/actual_pair001_startup_trace.npz',allow_pickle=False) as z:names=tuple(z['joint_names'].tolist());snapshot={k:z[k][-1].copy() for k in z.files if k!='joint_names'}
results=[]
for speed in (.015,.02):
 p=copy.deepcopy(plan);p['requested_forward_left_yaw']=[speed,0.,0.];p['candidate_id']='planned_opposing_pair_2s_forward_'+str(speed)
 r,data=evaluate(p,names,snapshot);np.savez_compressed(H/('target_only_'+str(speed)+'.npz'),**data,joint_names=np.asarray(names))
 results.append({'plan':p,'plan_sha256':hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'result':r})
 print(json.dumps({k:r[k] for k in ['candidate_id','completed_target_sequence','accepted_controls','failure','target_velocity_peak_rad_s','target_acceleration_peak_rad_s2','minimum_executed_joint_margin_rad']},allow_nan=False),flush=True)
report={'scope':'Target-only planned endpoint comparison; no new measured or synthetic-contact qualification','oracle_sha256':sha(H/'planned_target_oracle.py'),'base_plan_sha256':sha(H/'inputs/planned_target_base_plan.json'),'only_speed_and_candidate_identity_changed':True,'cases':results,'contact_aware_fixture_comparison':'report.json retains separate contact-aware .015/.02 ideal sharp-plane support failures; target-only results cannot overrule them','physics_admitted':False,'PPO_ready':False}
p=H/'target_speed_comparison.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
