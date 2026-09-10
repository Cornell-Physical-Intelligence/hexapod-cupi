from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;R=P.parent/'reference_physics_results_009';D=R/'run/wave'
a=json.loads((R/'remote_audit.json').read_text());assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in a['raw_payloads'].items())
with np.load(D/'trace.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
st=json.loads((D/'state.json').read_text());refs=json.loads((D/'reference_states.json').read_text());last=refs[-1]['result']['state'];t=d['time_s'][:,0]
assert len(t)==2400 and np.allclose(np.diff(t),.02,atol=1e-12)
assert np.array_equal(d['quaternion_world_wxyz'],d['quaternion_world_xyzw'][...,[3,0,1,2]])
events=[]
for leg in range(6):
 c=d['distal_contact'][:,0,leg];start=None
 for k in range(200,len(c)):
  if not c[k] and start is None:start=k
  if c[k] and start is not None:
   peak=float(d['reference_point_world_m'][start:k,0,leg,2].max());base=float(d['reference_point_world_m'][start-1,0,leg,2]);stable=bool(k+3<=len(c) and c[k:k+3].all());qual=bool(k-start>=2 and peak-base>=.002 and stable)
   events.append(dict(leg=str(d['legs'][leg]),off_start_index=start,off_samples=k-start,return_index=k,return_time_s=float(t[k]),measured_lift_m=peak-base,three_consecutive_return_contacts=stable,qualified_measured_event=qual));start=None
qual=[e for e in events if e['qualified_measured_event']]
stop=float(last['stop_requested_time_s']);quiet=float(last['reference_quiet_time_s']);idx=int(np.ceil(quiet/.02))+100
qst=st['gate']['final_quiet_stop_window'];assert qst['pass'];assert not d['requested_command'][idx:].any()
report={'raw_payloads_verified':len(a['raw_payloads']),'trace_sha256':hashlib.sha256((D/'trace.npz').read_bytes()).hexdigest(),'source_manifest_sha256':a['source_manifest_sha256'],'scope':'Independent raw contact/stop review of bounded009; notStage2/PPO','control_steps':len(t),'duration_s':len(t)*.02,'qualified_measured_events':len(qual),'qualified_leg_names':sorted(set(e['leg'] for e in qual)),'generator_confirmed_touchdowns':last['confirmed_touchdowns'],'raw_events':events,'zero_residual_actions_exact':bool(not d['raw_residual_action'].any()),'max_reference_execution_lag_rad':float(abs(d['reference_to_executable_lag_rad']).max()),'post_settle_min_distal_supports':int(d['distal_contact'][200:].sum(-1).min()),'post_settle_max_requested_torque_nm':float(abs(d['computed_torque_nm'][200:]).max()),'post_settle_nonfoot_samples':int(sum(np.count_nonzero(d[k][200:]) for k in ['shaft_contact','coxa_contact','femur_contact','base_contact'])),'termination_samples':int(np.count_nonzero(d['terminated'])+np.count_nonzero(d['truncated'])),'stop_request_s':stop,'reference_quiet_s':quiet,'reference_stop_latency_s':quiet-stop,'quiet_excluded_settling_s':2.,'scored_quiet_start_index':idx,'scored_quiet_duration_s':(len(t)-idx)*.02,'source_quiet_gate':qst,'source_independent_progress':st['gate']['independent_link_progress'],'stage2_complete':False,'actor_or_PPO_present':False}
assert len(qual)==11 and len(report['qualified_leg_names'])==6 and last['confirmed_touchdowns']==11
assert report['zero_residual_actions_exact'] and report['post_settle_min_distal_supports']>=5 and report['post_settle_max_requested_torque_nm']<=1.6 and report['termination_samples']==0 and report['post_settle_nonfoot_samples']==0
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['raw_events','source_quiet_gate']},indent=2))
