"""Read-only raw standing and early-failure replay; no application or GPU."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=REPO/'tmp/reference_physics_adapter_009/source_009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((ROOT/'remote_audit.json').read_text())
for name,h in a['raw_payloads'].items():assert sha(ROOT/name)==h,name
remote_map=json.loads((REPO/'tmp/reference_residual_ppo_cli_fix_001/raw/complete_run_sha256.json').read_text())
assert {name[4:]:h for name,h in a['raw_payloads'].items() if name.startswith('run/')}==remote_map
assert len(remote_map)==19
assert json.loads((ROOT/'run/inputs/study_before.sha256.json').read_text())==json.loads((REPO/'tmp/reference_physics_results_009/run/inputs/study_before.sha256.json').read_text())
sm=json.loads((SOURCE/'campaign_source_hashes.json').read_text());assert len(sm)==926
for name,h in sm.items():assert sha(SOURCE/name)==h
sys.path.insert(0,str(SOURCE/'tools'))
from screen_metrics import standing_screen,standing_quiet_review
with np.load(ROOT/'run/standing/trace.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();d={k:z[k] for k in z.files if k not in ('joint_names','legs')}
with np.load(ROOT/'run/standing/physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k] for k in z.files}
physical=standing_screen(d);quiet=standing_quiet_review(d,names)
assert physical['passed'] and quiet['passed'] and len(quiet['per_environment'])==32
assert d['time_s'].shape==(1000,32) and len(sub['time_s'])==8001
np.testing.assert_array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(1000),8)])
np.testing.assert_array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),1000)])
np.testing.assert_array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(8001))
np.testing.assert_allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,rtol=0,atol=1e-7)
for field,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[field][8::8],d[control])
campaign=json.loads((ROOT/'run/campaign.json').read_text());state=json.loads((ROOT/'run/standing/state.json').read_text())
assert campaign['status']=='failed' and campaign['PPO_updates_completed']==0 and set(campaign['accepted_phases'])=={'standing'}
assert state['status']=='completed' and state['control_steps']==1000 and state['gate']['passed']
assert sha(ROOT/'run/standing/state.json')==sha(ROOT/'run/standing/admission.json')==campaign['accepted_phases']['standing']['admission_sha256']
log=(ROOT/'run/logs/smoke.log').read_text();assert '/outputs/cuda:0/campaign.json' in log and 'REFERENCE_SCREEN_APP_START' not in log and 'REFERENCE_SCREEN_APP_READY' not in log
assert not any('/smoke/' in name or '/evaluate_' in name or name.endswith('.pt') for name in a['raw_payloads'])
report={'scope':'Independent complete001 raw audit; preparation002 is separate and has no physical result here','campaign_status':'failed','failure':'Early argument parser abbreviates --device into --device-run, overwriting exact proof path before AppLauncher','standing32_controls_each':1000,'standing_physical_pass':True,'standing_quiet_pass':True,'quiet_passed_replicas':sum(r['pass'] for r in quiet['per_environment']),'min_distal_supports_postsettle':int(d['distal_contact'][200:].sum(-1).min()),'postsettle400Hz_requested_peak_nm':float(np.abs(sub['computed_torque_nm'][1601:]).max()),'postsettle400Hz_applied_peak_nm':float(np.abs(sub['applied_torque_nm'][1601:]).max()),'substeps_including_initial':8001,'substep_counter_cadence_endpoint_parity':True,'raw_run_files_verified':19,'all_raw_files_including_pause_and_preflight':25,'PPO_updates':0,'smoke_AppLauncher_started':False,'calibration_started':False,'checkpoint_created':False,'retention_started':False,'asset_map_values_equal_admitted009':True,'owned_names_absent':2,'recorded_owned_IDs_absent':1,'smoke_container_ID_unknown':True,'restored_unix':a['pause_restoration']['restored_unix'],'source_and_dependencies_unchanged':True,'Stage2_complete':False,'velocity_fidelity_qualified':False}
print(json.dumps(report,indent=2))
