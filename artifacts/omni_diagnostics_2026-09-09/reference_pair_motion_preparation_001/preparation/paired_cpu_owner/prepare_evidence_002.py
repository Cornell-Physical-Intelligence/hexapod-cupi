"""Bounded CPU fixture comparisons; no actual motion, contact or torque prediction."""
from pathlib import Path
import hashlib,json
import numpy as np
from run_baseline import evaluate
from state_packet import pack
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (H/'report_002.json').exists():raise FileExistsError('Preserve existing evidence')
results=[];baseline=None
for speed in (.01,.015,.02):
 r,rows,g,f=evaluate(speed=speed);results.append(r)
 if speed==.01:baseline=rows
 print(json.dumps(r,allow_nan=False),flush=True)
rejected,_,_,_=evaluate(duration=6,preload=.01)
packet=pack(baseline[-1]);schema=packet['schema'];schema['sha256']=packet['schema_sha256'];(H/'STATE_SCHEMA_002.json').write_text(json.dumps(schema,indent=2)+'\n')
values={k:np.stack([r[k] for r in baseline]) for k in ('q_ref','v_ref','a_ref','requested_command','admitted_target_command','admitted_command')}
values['time_s']=np.asarray([r['target_time_s'] for r in baseline]);values['completed_pairs']=np.asarray([r['state']['completed_pairs'] for r in baseline]);values['liftoffs']=np.asarray([r['state']['liftoffs'] for r in baseline]);values['touchdowns']=np.asarray([r['state']['confirmed_touchdowns'] for r in baseline])
values['active_pair_mask']=np.asarray([[n in (r['state']['active_pair'] or []) for n in ('lf','lm','lr','rf','rm','rr')] for r in baseline]);values['ideal_desired_position_world_m']=np.asarray([r['state']['desired_position_world_m'] for r in baseline]);values['minimum_measured_fixture_support_margin_m']=np.asarray([r['diagnostics']['measured_support_margin_m'] for r in baseline])
np.savez_compressed(H/'baseline_targets_002.npz',**values,joint_names=np.asarray(baseline[-1]['joint_names_runtime']))
sources={str(p.relative_to(H)):sha(p) for p in sorted(H.rglob('*.py')) if 'history_before_handoff_guard' not in p.parts}
report={'scope':'CPU-only paired contact-aware controller with ideal synthetic measured pose/contact streams; no actual physics, torque or speed qualification',
 'geometry_and_residual_sources':json.loads((H/'SOURCE_INPUTS.json').read_text()),'runtime_sources_sha256':sources,
 'cases':results,'constant_joint_deflection_fixture_rejection':rejected,
 'constant_joint_deflection_caveat':'Frozen joint deflection is not a physical contact model; it loses a retained ideal sharp-plane contact as stance angles change. The rejection and initial failed assertion are preserved. No contact tolerance or gate was relaxed.',
 'new_state_version':'pair_contact_motion_v001','observation_fragment_version':packet['version'],'fragment_features':schema['features'],'fragment_schema_sha256':schema['sha256'],
 'fragment_scope':'Explicit controller state only; no full actor/critic concatenation, tensor backend, normalization or PPO registration',
 'old846_849_scalar_packet_unchanged_and_incompatible':True,
 'tests':{'count':14,'log_sha256':sha(H/'tests_002.log'),'initial_failed_expectation_log_sha256':sha(H/'tests_initial.log'),'zero_residual_full_forward_and_stop':True,'each_foot_independently_qualified':True,'JSON_controller_motor_state_and_history_replay_exact':True,'new_reset_epoch_clears_all_history_slots':True},
 'actual_new_GPU_controls':0,'actual_contact_confirmations':0,'torque_tested_by_CPU':False,'actual_all_three_static_pairs_previously_passed':True,'new_moving_pair_physics_admitted':False,'PPO_ready':False,'Stage2_complete':False,'gates_weakened':False,
 'next_real_physics':'Separate fresh32standing+allquiet, then bounded zero-residual0.01m/s forward/stop paired case with four-retained-foot>=1N support,>=50mm measured margin, two independent>=2mm flights/three-sample stable landings, actual progress/stop, every400Hz torque and named-frame/source/asset checks; preserve original wave gates separately.'}
(H/'report_002.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('fragment features',schema['features'],schema['sha256'])
