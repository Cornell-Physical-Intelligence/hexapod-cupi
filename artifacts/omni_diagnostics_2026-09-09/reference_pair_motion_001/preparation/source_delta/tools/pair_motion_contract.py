"""New bounded four-support paired forward/stop physics contract; no policy packet."""
from pathlib import Path
from types import SimpleNamespace
import json
from screen_contract import preflight as standing_preflight,digest
STARTUP_CONTROLS=200
PAIR_HOLD_CONTROLS=100
MOTION_START=300
MOTION_END=1500
STEPS=2400
CASES={'paired_forward':(.01,0.,0.)}
PROTOCOL={
 'name':'paired_contact_motion_physics001',
 'parent_source009_manifest_sha256':'04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',
 'paired_owner_freeze_sha256':'636844a6032642014304408f1021e61478be3aed46642fd67686ed9d2f9a6f55',
 'paired_controller_sha256':'d26e058241985e87a8dac37603d5e1147e89c637001b1136c30d8249af9d8a04',
 'paired_state_version':'pair_contact_motion_v001',
 'observation_fragment_version':'pair_contact_motion_fragment_v001',
 'observation_fragment_features':1014,
 'old846_849_packet_compatible':False,'actor_packet_emitted':False,'PPO_permitted':False,
 'fresh_standing':{'num_envs':32,'steps':1000,'all_physical_and_quiet_required':True},
 'cold_case':{'num_envs':1,'steps':STEPS,'seed':0,'dt_s':.02,'startup_controls':STARTUP_CONTROLS,
  'paired_reference_hold_controls':PAIR_HOLD_CONTROLS,'motion_start_control':MOTION_START,'motion_end_exclusive':MOTION_END,
  'motion_controls':1200,'stop_controls':900,'total_physical_s':48.,'paired_sequence_s':44.},
 'requested_forward_left_yaw':[.01,0.,0.],
 'pair_order':[['lm','rm'],['lf','rr'],['lr','rf']],
 'support_contract':'New four retained feet; each moving foot has independent actual2mm flight/apex/descent/three-confirmation landing. Original scalar five-support gate is unchanged.',
 'minimum_retained_supports':4,'minimum_measured_articulated_COM_margin_m':.05,
 'contact_classification':'Pinned runtime finite point, norm(normalForce)>1N and distal tibia local+Y threshold; all14 source-bound sensor clocks fresh after8normal updates',
 'no_completed_foot_contact_loss_before_partner':True,'all6supports_before_next_pair':True,
 'torque':'All recorded400Hz poststartup requested <=1.6Nm; applied <=1.60001Nm. Initialreset sampled separately, nothardwarestartupqualification.',
 'body_tracking_error_m':.035,'retained_toe_drift_m':.02,
 'target_budgets':{'reference_velocity_rad_s':1.75,'reference_acceleration_rad_s2':6.,'total_velocity_rad_s':2.,'total_acceleration_rad_s2':8.,'joint_reserve_rad':.02},
 'motion_fraction_required':.5,'original50Hz_displacement_integral_difference_m':.005,
 'quiet':'Same frozen QUIET_GATES;>=10s contiguous measured zero-command afterreferencequiet+2s settle; both reference and actual quiet required',
 'contact_sample_rate_hz':50,'pose_joint_torque_sample_rate_hz':400,
 'full_paired_state_recorded':True,'numerical_replay_absolute_tolerance':1e-8,
 'stage2_complete':False,'production_adoption':False,'terrain_qualified':False,
 'execution':'Fresh standing then exactly one paired forward/stop; fail closed, no automatic retry or PPO'}

def check_pair(source):
 source=Path(source);p=source/'tools/paired_runtime'
 if digest(p/'pair_motion.py')!=PROTOCOL['paired_controller_sha256']:raise ValueError('Exact reviewed paired controller required')
 owner=json.loads((p/'FREEZE_SHA256.json').read_text())
 if digest(p/'FREEZE_SHA256.json')!=PROTOCOL['paired_owner_freeze_sha256']:raise ValueError('Paired owner freeze differs')
 required=['pair_motion.py','checkpoint.py','state_packet.py','SOURCE_INPUTS.json','STATE_SCHEMA_002.json','CONTRACT.json']
 required+=sorted(json.loads((p/'SOURCE_INPUTS.json').read_text())['oracle_files'])
 for rel in required:
  if digest(p/rel)!=owner['files'][rel]:raise ValueError('Paired dependency changed: '+rel)
 if digest(p/'oracle/wave_reference.py')!=digest(source/'tools/wave_reference.py'):raise ValueError('Scalar wave oracle lineage differs')
 return required

def preflight(args,source):
 if args.mode!='paired_motion' or args.case!='paired_forward' or args.num_envs!=1 or args.steps!=STEPS or args.admission is None:
  raise ValueError('Only declared1x2400paired_forward afterfreshstanding is allowed')
 delegated=SimpleNamespace(**vars(args));delegated.mode='standing';delegated.num_envs=32;delegated.steps=1000;delegated.admission=None
 identity,geometry=standing_preflight(delegated,source);check_pair(source)
 receipt=json.loads(Path(args.admission).read_text());gate=receipt.get('gate',{})
 if (receipt.get('identity')!=identity or receipt.get('mode')!='standing' or receipt.get('status')!='completed' or gate.get('passed') is not True
     or gate.get('num_envs')!=32 or gate.get('control_steps')!=1000 or gate.get('all_replica_quiet',{}).get('passed') is not True):
  raise ValueError('Fresh exact-source32x1000physical ANDquiet admission required')
 return {'standing_identity':identity,'paired_motion_protocol':PROTOCOL,'case':args.case,
         'requested_forward_left_yaw':list(CASES[args.case]),'standing_admission_sha256':digest(Path(args.admission))},geometry
