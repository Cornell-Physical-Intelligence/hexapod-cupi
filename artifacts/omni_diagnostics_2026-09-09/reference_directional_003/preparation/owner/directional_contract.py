"""Two previously unmeasured named low-speed directional reference cases; no wider policy admission."""
from pathlib import Path
from types import SimpleNamespace
import json,math
from screen_contract import preflight as standing_preflight,digest
CASES={'left_turn':(0.,0.,.015),
       'forward_right_arc':(.005/math.sqrt(2),-.005/math.sqrt(2),-.01)}
PROTOCOL={'name':'reference_directional_discriminator003','parent_source009_manifest_sha256':'04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',
 'parent_directional001_manifest_sha256':'373c9ea08406f6f3593279fa9fda24badf56ae86bd938e140dba7d734b5ee919',
 'prior_measured_case_not_retried':{'case':'reverse','admitted':False,'original50Hz_gap_m':.005875744391232729,'bound_m':.005,'actual_review_freeze_sha256':'eeab7f65062d0545c4c4757a94adfdfe4eb3d2c9c6b1024fc738022ed72a4714'},
 'parent_directional002_manifest_sha256':'7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58',
 'prior_measured_strafe_not_retried':{'case':'left_strafe','admitted':False,'controls':505,'reason':'four admitted supports while LM swings; RR normal0.945591N below unchanged1N threshold','actual_review_freeze_sha256':'903a88c8bdf0a1ad5b4d06be26ed2cbbaafd7a5f223425b3dbdc841a8c6c59d9'},
 'cases':{k:list(v) for k,v in CASES.items()},'case_order':list(CASES),
 'command_frame':'robot navigation: forward=-bodyY,left=+bodyX,yaw=+bodyZ; worldZ up, rawSDKquaternionXYZW',
 'fresh_standing':{'num_envs':32,'steps':1000,'all32_physical_and_quiet_required':True},
 'cold_case':{'num_envs':1,'steps':2400,'seed':0,'startup_controls':200,'motion_controls':1200,'stop_controls':1000,'dt_s':.02},
 'wave_runtime_sha256':'8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893',
 'wave_owner_freeze_sha256':'5b6c076cb03426ac8e24cc6df7e8684e48862aa78a495bd61ae49773a7826814',
 'recorded_wave_state_compatibility':{'observation_owner_freeze_sha256':'22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63','actor_fields':846,'critic_fields':849,
  'actual_runtime_joint_order_schema_sha256':'c2ab56b61c8ca56412090cd42cf109c90576bd5e78602a17eb29ada6d0559845','device_observation_packet_emitted_here':False},
 'directional_motion_fraction_required':.5,'original_world_position_velocity_integral_gap_m':.005,
 'pure_turn_max_planar_excursion_m':.01,'pure_turn_requires_nonzero_translation':False,
 'signed_yaw_actual_heading_and_reported_body_gyro_both_required':True,'angular_position_rate_difference_is_diagnostic_only':True,
 'no_command_derating':True,'same_target_filter_and_landing_state_machine':True,'original_physical_support_torque_flight_quiet_gates_retained':True,
 'PPO_permitted':False,'stage2_complete':False,'all_omni_directions_qualified':False,'production_adoption':False,
 'execution':'Fresh independent cold cases after shared exact standing; stop campaign on first rejected case, preserve unrun cases as untested'}

def check_wave(source):
 if digest(Path(source)/'tools/wave_reference.py')!=PROTOCOL['wave_runtime_sha256']:raise ValueError('Exact wave005 required; no changed targets or state semantics')

def preflight(args,source):
 if args.mode!='directional' or args.case not in CASES or args.num_envs!=1 or args.steps!=2400 or args.admission is None:
  raise ValueError('Only two previously unmeasured named1x2400 cases after exact fresh standing are allowed')
 delegated=SimpleNamespace(**vars(args));delegated.mode='standing';delegated.num_envs=32;delegated.steps=1000;delegated.admission=None
 identity,geometry=standing_preflight(delegated,source);check_wave(source)
 receipt=json.loads(Path(args.admission).read_text());gate=receipt.get('gate',{})
 if (receipt.get('identity')!=identity or receipt.get('mode')!='standing' or receipt.get('status')!='completed' or gate.get('passed') is not True
     or gate.get('num_envs')!=32 or gate.get('control_steps')!=1000 or gate.get('all_replica_quiet',{}).get('passed') is not True):
  raise ValueError('Fresh exact-source32x1000 physical and quiet admission required')
 return {'standing_identity':identity,'directional_protocol':PROTOCOL,'case':args.case,'requested_forward_left_yaw':list(CASES[args.case]),
         'standing_admission_sha256':digest(Path(args.admission))},geometry
