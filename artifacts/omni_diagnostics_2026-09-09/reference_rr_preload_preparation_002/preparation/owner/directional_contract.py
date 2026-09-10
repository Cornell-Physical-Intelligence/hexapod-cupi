"""One RR preload diagnostic during left strafe; no wider policy or arc admission."""
from pathlib import Path
from types import SimpleNamespace
import json,math
from screen_contract import preflight as standing_preflight,digest
from rr_preload_contract import PROPOSAL as RR_PRELOAD_PROPOSAL
CASES={'left_strafe':(0.,.005,0.)}
PROTOCOL={'name':'reference_rr_first_landing_preload_diagnostic001','parent_source009_manifest_sha256':'04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',
 'parent_directional002_manifest_sha256':'7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58',
 'parent_directional001_manifest_sha256':'373c9ea08406f6f3593279fa9fda24badf56ae86bd938e140dba7d734b5ee919',
 'prior_measured_case_not_retried':{'case':'reverse','admitted':False,'original50Hz_gap_m':.005875744391232729,'bound_m':.005,'actual_review_freeze_sha256':'eeab7f65062d0545c4c4757a94adfdfe4eb3d2c9c6b1024fc738022ed72a4714'},
 'cases':{k:list(v) for k,v in CASES.items()},'case_order':list(CASES),
 'command_frame':'robot navigation: forward=-bodyY,left=+bodyX,yaw=+bodyZ; worldZ up, rawSDKquaternionXYZW',
 'fresh_standing':{'num_envs':32,'steps':1000,'all32_physical_and_quiet_required':True},
 'cold_case':{'num_envs':1,'steps':2400,'seed':0,'startup_controls':200,'motion_controls':1200,'stop_controls':1000,'dt_s':.02},
 'base_wave005_runtime_sha256':'8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893',
 'wave_runtime_sha256':'dbc0b046a9a6b46fae43ac8a3f01ce9162860a6ccb9deac54193f342671255e4',
 'preload_helper_sha256':'a844e65725607dc9fbc97d3585e7b03a8f2ec2d06ccf060ef600bb9235925c44',
 'preload_metadata_sha256':'3eef15aeedae1ede454d2069c2d5eb9bb5a8644b3966008f164c9c685e331969',
 'rr_preload_diagnostic':RR_PRELOAD_PROPOSAL,
 'base_wave_owner_freeze_sha256':'5b6c076cb03426ac8e24cc6df7e8684e48862aa78a495bd61ae49773a7826814',
 'recorded_wave_state_compatibility':{'observation_owner_freeze_sha256':'22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63','actor_fields':846,'critic_fields':849,
  'actual_runtime_joint_order_schema_sha256':'c2ab56b61c8ca56412090cd42cf109c90576bd5e78602a17eb29ada6d0559845','device_observation_packet_emitted_here':False,
  'inherited_fields_preserved_but_new_diagnostic_state_not_bound_to_old_actor':True,
  'old_checkpoint_or_observation_adoption_permitted':False},
 'directional_motion_fraction_required':.5,'original_world_position_velocity_integral_gap_m':.005,
 'pure_turn_max_planar_excursion_m':.01,'pure_turn_requires_nonzero_translation':False,
 'signed_yaw_actual_heading_and_reported_body_gyro_both_required':True,'angular_position_rate_difference_is_diagnostic_only':True,
 'no_command_derating':True,'same_target_filter_and_landing_state_machine':True,'original_physical_support_torque_flight_quiet_gates_retained':True,
 'PPO_permitted':False,'stage2_complete':False,'all_omni_directions_qualified':False,'production_adoption':False,
 'execution':'Fresh32standing then one cold left_strafe; preserve original failure gates; no automatic continuation',
 'arc003_is_not_this_trial':{'lost_support':'rm','rr_final_force_n':8.02679,'rm_final_force_n':.883063,'no_arc_fix_claim':True}}

def check_wave(source):
 if digest(Path(source)/'tools/wave_reference.py')!=PROTOCOL['wave_runtime_sha256']:raise ValueError('Exact separately bound RR diagnostic required; no changed source')
 if digest(Path(source)/'tools/rr_preload_diagnostic.py')!=PROTOCOL['preload_helper_sha256']:raise ValueError('Exact RR preload helper required')
 if digest(Path(source)/'tools/rr_preload_contract.py')!=PROTOCOL['preload_metadata_sha256']:raise ValueError('Exact stdlib RR metadata required')

def preflight(args,source):
 if args.mode!='directional' or args.case not in CASES or args.num_envs!=1 or args.steps!=2400 or args.admission is None:
  raise ValueError('Only one RR-diagnostic left_strafe1x2400 case after exact fresh standing is allowed')
 delegated=SimpleNamespace(**vars(args));delegated.mode='standing';delegated.num_envs=32;delegated.steps=1000;delegated.admission=None
 identity,geometry=standing_preflight(delegated,source);check_wave(source)
 receipt=json.loads(Path(args.admission).read_text());gate=receipt.get('gate',{})
 if (receipt.get('identity')!=identity or receipt.get('mode')!='standing' or receipt.get('status')!='completed' or gate.get('passed') is not True
     or gate.get('num_envs')!=32 or gate.get('control_steps')!=1000 or gate.get('all_replica_quiet',{}).get('passed') is not True):
  raise ValueError('Fresh exact-source32x1000 physical and quiet admission required')
 return {'standing_identity':identity,'directional_protocol':PROTOCOL,'case':args.case,'requested_forward_left_yaw':list(CASES[args.case]),
         'standing_admission_sha256':digest(Path(args.admission))},geometry
