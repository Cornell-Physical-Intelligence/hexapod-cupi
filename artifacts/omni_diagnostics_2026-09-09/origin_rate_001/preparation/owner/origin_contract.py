"""Source-bound matched-initial-state translation diagnostic; no admission reuse."""
from pathlib import Path
from types import SimpleNamespace
import json
from screen_contract import preflight as standing_preflight,digest
INITIAL_SHA='0f849142db07be3addf761dc8e3cdac18761f0104fff3d4cf8c76bad7f5cce38'
CASES={'origin_a':(0.,0.),'near':(3.,-5.),'far':(30.,-50.),'near_opposite':(-3.,5.),'origin_repeat':(0.,0.)}
PROTOCOL={'name':'matched_origin_rate_fidelity_001','parent_source009_manifest_sha256':'04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',
 'initial_state_sha256':INITIAL_SHA,'selected009replica':6,'cases':{k:list(v) for k,v in CASES.items()},
 'fresh_standing':{'replicas':32,'controls':1000,'physical_and_all32_quiet_required':True},
 'case':{'replicas':1,'controls':1000,'seed':0,'settle_controls':200,'startup_C2_s':2.,'canonical_hold_s':2.},
 'independent_variable':'GlobalXY translation only; common cold reset state, command/controller and infiniteZ0 contact geometry',
 'reset_state_injected_inside_reset_before_observed_physics':True,'body_pose_written_during_steps':False,
 'original_physical_and_quiet_thresholds_retained':True,'quiet_miss_is_reported_not_relabelled_pass':True,
 'rate_discrepancy_pass_threshold':None,'finite_difference_replaces_sdk_velocity':False,
 'PPO_permitted':False,'wave_permitted':False,'production_adoption':False,'stage2_complete':False,
 'native_warmstart_cache_copied':False,'maximum_cases':5}

def load_initial(source):
 path=Path(source)/'tools/origin_initial_state.json'
 if digest(path)!=INITIAL_SHA:raise ValueError('Matched initial-state identity differs')
 value=json.loads(path.read_text())
 if value['selected_environment']!=6 or value['source_manifest_sha256']!=PROTOCOL['parent_source009_manifest_sha256']:
  raise ValueError('Wrong selected initial-state lineage')
 return value

def preflight(args,source):
 if args.mode!='origin' or args.case not in CASES or args.num_envs!=1 or args.steps!=1000 or args.admission is None:
  raise ValueError('Only five named single-replica1000-control cases after fresh standing are allowed')
 delegated=SimpleNamespace(**vars(args));delegated.mode='standing';delegated.num_envs=32;delegated.steps=1000;delegated.admission=None
 standing_identity,geometry=standing_preflight(delegated,source)
 admission=json.loads(args.admission.read_text());gate=admission.get('gate',{})
 if (admission.get('identity')!=standing_identity or admission.get('mode')!='standing' or admission.get('status')!='completed'
     or gate.get('passed') is not True or gate.get('num_envs')!=32 or gate.get('control_steps')!=1000
     or gate.get('all_replica_quiet',{}).get('passed') is not True):
  raise ValueError('Fresh exact-source32x1000 physical and quiet admission required')
 load_initial(source)
 return {'standing_identity':standing_identity,'origin_protocol':PROTOCOL,'case':args.case,'translation_xy_m':list(CASES[args.case]),'standing_admission_sha256':digest(args.admission)},geometry
