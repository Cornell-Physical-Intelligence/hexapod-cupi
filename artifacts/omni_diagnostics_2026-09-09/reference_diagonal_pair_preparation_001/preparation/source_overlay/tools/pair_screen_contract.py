"""Separate full-C load-transfer diagnostic contract; no walking-gate mutation."""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from screen_contract import preflight as standing_preflight

PAIR_CASES={'lf_rr':{'pair_legs':['lf','rr'],'required_supports':['lm','lr','rf','rm']},'lr_rf':{'pair_legs':['lr','rf'],'required_supports':['lf','lm','rm','rr']}}
PAIR_PROTOCOL={
    'name':'diagonal_pairs_static_transfer_diagnostic001',
    'scope':'Separate proposed four-support diagnostic, not walking or Stage2 admission',
    'parent_source_manifest_sha256':'69a1a23304448d53f7681dfa87ab4fa652ad682ba0e45406e82187a66c4224f0',
    'pair_owner_freeze_sha256':'6cd797bee322d84328a6bbf9ff4aff289c57af45fbca27bc425d5e8d7a83e334',
    'num_envs':1,'startup_controls':200,'diagnostic_controls':1100,'total_controls':1300,
    'cases':PAIR_CASES,'case_order':['lf_rr','lr_rf'],'independent_cold_cases':True,
    'schedule_seconds_after_settled_reset':{'baseline':2.,'raise':3.,'hold_unloaded':2.,'return':3.,'quiet_including2s_settle':12.},
    'lift_m':.007,'measured_pair_clearance_m':.002,'unloaded_hold_s':1.,
    'proposed_bounds':{'corner_support_margin_m':.05,'body_displacement_m':.015,'body_angle_rad':.10,
        'corner_drift_m':.010,'corner_slip_mps':.020,'relative_vertical_force_balance_error':.05},
    'requested_torque_limit_nm':1.6,'applied_torque_limit_nm':1.60001,
    'complete400Hz_torque_and_joint_angles':True,'fresh32x1000_physical_and_unchanged_quiet_admission_required':True,
    'original_wave_gates_changed':False,'automatic_wave_or_PPO':False,'production_adoption':False,
    'measured_contact_scope':'50Hz distal/nonfoot classifications plus complete400Hz torque; not fullsubstep collision classification'}


def preflight(args,source):
    if getattr(args,'pair_case',None) not in PAIR_CASES:raise ValueError('Only the two unmeasured named diagonal cases are allowed')
    if args.mode!='pair' or args.num_envs!=1 or args.steps!=1300:
        raise ValueError('Only separately identified1x1300 diagonal-pair diagnostic is permitted')
    if args.admission is None:raise ValueError('Fresh exact-source standing admission required')
    delegated=SimpleNamespace(**vars(args));delegated.mode='standing';delegated.num_envs=32;delegated.steps=1000;delegated.admission=None
    identity,geometry=standing_preflight(delegated,source)
    admitted=json.loads(Path(args.admission).read_text());gate=admitted.get('gate',{})
    if (admitted.get('mode')!='standing' or admitted.get('identity')!=identity or admitted.get('status')!='completed'
        or not gate.get('passed') or gate.get('num_envs')!=32 or gate.get('control_steps')!=1000
        or not gate.get('all_replica_quiet',{}).get('passed')):
        raise ValueError('Pair diagnostic requires exact current-source32x1000standing and all32quiet')
    return {'standing_identity':identity,'pair_protocol':PAIR_PROTOCOL,'pair_case':args.pair_case,'pair_partition':PAIR_CASES[args.pair_case],
            'standing_admission_sha256':hashlib.sha256(Path(args.admission).read_bytes()).hexdigest()},geometry
