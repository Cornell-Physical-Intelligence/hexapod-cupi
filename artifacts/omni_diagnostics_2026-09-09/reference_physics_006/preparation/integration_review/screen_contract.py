"""Fail-closed source/asset/phase contract for a bounded, policy-free screen."""
import hashlib
import json
import math
from pathlib import Path

from solver_comparison import PROTOCOL

VARIANT = 'f050_t060'
URDF_SHA = 'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c'
RUNTIME_TREE = 'abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
PLAN_SHA = '356b72f0e82b207ba6105cb42057643d21a5b911d0517661465dcfe57745d00f'
ACTUATOR = {'joint_names_expr': ['.*'], 'saturation_effort': 5.5, 'effort_limit': 1.6,
            'effort_limit_sim': 5.5, 'velocity_limit': 50.26548245743668,
            'velocity_limit_sim': 55.29203070318036, 'stiffness': 30., 'damping': .6,
            'armature': .0007, 'friction': .01, 'dynamic_friction': .01, 'viscous_friction': .002}
OPTIONS = dict(profile='formal_004', residual_radius_rad=.02, residual_velocity_rad_s=.25,
               residual_acceleration_rad_s2=2., total_acceleration_rad_s2=8.)
WAVE = dict(settle_steps=200, forward_steps=1200, stop_steps=1000,
            requested_forward_mps=.005, requested_left_mps=0., requested_yaw_rad_s=0.)
STARTUP = dict(duration_s=2., canonical_hold_before_scoring_s=2.,
               path='quintic_C2_target_only', physical_reset_noise_preserved=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_source(source):
    source = Path(source).resolve()
    path = source / 'campaign_source_hashes.json'
    mapping = json.loads(path.read_text())
    for name, expected in mapping.items():
        item = (source / name).resolve()
        if source not in item.parents or not item.is_file() or digest(item) != expected:
            raise ValueError('Changed or escaping frozen source: ' + name)
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual != set(mapping) | {'campaign_source_hashes.json'}:
        raise ValueError('Frozen source has extra or missing files')
    return digest(path)


def preflight(args, source):
    if args.mode != 'standing':
        raise ValueError('Source006 is standing-only; no wave or policy admission')
    if args.variant != VARIANT or args.stance_index != 0:
        raise ValueError('Only exact admitted full-C stance0 is in this screen')
    expected_envs, expected_steps = (32, 1000) if args.mode == 'standing' else (1, 2400)
    if args.num_envs != expected_envs or args.steps != expected_steps:
        raise ValueError('Exact32x1000standing or1x2400wave required')
    if Path(args.output).exists():
        raise FileExistsError('Use a fresh immutable output directory')
    manifest = json.loads((args.package / 'manifest.json').read_text())
    records = [r for r in manifest['variants'] if r['variant'] == VARIANT]
    if len(records) != 1:
        raise ValueError('Exactly one selected C variant record required')
    record = records[0]
    if digest(args.package / record['urdf']) != URDF_SHA or record['sha256'] != URDF_SHA:
        raise ValueError('Full C URDF identity changed')
    reference = json.loads(args.geometry_reference.read_text())
    if reference['urdf_sha256'] != URDF_SHA:
        raise ValueError('Reference geometry belongs to another asset')
    plan_path = args.package / 'training_plan.json'
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != PLAN_SHA or plan['physics_dt_s'] != .0025 or plan['decimation'] != 8:
        raise ValueError('Exact admitted plan and .0025x8 physics required')
    if manifest['actuator_config_snapshot'] != ACTUATOR:
        raise ValueError('Exact admitted motor gains, caps, velocity limits, friction and armature required')
    if (not math.isclose(record['femur_length_m'], .0725, abs_tol=1e-9)
            or not math.isclose(record['tibia_length_m'], .12600000844001769, abs_tol=1e-9)):
        raise ValueError('Selected C segment lengths changed')
    stance = plan['variants'][VARIANT]['stances'][0]
    expected = {manifest['link_joint_mapping'][leg]['joints'][part]: math.radians(degrees)
                for leg in ('lf','lm','lr','rf','rm','rr')
                for part, degrees in [('coxa',0),('femur',40),('tibia',120)]}
    if (set(stance['joint_positions_rad']) != set(expected)
            or any(not math.isclose(stance['joint_positions_rad'][name], value, abs_tol=1e-9) for name, value in expected.items())
            or not math.isclose(stance['root_height_at_contact_m'], .13053251856352807, abs_tol=1e-9)
            or not math.isclose(stance['suggested_reset_root_height_m'], .13653251856352808, abs_tol=1e-9)):
        raise ValueError('Exact admitted named40/120 stance and plate/reset heights required')
    identity = dict(variant=VARIANT, stance_index=0, urdf_sha256=URDF_SHA,
        plan_sha256=digest(plan_path),
        source_manifest_sha256=verify_source(source),
        geometry_reference_sha256=digest(args.geometry_reference),
        controller=OPTIONS, wave_schedule=WAVE, startup=STARTUP, runtime_tree_sha256=RUNTIME_TREE,
        solver_comparison=PROTOCOL)
    if args.mode == 'wave':
        if args.admission is None:
            raise ValueError('Fresh exact-source zero-residual standing admission required')
        admission = json.loads(args.admission.read_text())
        if admission.get('identity') != identity or admission.get('status') != 'completed' or not admission.get('gate', {}).get('passed'):
            raise ValueError('Mismatched or failed zero-residual standing admission')
    elif args.admission is not None:
        raise ValueError('Standing must start fresh, without another admission or checkpoint')
    return identity, reference
