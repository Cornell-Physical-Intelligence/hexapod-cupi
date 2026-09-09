"""CPU kinematics for the measured, regularized MKII physical four-bar.

Matrices act on column vectors. Passive relations here construct valid initial
states and audit geometry; the simulator must integrate passive dynamics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from audit_mkii_stance import _origin, _rotation, primitive_bottom_z

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf"
PINS = ROOT / "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json"
CONTRACT = ROOT / "configs/mkii_fourbar_v3_kinematics.json"
USD_RELATIVE = "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
ACTIVE_SUFFIXES = ("coxa_yaw", "femur_pitch", "tibia_lever_pivot")
TREE_SUFFIXES = ACTIVE_SUFFIXES + ("tibia_pitch", "tibia_rod_pivot")
SCHEMA = "hexapod.mkii_fourbar_v3.kinematics.v1"

CONTRACT_NUMERIC_ATOL = 1e-14
CONTRACT_ANGLE_DIAGNOSTIC_ATOL = 2e-11


def compare_kinematic_contract(supplied, derived):
    """Exact structure/identity with tightly bounded platform math roundoff.

Mac NumPy versus Spark's SDK NumPy differed by <=4.45e-16 rad in actual
positions/limits and <=2.78e-17 m in clearance. The diagnostic acos(trace)
of a near-identity rotation amplifies roundoff to <=7.90e-12 rad; only that
named non-control field receives the larger tolerance. No relative tolerance,
type coercion, missing/extra keys, list reordering, NaN or infinity is accepted.
The authored JSON SHA-256 remains a separate exact identity check.
    """
    errors, differences = [], []
    def visit(a, b, path):
        label = '/'.join(path) or '<root>'
        if type(a) is not type(b):
            errors.append(f'{label}: type differs')
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                errors.append(f'{label}: keys differ')
            for key in sorted(a.keys() & b.keys()):
                visit(a[key], b[key], (*path, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                errors.append(f'{label}: list length differs')
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, (*path, str(index)))
        elif isinstance(a, float):
            diagnostic = (len(path) == 3 and path[0] == 'phase_bridge'
                          and path[1] in LEGS
                          and path[2] == 'stance_full_pose_orientation_residual_rad')
            tolerance = CONTRACT_ANGLE_DIAGNOSTIC_ATOL if diagnostic else CONTRACT_NUMERIC_ATOL
            if not math.isfinite(a) or not math.isfinite(b):
                errors.append(f'{label}: nonfinite number')
            elif a != b:
                delta = abs(a-b)
                differences.append({'path': label, 'absolute_difference': delta, 'tolerance': tolerance})
                if delta > tolerance:
                    errors.append(f'{label}: numeric difference {delta:.17g} exceeds {tolerance:.17g}')
        elif a != b:
            errors.append(f'{label}: value differs')
    visit(supplied, derived, ())
    return {'pass': not errors, 'errors': errors, 'bounded_numeric_differences': differences,
            'numeric_absolute_tolerance': CONTRACT_NUMERIC_ATOL,
            'orientation_diagnostic_absolute_tolerance_rad': CONTRACT_ANGLE_DIAGNOSTIC_ATOL}



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rz(q):
    t = np.eye(4)
    t[:3, :3] = _rotation(np.array([0., 0., 1.]), float(q))
    return t


def angle(rotation):
    return math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1)))


def load_model(urdf=URDF, pins=PINS):
    root, reference = ET.parse(urdf).getroot(), json.loads(Path(pins).read_text())
    expected = reference['sources']['robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf']['sha256']
    if sha256(urdf) != expected:
        raise ValueError("Linkage URDF does not match measured pin reference")
    joints = {j.get('name'): j for j in root.findall('joint')}
    links = {link.get('name'): link for link in root.findall('link')}
    if len(joints) != 30 or len(links) != 31 or len(root.findall('./joint/mimic')) != 12:
        raise ValueError("Expected source linkage 31 bodies, 30 tree joints and 12 removable mimics")
    if set(joints) != {f'{leg}_{suffix}' for leg in LEGS for suffix in TREE_SUFFIXES}:
        raise ValueError("Unexpected source joint identity")
    return root, reference, joints


def joint_frames(root, reference):
    """Return all 36 physical hinges; preserve upstream URDF frames exactly."""
    source = {j.get('name'): j for j in root.findall('joint')}
    result = {}
    for leg in LEGS:
        for suffix in ACTIVE_SUFFIXES[:2]:
            name = f'{leg}_{suffix}'
            j = source[name]
            alignment = np.eye(4)
            axis = np.array([float(v) for v in j.find('axis').get('xyz').split()])
            if np.allclose(axis, [0, 0, -1], atol=0, rtol=0):
                alignment[:3, :3] = np.diag([1., -1., -1.])
            elif not np.allclose(axis, [0, 0, 1], atol=0, rtol=0):
                raise ValueError(f'Unsupported upstream axis {name}')
            result[name] = {'body0': j.find('parent').get('link'),
                            'body1': j.find('child').get('link'),
                            'body0_from_hinge_matrix': (_origin(j) @ alignment).tolist(),
                            'body1_from_hinge_matrix': alignment.tolist(),
                            'actuated': True, 'exclude_from_articulation': False}
        for hinge in reference['legs'][leg]['proposed_hinges'].values():
            result[hinge['joint_name']] = {key: hinge[key] for key in (
                'body0', 'body1', 'body0_from_hinge_matrix',
                'body1_from_hinge_matrix', 'actuated', 'exclude_from_articulation')}
    return result


def relative_pose(frame, q):
    return (np.array(frame['body0_from_hinge_matrix']) @ rz(q)
            @ np.linalg.inv(np.array(frame['body1_from_hinge_matrix'])))


def forward_kinematics(frames, positions):
    tree = {name: frame for name, frame in frames.items() if not frame['exclude_from_articulation']}
    if set(positions) != set(tree) or not all(math.isfinite(float(q)) for q in positions.values()):
        raise ValueError('Positions must name all 30 finite tree coordinates exactly once')
    transforms = {'body': np.eye(4)}
    pending = dict(tree)
    while pending:
        before = len(pending)
        for name, frame in list(pending.items()):
            if frame['body0'] in transforms:
                if frame['body1'] in transforms:
                    raise ValueError('Repeated child / cyclic articulation')
                transforms[frame['body1']] = transforms[frame['body0']] @ relative_pose(frame, positions[name])
                del pending[name]
        if len(pending) == before:
            raise ValueError('Disconnected / cyclic articulation')
    return transforms


def expand_active(active):
    expected = {f'{leg}_{suffix}' for suffix in ACTIVE_SUFFIXES for leg in LEGS}
    if set(active) != expected:
        raise ValueError('Expected the 18 named active coordinates')
    result = dict(active)
    for leg in LEGS:
        result[f'{leg}_tibia_pitch'] = active[f'{leg}_tibia_lever_pivot']
        result[f'{leg}_tibia_rod_pivot'] = -active[f'{leg}_tibia_lever_pivot']
    return result


def old_knee_phase(source_joint, frame, canonical_q):
    """Exact nearest-Z rotation phase, unwrapped about the CAD branch.

The new pin axis differs slightly from the old URDF axis. A scalar phase
cannot make the old and corrected full transforms identical. This explicitly
defined projection supplies an invertible angle bridge, with residuals recorded.
    """
    old_from_new = np.linalg.inv(_origin(source_joint)) @ relative_pose(frame, canonical_q)
    r = old_from_new[:3, :3]
    phase = math.atan2(r[1, 0]-r[0, 1], r[0, 0]+r[1, 1])
    r0 = (np.linalg.inv(_origin(source_joint)) @ relative_pose(frame, 0))[:3, :3]
    zero = math.atan2(r0[1, 0]-r0[0, 1], r0[0, 0]+r0[1, 1])
    return phase + 2*math.pi*round((zero + canonical_q-phase)/(2*math.pi))


def inverse_old_phase(source_joint, frame, old_q):
    lo, hi = -math.pi, math.pi
    if not old_knee_phase(source_joint, frame, lo) < old_q < old_knee_phase(source_joint, frame, hi):
        raise ValueError('Old angle outside unwrapped physical branch')
    for _ in range(65):
        mid = (lo+hi)/2
        if old_knee_phase(source_joint, frame, mid) < old_q:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def closure_errors(frames, positions):
    poses = forward_kinematics(frames, positions)
    rows = {}
    for name, f in frames.items():
        if f['exclude_from_articulation']:
            a = poses[f['body0']] @ np.array(f['body0_from_hinge_matrix'])
            b = poses[f['body1']] @ np.array(f['body1_from_hinge_matrix'])
            rows[name] = {'point_error_m': float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
                          'axis_error_rad': float(np.linalg.norm(np.cross(a[:3, 2], b[:3, 2]))),
                          'axis_dot': float(a[:3, 2] @ b[:3, 2])}
    return rows


def collision_heights(root, poses, root_height=0.):
    rows = []
    for link in root.findall('link'):
        name = link.get('name')
        for index, collision in enumerate(link.findall('collision')):
            shapes = list(collision.find('geometry'))
            if len(shapes) != 1:
                raise ValueError('One collision shape required')
            rows.append({'link': name, 'index': index, 'shape': shapes[0].tag,
                         'foot': name.endswith('_tibia') and shapes[0].tag == 'sphere',
                         'bottom_z_m': primitive_bottom_z(poses[name] @ _origin(collision), shapes[0]) + root_height})
    return rows


def make_contract(urdf=URDF, pins=PINS):
    root, reference, source = load_model(urdf, pins)
    frames = joint_frames(root, reference)
    active = tuple(f'{leg}_{suffix}' for suffix in ACTIVE_SUFFIXES for leg in LEGS)
    tree = tuple(f'{leg}_{suffix}' for suffix in TREE_SUFFIXES for leg in LEGS)
    limits, defaults, bridge, relations = {}, {}, {}, {}
    for leg in LEGS:
        for suffix, default in [('coxa_yaw', 0.), ('femur_pitch', -.25)]:
            name = f'{leg}_{suffix}'
            limits[name] = [float(source[name].find('limit').get(k)) for k in ['lower', 'upper']]
            defaults[name] = default
        knee, lever, rod = (f'{leg}_{s}' for s in ['tibia_pitch', 'tibia_lever_pivot', 'tibia_rod_pivot'])
        old_limit = [float(source[knee].find('limit').get(k)) for k in ['lower', 'upper']]
        mapped = [inverse_old_phase(source[knee], frames[knee], q) for q in old_limit]
        stance = inverse_old_phase(source[knee], frames[knee], -.55)
        for name in [knee, lever]:
            limits[name] = mapped
            defaults[name] = stance
        limits[rod] = [-mapped[1], -mapped[0]]
        defaults[rod] = -stance
        relations[knee] = {'source_joint': lever, 'multiplier': 1., 'offset_rad': 0.}
        relations[rod] = {'source_joint': lever, 'multiplier': -1., 'offset_rad': 0.}
        old_pose = _origin(source[knee]) @ rz(-.55)
        new_pose = relative_pose(frames[knee], stance)
        residual = np.linalg.inv(old_pose) @ new_pose
        bridge[leg] = {
            'definition': 'nearest-Z rotation phase of inverse(old URDF knee origin) @ corrected femur_from_tibia(q), continuously unwrapped about raw CAD',
            'canonical_zero_old_knee_phase_rad': old_knee_phase(source[knee], frames[knee], 0.),
            'old_knee_stance_rad': -.55, 'canonical_stance_rad': stance,
            'old_knee_limits_rad': old_limit, 'canonical_limits_rad': mapped,
            'stance_phase_error_rad': old_knee_phase(source[knee], frames[knee], stance)+.55,
            'stance_full_pose_orientation_residual_rad': angle(residual[:3, :3]),
            'stance_full_pose_translation_residual_m': float(np.linalg.norm(residual[:3, 3])),
            'source_mimic_limits_used': False,
            'limit_scope': 'Provisional old knee operating interval transformed through exact scalar phase; old lever/rod mimic limits and 0.5 effort are not motor limits or measured stops',
            'toggle_canonical_q_rad': reference['legs'][leg]['toggle_diagnostic']['canonical_q_at_nearest_flattened_parallelograms_rad'],
        }
    poses = forward_kinematics(frames, defaults)
    heights = collision_heights(root, poses)
    feet = {leg: min(row['bottom_z_m'] for row in heights if row['foot'] and row['link'] == f'{leg}_tibia') for leg in LEGS}
    nominal = math.ceil(-min(feet.values())*1e6)/1e6
    reset = nominal+.005
    paths = {'body': 'Geometry/body'}
    for leg in LEGS:
        paths[f'{leg}_coxa'] = f'Geometry/body/{leg}_coxa'
        paths[f'{leg}_femur'] = paths[f'{leg}_coxa']+f'/{leg}_femur'
        for suffix in ['tibia', 'tibia_push_lever']:
            paths[f'{leg}_{suffix}'] = paths[f'{leg}_femur']+f'/{leg}_{suffix}'
        paths[f'{leg}_tibia_pushrod'] = paths[f'{leg}_tibia_push_lever']+f'/{leg}_tibia_pushrod'
    return {'schema': SCHEMA, 'asset_name': 'mkii_fourbar_v3',
            'usd_path_relative': USD_RELATIVE, 'usd_default_prim': 'Robot',
            'source_sha256': {'linkage_urdf': sha256(urdf), 'cad_pin_frames': sha256(pins)},
            'coordinate_convention': 'Z up, anatomical forward -bodyY, left +bodyX; SI radians/metres; matrices act on column vectors',
            'fourbar_zero': 'Raw assembly CAD relative to each retained upstream femur frame; not hardware encoder zero',
            'active_joint_names': list(active), 'tree_joint_names': list(tree),
            'closure_joint_names': [f'{leg}_tibia_loop_closure' for leg in LEGS],
            'passive_relations': relations, 'default_joint_positions_rad': defaults,
            'joint_limits_rad': limits, 'joint_frames': frames, 'body_paths': paths,
            'nominal_height_m': nominal, 'reset_root_height_m': reset,
            'reset_joint_jitter_rad': 0., 'reset_clearance_m': .005,
            'reset_scope': 'Nominal kinematic pose; no independent passive jitter; static equilibrium and dynamic closure require live qualification',
            'nominal_foot_clearance_m': {leg: q+reset for leg, q in feet.items()},
            'nominal_nonfoot_min_clearance_m': min(r['bottom_z_m']+reset for r in heights if not r['foot']),
            'phase_bridge': bridge,
            'actuator_model': 'Specified separately by a versioned motor contract; geometry alone does not certify RS05 torque-speed/thermal behavior',
            'physical_validation': 'not_performed'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=CONTRACT)
    args = parser.parse_args()
    result = make_contract()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit('Refusing to overwrite a kinematic contract')
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'contract': str(args.out), 'reset_root_height_m': result['reset_root_height_m'], 'nominal_foot_clearance_m': result['nominal_foot_clearance_m']}))
