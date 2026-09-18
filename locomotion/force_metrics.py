"""Summarize recorded 400 Hz contact-normal loads and motor torque."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .env_config import JOINT_NAMES, LEGS, MASS_KG, MODEL_SHA256

DT = .0025
FIELDS = ('time_s', 'sequence', 'explicit_counter', 'command', 'distal_force_world_n',
          'nonfoot_force_world_n', 'applied_torque_nm', 'computed_torque_nm')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {'mean': float(values.mean()), 'rms': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.quantile(values, .95, method='linear')), 'peak': float(values.max())}


def window(raw, mask, replica):
    count = int(mask.sum())
    if not count:
        return {'status': 'empty', 'samples': 0, 'duration_s': 0., 'metrics': None}
    feet = raw['distal_force_world_n'][mask, replica].astype(np.float64)
    other = raw['nonfoot_force_world_n'][mask, replica].astype(np.float64)
    magnitude = np.linalg.norm(feet, axis=-1)
    total = feet.sum(axis=1)+other.sum(axis=1)
    foot_rows = {}
    for index, leg in enumerate(LEGS):
        values = magnitude[:, index]
        contact = values > 1.
        foot_rows[leg] = {'normal_resultant_magnitude_n': stats(values),
            'contact_fraction': float(contact.mean()),
            'mean_magnitude_during_contact_n': float(values[contact].mean()) if contact.any() else None}
    motors = {}
    for key, label in [('applied_torque_nm', 'applied'), ('computed_torque_nm', 'requested')]:
        values = np.abs(raw[key][mask, replica].astype(np.float64))
        rms = np.sqrt(np.mean(values**2, axis=0))
        worst = int(np.argmax(rms))
        motors[label] = {'mean_abs_across_joints_nm': float(values.mean()),
            'worst_joint_rms_nm': float(rms[worst]), 'worst_joint': JOINT_NAMES[worst],
            'per_joint_abs_nm': {name: stats(values[:, i]) for i, name in enumerate(JOINT_NAMES)}}
    return {'status': 'available', 'samples': count, 'duration_s': count*DT,
        'first_sample_s': float(raw['time_s'][mask][0]), 'last_sample_s': float(raw['time_s'][mask][-1]),
        'total_vertical_support_n': stats(total[:, 2]),
        'mean_vertical_support_body_weights': float(total[:, 2].mean()/(MASS_KG*9.81)),
        'total_normal_resultant_magnitude_n': stats(np.linalg.norm(total, axis=-1)),
        'sum_foot_normal_magnitudes_n': stats(magnitude.sum(axis=1)),
        'per_foot': foot_rows, 'motor_torque': motors}


def summarize(raw, cases, *, settle_seconds=2.):
    if not np.isfinite(settle_seconds) or settle_seconds < 0:
        raise ValueError('Settling window must be finite and nonnegative')
    n = len(raw['time_s'])
    if not n:
        raise ValueError('No native force samples')
    feet = raw['distal_force_world_n']
    if feet.ndim != 4 or feet.shape[2:] != (6, 3):
        raise ValueError('Expected time, replica, six feet, XYZ force layout')
    replicas = feet.shape[1]
    shapes = {'time_s': (n,), 'sequence': (n,), 'explicit_counter': (n,),
        'command': (n, replicas, 3), 'distal_force_world_n': (n, replicas, 6, 3),
        'nonfoot_force_world_n': (n, replicas, 4, 3),
        'applied_torque_nm': (n, replicas, 18), 'computed_torque_nm': (n, replicas, 18)}
    for name, shape in shapes.items():
        value = raw[name]
        if value.shape != shape or value.dtype.kind not in 'fiub' or not np.isfinite(value).all():
            raise ValueError('Incomplete or nonfinite force channel: '+name)
    if not np.array_equal(raw['sequence'], np.arange(n)) or not np.all(np.diff(raw['explicit_counter']) == 1):
        raise ValueError('Noncontiguous native sequence or counter')
    if not np.allclose(raw['time_s'], (np.arange(n)+1)*DT, atol=1e-9, rtol=0):
        raise ValueError('Force averages require contiguous 400 Hz samples')
    if not cases or len(cases) > replicas or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Cases must identify distinct recorded replicas')
    if any(type(c['controls']) is not int or c['controls'] < 1 for c in cases):
        raise ValueError('Cases must declare a positive control count')
    after = raw['time_s'] > settle_seconds+1e-9
    rows = []
    for replica, case in enumerate(cases):
        moving = np.any(np.abs(raw['command'][:, replica]) > 1e-8, axis=1)
        rows.append({'case_id': case['case_id'], 'replica': replica,
            'expected_samples': case['controls']*8, 'recorded_samples': n,
            'requested_window_complete': n == case['controls']*8,
            'windows': {'full_trial': window(raw, np.ones(n, dtype=bool), replica),
                'startup': window(raw, ~after, replica),
                'commanded_locomotion_after_settle': window(raw, after & moving, replica)}})
    return rows


def report_for(directory, *, settle_seconds=2.):
    directory = Path(directory)
    declaration_path = directory/'declaration.json'
    capture_path = directory/'native400hz/capture.json'
    declaration = json.loads(declaration_path.read_text())
    capture = json.loads(capture_path.read_text())
    if declaration['model_sha256'] != MODEL_SHA256 or capture['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Force summary requires the named canonical model and joint order')
    files = capture['substep_files']
    if len(files) != len(set(files)):
        raise ValueError('Repeated native chunk')
    inputs = {'declaration.json': sha(declaration_path), 'native400hz/capture.json': sha(capture_path)}
    chunks = []
    for name in files:
        path = capture_path.parent/name
        if Path(name).name != name or path.is_symlink() or sha(path) != capture['files'][name]:
            raise ValueError('Missing, unsafe or changed native force chunk')
        with np.load(path, allow_pickle=False) as data:
            chunks.append({key: data[key].copy() for key in FIELDS})
        inputs['native400hz/'+name] = sha(path)
    if not chunks:
        raise ValueError('No native force chunks')
    raw = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in FIELDS}
    if ('assigned_case_ids' in declaration
            and len(declaration['assigned_case_ids']) != raw['distal_force_world_n'].shape[1]):
        raise ValueError('Native replica count differs from the case declaration')
    if len(raw['time_s']) != capture['steps'] or not np.array_equal(
            raw['explicit_counter'], capture['initial_counter']+np.arange(capture['steps'])+1):
        raise ValueError('Native capture length or counter binding differs')
    return {'schema': 'canonical_locomotion_force_metrics_v1', 'status': 'available',
        'source_directory': str(directory.resolve()), 'source_files': inputs, 'analysis_source_sha256': sha(__file__),
        'model_sha256': MODEL_SHA256, 'checkpoint_sha256': declaration.get('checkpoint_sha256'),
        'recording_source_sha256': declaration.get('source_sha256'), 'capture_failure': capture['failure'],
        'physics_hz': 400, 'settle_seconds': settle_seconds, 'reference_mass_kg': MASS_KG,
        'reference_weight_n': MASS_KG*9.81, 'units': {'contact': 'N', 'motor': 'N*m'},
        'definitions': {
            'force': 'World-frame sums of recorded patch normal force vectors; tangential friction is not recorded.',
            'vertical_support': 'World +Z component of the sum over all feet and nonfoot categories.',
            'per_foot': 'Magnitude of each foot normal-force resultant; full-window mean includes swing zeros.',
            'contact_mean': 'Conditional mean for foot resultant magnitude > 1 N; empty contact returns null.',
            'motor': 'Absolute applied and requested torque per joint; RMS is a load descriptor, not power or thermal qualification.',
            'averaging': 'Equal-duration 400 Hz samples, float64 reductions and linear p95 interpolation.',
            'locomotion_window': 'Nonzero commanded translation or yaw after the explicit settling interval; achieved speed does not filter rows.',
            'status': 'Available metrics describe captured rows, including failed prefixes; no acceptance gate changes.'},
        'cases': summarize(raw, declaration['cases'], settle_seconds=settle_seconds)}


def write_summary(directory, output, *, settle_seconds=2.):
    output = Path(output)
    # Reject output reuse before reading inputs; preserve original attempt files.
    with output.open('x') as stream:
        try:
            report = report_for(directory, settle_seconds=settle_seconds)
        except (ValueError, KeyError, OSError, TypeError) as error:
            report = {'schema': 'canonical_locomotion_force_metrics_v1', 'status': 'unavailable',
                      'error': repr(error), 'analysis_source_sha256': sha(__file__)}
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settle-seconds', type=float, default=2.)
    args = parser.parse_args()
    report = write_summary(args.directory, args.output, settle_seconds=args.settle_seconds)
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if report['status'] == 'available' else 1


if __name__ == '__main__':
    raise SystemExit(main())
