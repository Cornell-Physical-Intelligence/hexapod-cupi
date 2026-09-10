#!/usr/bin/env python3
"""Verify that a coordinate rebase preserves the frozen CAD assembly pose.
Uses each model's recorded CAD coordinates, not its new neutral pose. This
compares transform and mass-tensor bookkeeping, not hardware identification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_updated_usd import from_record, transform
from audit_joint_review_usd import evaluate_graph


def pose_data(path):
    d = json.loads(path.read_text())
    root = transform([0, 0, d['cad_root_height_m']], np.eye(3))
    edges = {j['name']: {'parent': j['parent'], 'child': j['child'],
                         'frame0': from_record(j), 'frame1': np.eye(4)} for j in d['joints']}
    frames = evaluate_graph(root, edges, d['cad_pose'])
    parts = {p['id']: {'mesh': p['mesh'], 'frame': frames[p['link']] @ from_record(p)} for p in d['parts']}
    masses = {}
    for link in d['links']:
        frame = frames[link['name']]
        masses[link['name']] = {'mass': link['mass'],
            'com': frame[:3, :3] @ np.asarray(link['com']) + frame[:3, 3],
            'tensor': frame[:3, :3] @ np.asarray(link['inertia']) @ frame[:3, :3].T}
    return parts, masses


def check(old_path, new_path):
    old_parts, old_mass = pose_data(old_path)
    new_parts, new_mass = pose_data(new_path)
    errors = []
    if set(old_parts) != set(new_parts) or set(old_mass) != set(new_mass):
        raise ValueError('Part/body membership changed')
    max_position, max_rotation = 0., 0.
    for part_id, old in old_parts.items():
        new = new_parts[part_id]
        if old['mesh'] != new['mesh']:
            errors.append('Part mesh changed: ' + str(part_id))
        max_position = max(max_position, float(abs(old['frame'][:3, 3] - new['frame'][:3, 3]).max()))
        max_rotation = max(max_rotation, float(abs(old['frame'][:3, :3] - new['frame'][:3, :3]).max()))
    mass_error = max(abs(old_mass[n]['mass'] - new_mass[n]['mass']) for n in old_mass)
    com_error = max(float(abs(old_mass[n]['com'] - new_mass[n]['com']).max()) for n in old_mass)
    tensor_error = max(float(abs(old_mass[n]['tensor'] - new_mass[n]['tensor']).max()) for n in old_mass)
    if max_position > 1e-8 or max_rotation > 1e-8:
        errors.append('CAD-pose visual placement changed under coordinate rebase')
    if mass_error > 1e-9 or com_error > 1e-8 or tensor_error > 1e-9:
        errors.append('CAD-pose body mass/COM/tensor changed under coordinate rebase')
    return {'pass': not errors, 'errors': errors,
        'status': 'coordinate-rebase invariance only; no new physical calibration',
        'old_model_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'new_model_sha256': hashlib.sha256(new_path.read_bytes()).hexdigest(),
        'parts': len(old_parts), 'bodies': len(old_mass),
        'max_part_position_error_m': max_position,
        'max_part_rotation_matrix_error': max_rotation,
        'max_body_mass_error_kg': mass_error,
        'max_body_world_com_error_m': com_error,
        'max_body_world_tensor_error_kg_m2': tensor_error}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('old_model', type=Path)
    p.add_argument('new_model', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new audit filename')
    report = check(args.old_model, args.new_model)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
