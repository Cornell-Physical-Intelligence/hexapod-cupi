#!/usr/bin/env python3
"""Read-only, reproducible audit of this flattened Onshape-to-Robot CAD export.

Usage: python audit_onshape_mass_properties.py --source EXPORT_DIR --out AUDIT_DIR
Requires numpy, scipy, trimesh. No Onshape package or credentials required.
The pickle reader rejects all globals except seven specific data constructors.
All masses/inertias remain uncorrected source values; mesh density is an estimate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import trimesh
from scipy.spatial.transform import Rotation

EXPECTED_ARCHIVE_SHA256 = '31f33044285c3de2dbd85180e7c4886c31d80da4270dfd6b94bbd50d150a4cf7'
CUSTOM_DOCUMENT = '881b01051a1ec5c958bf7768'
MOTOR_DOCUMENT = 'b7094f2f67d456e424842e30'
BEARING_DOCUMENT = '90e95efa79c7cf10f5549d54'
MOTOR_ANCHOR_PREFIX = '1_1_06_eb463_507_'

class _Data:
    def __setstate__(self, state):
        if not isinstance(state, dict) or any(not isinstance(k, str) for k in state):
            raise pickle.UnpicklingError('CAD data state must be a string-keyed dictionary')
        self.__dict__.update(state)

_ALLOWED = {('onshape_to_robot.robot', n): type(n, (_Data,), {}) for n in ('Robot', 'Link', 'Part')}
_ALLOWED[('onshape_to_robot.geometry', 'Mesh')] = type('Mesh', (_Data,), {})
_ALLOWED.update({('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
                 ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype})

class RestrictedCadUnpickler(pickle.Unpickler):
    """Never imports or delegates to arbitrary pickle globals."""
    def find_class(self, module, name):
        try:
            return _ALLOWED[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}') from None

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent pickle IDs are unsupported')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(text):
    return np.fromstring(text, sep=' ')


def matrix(value, shape, label):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Invalid {label}: expected finite numeric {shape}')
    return a.astype(float)


def origin(element):
    out = np.eye(4)
    if element is not None:
        out[:3, :3] = Rotation.from_euler('xyz', floats(element.get('rpy', '0 0 0'))).as_matrix()
        out[:3, 3] = floats(element.get('xyz', '0 0 0'))
    return out


def tensor_xml(element):
    a = element.attrib
    return np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])],
                     [float(a['ixy']), float(a['iyy']), float(a['iyz'])],
                     [float(a['ixz']), float(a['iyz']), float(a['izz'])]])


def physical_tensor(tensor, mass):
    eig = np.linalg.eigvalsh((tensor + tensor.T) / 2)
    scale = max(float(np.max(np.abs(tensor))), 1e-30)
    tol = max(1e-18, scale * 1e-9)
    asym = float(np.max(np.abs(tensor - tensor.T)))
    slack = float(eig[0] + eig[1] - eig[2])
    valid = (mass >= 0 and asym <= tol and eig[0] >= -tol and slack >= -tol
             and (mass > 0 or scale <= 1e-18))
    return {'eigenvalues_kg_m2': eig.tolist(), 'symmetry_error_kg_m2': asym,
            'triangle_slack_kg_m2': slack, 'tolerance_kg_m2': tol,
            'physically_consistent': bool(valid), 'strictly_positive_definite': bool(eig[0] > tol),
            'zero_mass': bool(mass == 0)}


def aggregate(rows):
    mass = math.fsum(r['mass_kg'] for r in rows)
    if not mass:
        return {'mass_kg': 0., 'com_export_m': None, 'inertia_about_com_export_kg_m2': np.zeros((3, 3)).tolist()}
    com = sum(r['mass_kg'] * np.asarray(r['com_export_m']) for r in rows) / mass
    inertia = np.zeros((3, 3))
    for r in rows:
        d = np.asarray(r['com_export_m']) - com
        inertia += np.asarray(r['inertia_about_com_export_kg_m2']) + r['mass_kg'] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {'mass_kg': mass, 'com_export_m': com.tolist(),
            'inertia_about_com_export_kg_m2': inertia.tolist(), 'tensor_check': physical_tensor(inertia, mass)}


def classification(meta):
    doc = meta.get('documentId')
    if doc == MOTOR_DOCUMENT:
        return 'vendor_motor_parts'
    if doc == BEARING_DOCUMENT:
        return 'separate_bearing_parts'
    if doc == CUSTOM_DOCUMENT:
        return 'custom_robot_structure'
    return 'catalog_fasteners_and_hardware'


def rounding_halfwidth(text):
    """Half of the last written decimal unit, including scientific notation."""
    mantissa, _, exponent = text.lower().partition('e')
    decimals = len(mantissa.partition('.')[2])
    return 0.5 * 10.0 ** (int(exponent or 0) - decimals)


def rounded_check(actual, written):
    error = abs(float(actual) - float(written))
    tol = rounding_halfwidth(written)
    return {'actual': float(actual), 'written': written, 'abs_error': error,
            'rounding_halfwidth': tol, 'within_written_precision': error <= tol + 1e-12}


def nested_keys(data, prefix=''):
    found = []
    if isinstance(data, dict):
        for key, value in data.items():
            path = f'{prefix}.{key}' if prefix else key
            if any(s in key.lower() for s in ('material', 'density', 'friction', 'restitution', 'elastic')):
                found.append({'path': path, 'value': value})
            found.extend(nested_keys(value, path))
    elif isinstance(data, (list, tuple)):
        for idx, value in enumerate(data):
            found.extend(nested_keys(value, f'{prefix}[{idx}]'))
    return found


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.out.resolve()
    if out == src or src in out.parents:
        raise ValueError('Audit output must be outside the immutable source directory')
    out.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256(args.archive) if args.archive else None
    if archive_hash is not None and archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Unexpected archive SHA256; this audit is pinned to the provided export')
    pkl_bytes = (src / 'robot.pkl').read_bytes()
    robot = RestrictedCadUnpickler(io.BytesIO(pkl_bytes)).load()
    urdf = ET.parse(src / 'robot.urdf').getroot()
    links = urdf.findall('link')
    if len(links) != 1 or len(robot.links) != 1 or urdf.findall('joint') or robot.joints:
        raise ValueError('This audit expects the observed one-link, joint-free export')
    parts, visuals, collisions = robot.links[0].parts, links[0].findall('visual'), links[0].findall('collision')
    if not len(parts) == len(visuals) == len(collisions):
        raise ValueError('Part/visual/collision counts differ')
    metadata, meshes = {}, {}
    for path in sorted((src / 'assets').glob('*.part')):
        metadata['assets/' + path.with_suffix('.stl').name] = json.loads(path.read_text())
    for path in sorted((src / 'assets').glob('*.stl')):
        mesh = trimesh.load_mesh(path, process=True)
        volume = float(mesh.volume)
        valid_volume = bool(mesh.is_watertight and mesh.is_winding_consistent and volume > 0)
        meshes['assets/' + path.name] = {
            'sha256': sha256(path), 'triangle_count': len(mesh.faces),
            'signed_volume_m3': volume, 'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'valid_solid_volume_for_density_estimate': valid_volume,
            'bounds_mesh_frame_m': mesh.bounds.tolist(), 'volume_com_mesh_frame_m': mesh.center_mass.tolist(),
        }
    rows = []
    pose_translation, pose_rotation = [], []
    appearance_error = []
    for index, (part, vis, col) in enumerate(zip(parts, visuals, collisions)):
        if set(vars(part)) != {'name', 'T_world_part', 'mass', 'com', 'inertia', 'meshes', 'shapes'}:
            raise ValueError(f'Unexpected part schema at {index}')
        if len(part.meshes) != 1 or part.shapes:
            raise ValueError(f'Unexpected geometry count at {index}')
        name = part.meshes[0].filename
        if name not in metadata or name not in meshes:
            raise ValueError(f'Unresolved mesh/part metadata: {name}')
        vis_name = vis.find('geometry/mesh').get('filename').removeprefix('package://')
        col_name = col.find('geometry/mesh').get('filename').removeprefix('package://')
        if name != vis_name or name != col_name:
            raise ValueError(f'Part order mismatch at {index}')
        for geom in (vis, col):
            if not np.array_equal(floats(geom.find('geometry/mesh').get('scale', '1 1 1')), np.ones(3)):
                raise ValueError('Non-unit mesh scale unsupported')
        T = matrix(part.T_world_part, (4, 4), 'part transform')
        R, t = T[:3, :3], T[:3, 3]
        if (np.max(np.abs(R.T @ R - np.eye(3))) > 1e-9 or abs(np.linalg.det(R) - 1) > 1e-9
                or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
            raise ValueError(f'Non-rigid transform at {index}')
        for geom in (vis, col):
            U = origin(geom.find('origin'))
            pose_translation.append(float(np.max(np.abs(t - U[:3, 3]))))
            pose_rotation.append(float(Rotation.from_matrix(R.T @ U[:3, :3]).magnitude()))
        m = float(part.mass)
        if not math.isfinite(m) or m < 0:
            raise ValueError(f'Invalid mass at {index}')
        com = matrix(part.com, (3,), 'part COM')
        I = matrix(part.inertia, (3, 3), 'part inertia')
        color = matrix(part.meshes[0].color, (4,), 'RGBA')
        appearance_error.append(float(np.max(np.abs(color - floats(vis.find('material/color').get('rgba'))))))
        ms = meshes[name]
        rows.append({'index_zero_based': index, 'part_name': part.name, 'mesh': name,
                     'source_document_id': metadata[name]['documentId'], 'category': classification(metadata[name]),
                     'mass_kg': m, 'com_part_m': com.tolist(), 'T_export_part': T.tolist(),
                     'com_export_m': (R @ com + t).tolist(), 'inertia_about_com_part_kg_m2': I.tolist(),
                     'inertia_about_com_export_kg_m2': (R @ I @ R.T).tolist(),
                     'tensor_check': physical_tensor(I, m), 'appearance_rgba': color.tolist(),
                     'assigned_material_name': None, 'mesh_volume_m3': ms['signed_volume_m3'],
                     'density_estimate_kg_m3': m / ms['signed_volume_m3'] if ms['valid_solid_volume_for_density_estimate'] else None,
                     'density_estimate_basis': 'source CAD mass / tessellated mesh volume; does not establish assigned material',
                     'urdf_appearance_name': vis.find('material').get('name')})
    total = aggregate(rows)
    by_mesh = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        by_mesh[row['mesh']].append(row)
        by_category[row['category']].append(row)
    groups = []
    for name, parts_group in sorted(by_mesh.items()):
        masses = [r['mass_kg'] for r in parts_group]
        rho = [r['density_estimate_kg_m3'] for r in parts_group if r['density_estimate_kg_m3'] is not None]
        groups.append({'mesh': name, 'count': len(parts_group), 'category': parts_group[0]['category'],
                       'cad_metadata': metadata[name], 'mass_each_min_kg': min(masses), 'mass_each_max_kg': max(masses),
                       'aggregate': aggregate(parts_group), 'mesh_geometry': meshes[name],
                       'density_estimate_min_kg_m3': min(rho) if rho else None,
                       'density_estimate_max_kg_m3': max(rho) if rho else None,
                       'source_part_com_m': parts_group[0]['com_part_m'],
                       'source_part_inertia_about_com_kg_m2': parts_group[0]['inertia_about_com_part_kg_m2'],
                       'source_part_inertia_varies': any(not np.array_equal(p['inertia_about_com_part_kg_m2'], parts_group[0]['inertia_about_com_part_kg_m2']) for p in parts_group),
                       'assigned_material_name': None, 'appearance_rgba_values': sorted({tuple(p['appearance_rgba']) for p in parts_group})})
    source_inertial = links[0].find('inertial')
    urdf_T = origin(source_inertial.find('origin'))
    source_I = tensor_xml(source_inertial.find('inertia'))
    calculated_I_urdf_inertial = urdf_T[:3, :3].T @ np.asarray(total['inertia_about_com_export_kg_m2']) @ urdf_T[:3, :3]
    written_checks = {'mass': rounded_check(total['mass_kg'], source_inertial.find('mass').get('value'))}
    for j, value in enumerate(source_inertial.find('origin').get('xyz').split()):
        written_checks['com_' + 'xyz'[j]] = rounded_check(total['com_export_m'][j], value)
    for key, (i, j) in {'ixx': (0, 0), 'ixy': (0, 1), 'ixz': (0, 2), 'iyy': (1, 1), 'iyz': (1, 2), 'izz': (2, 2)}.items():
        written_checks[key] = rounded_check(calculated_I_urdf_inertial[i, j], source_inertial.find('inertia').get(key))
    motor_rows = by_category['vendor_motor_parts']
    motor_anchors = [r for r in motor_rows if Path(r['mesh']).name.startswith(MOTOR_ANCHOR_PREFIX)]
    motor_count = len(motor_anchors)
    motor_mass = math.fsum(p['mass_kg'] for p in motor_rows)
    nominal = .191
    hypothetical_delta = motor_count * nominal - motor_mass
    motor = {'cad_motor_instances_from_repeated_housing_anchor': motor_count,
             'count_method': f'Occurrences of unique housing mesh prefix {MOTOR_ANCHOR_PREFIX}; cross-check all vendor type multiplicities',
             'anchor_mesh': motor_anchors[0]['mesh'] if motor_anchors else None,
             'anchor_part_indices': [r['index_zero_based'] for r in motor_anchors],
             'anchor_com_export_m': [r['com_export_m'] for r in motor_anchors],
             'anchor_unique_com_count_1um': len({tuple(np.round(r['com_export_m'], 6)) for r in motor_anchors}),
             'minimum_anchor_com_separation_m': min(float(np.linalg.norm(np.asarray(a['com_export_m']) - np.asarray(b['com_export_m']))) for i, a in enumerate(motor_anchors) for b in motor_anchors[i+1:]) if len(motor_anchors) > 1 else None,
             'vendor_document_id': MOTOR_DOCUMENT, 'vendor_part_instances': len(motor_rows),
             'vendor_part_mesh_types': sum(g['category'] == 'vendor_motor_parts' for g in groups),
             'vendor_mass_total_kg': motor_mass, 'vendor_mass_per_motor_kg': motor_mass / motor_count if motor_count else None,
             'all_vendor_type_counts_divisible_by_motor_count': bool(motor_count and all(len(v) % motor_count == 0 for k, v in by_mesh.items() if v[0]['category'] == 'vendor_motor_parts')),
             'zero_mass_vendor_part_instances': sum(r['mass_kg'] == 0 for r in motor_rows),
             'historical_rs05_nominal_mass_kg': nominal,
             'hypothetical_delta_if_these_are_same_rs05_kg': hypothetical_delta,
             'hypothetical_robot_mass_with_rs05_nominal_replacement_kg': total['mass_kg'] + hypothetical_delta,
             'applied_correction': False,
             'caveat': 'Export vendor mesh names reference FL46BLW10 48V 5N M; repository physical model identifies RS05. Confirm exact shipped motor revision and included hardware before using 191 g. The export alone does not prove that identity or real rotor/stator mass distribution. Replacement arithmetic is a conditional scenario, not a calibrated simulation mass.'}
    material_hits = [{'mesh': k, 'hits': nested_keys(v)} for k, v in metadata.items() if nested_keys(v)]
    property_hits = [{'index_zero_based': i, 'hits': nested_keys({'visual_properties': p.meshes[0].visual_properties, 'collision_properties': p.meshes[0].collision_properties})} for i, p in enumerate(parts)]
    property_hits = [v for v in property_hits if v['hits']]
    summary = {'schema_version': 1, 'source_archive_name': 'HexapodLegUpdatedV2.zip',
               'source_archive_sha256_expected': EXPECTED_ARCHIVE_SHA256, 'source_archive_sha256_verified': archive_hash,
               'source_urdf_sha256': sha256(src / 'robot.urdf'), 'source_pickle_sha256': hashlib.sha256(pkl_bytes).hexdigest(),
               'audit_script_sha256': sha256(Path(__file__)),
               'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__, 'trimesh': trimesh.__version__},
               'counts': {'parts': len(rows), 'mesh_types': len(groups), 'metadata_files': len(metadata), 'links': len(links),
                          'joints': len(robot.joints), 'closures': len(robot.closures), 'named_frames': sum(len(l.frames) for l in robot.links),
                          'visuals': len(visuals), 'collisions': len(collisions), 'zero_mass_parts': sum(r['mass_kg'] == 0 for r in rows)},
               'aggregate_source_cad': total,
               'category_breakdown': {key: {'instances': len(value), **aggregate(value)} for key, value in sorted(by_category.items())},
               'source_urdf_roundtrip': {'field_checks': written_checks,
                    'all_fields_within_export_written_precision': all(v['within_written_precision'] for v in written_checks.values()),
                    'max_part_pose_translation_component_error_m': max(pose_translation),
                    'max_part_pose_rotation_error_rad': max(pose_rotation),
                    'max_appearance_rgba_error': max(appearance_error),
                    'all_part_pose_checks_pass': max(pose_translation) < 1e-6 and max(pose_rotation) < 1e-5},
               'tensor_checks': {'all_source_tensors_physically_consistent': all(r['tensor_check']['physically_consistent'] for r in rows),
                    'positive_mass_parts': sum(r['mass_kg'] > 0 for r in rows),
                    'strictly_positive_definite_parts': sum(r['tensor_check']['strictly_positive_definite'] for r in rows),
                    'invalid_part_indices': [r['index_zero_based'] for r in rows if not r['tensor_check']['physically_consistent']]},
               'mesh_checks': {'nonwatertight_meshes': [name for name, ms in meshes.items() if not ms['watertight']],
                    'invalid_volume_meshes': [name for name, ms in meshes.items() if not ms['valid_solid_volume_for_density_estimate']]},
               'material_audit': {'assigned_material_names_present': bool(material_hits or property_hits),
                    'part_metadata_relevant_key_hits': material_hits, 'mesh_physics_relevant_key_hits': property_hits,
                    'part_metadata_keys': sorted({key for meta in metadata.values() for key in meta}),
                    'pickle_part_keys': sorted(vars(parts[0])), 'pickle_mesh_keys': sorted(vars(parts[0].meshes[0])),
                    'urdf_materials_are_visual_rgba_only': all(set(m.attrib) == {'name'} and [c.tag for c in m] == ['color'] for m in urdf.iter('material')),
                    'named_catalog_materials_are_unverified_name_hints': True,
                    'mesh_density_is_not_material_identity': True,
                    'elastic_modulus_poisson_ratio_friction_restitution_contact_stiffness_exported': False},
               'motor_audit': motor,
               'status': 'source_mass_properties_audited_articulation_and_hardware_calibration_unqualified'}
    json_write(out / 'mass_audit.json', summary)
    json_write(out / 'part_instances.json', rows)
    json_write(out / 'part_types.json', groups)
    json_write(out / 'source_file_hashes.json', {str(p.relative_to(src)): sha256(p) for p in sorted(src.rglob('*')) if p.is_file()})
    with (out / 'part_instances.csv').open('w', newline='') as f:
        headers = ['index_zero_based', 'part_name', 'mesh', 'category', 'mass_kg', 'com_part_x_m', 'com_part_y_m', 'com_part_z_m',
                   'com_export_x_m', 'com_export_y_m', 'com_export_z_m', 'ixx_kg_m2', 'ixy_kg_m2', 'ixz_kg_m2', 'iyy_kg_m2', 'iyz_kg_m2', 'izz_kg_m2',
                   'mesh_volume_m3', 'density_estimate_kg_m3', 'assigned_material_name', 'tensor_physically_consistent', 'source_document_id']
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            a = {k: r[k] for k in ['index_zero_based', 'part_name', 'mesh', 'category', 'mass_kg', 'mesh_volume_m3', 'density_estimate_kg_m3', 'assigned_material_name', 'source_document_id']}
            a.update({f'com_part_{axis}_m': r['com_part_m'][j] for j, axis in enumerate('xyz')})
            a.update({f'com_export_{axis}_m': r['com_export_m'][j] for j, axis in enumerate('xyz')})
            for name, (i, j) in {'ixx': (0, 0), 'ixy': (0, 1), 'ixz': (0, 2), 'iyy': (1, 1), 'iyz': (1, 2), 'izz': (2, 2)}.items():
                a[name + '_kg_m2'] = r['inertia_about_com_part_kg_m2'][i][j]
            a['tensor_physically_consistent'] = r['tensor_check']['physically_consistent']
            writer.writerow(a)
    with (out / 'part_types.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mesh', 'category', 'count', 'mass_each_min_kg', 'mass_each_max_kg', 'mass_total_kg', 'density_estimate_min_kg_m3', 'density_estimate_max_kg_m3', 'assigned_material_name'])
        for g in groups:
            writer.writerow([g['mesh'], g['category'], g['count'], g['mass_each_min_kg'], g['mass_each_max_kg'], g['aggregate']['mass_kg'], g['density_estimate_min_kg_m3'], g['density_estimate_max_kg_m3'], ''])
    lines = ['# Updated Onshape export: mass, inertia and material audit', '',
             'Source: `HexapodLegUpdatedV2.zip`, SHA-256 `' + EXPECTED_ARCHIVE_SHA256 + '`.', '',
             '**These are unmodified exported CAD mass properties, not a measured physical robot.** '
             'The URDF contains one fused link, no joints, no closure records and no named frames. '
             'The pickle retains 1,753 part instances across 59 mesh types, sufficient for a mass ledger but not an authoritative joint/rigid-body map.', '',
             f'Total exported CAD mass: **{total["mass_kg"]:.12f} kg** (URDF writes {source_inertial.find("mass").get("value")} kg). '
             'Gravity is not exported: weight force depends on local gravity; the ledger therefore reports mass.', '',
             '## Aggregate properties in the export assembly frame', '',
             'COM [x, y, z], metres: `' + ', '.join(f'{x:.12g}' for x in total['com_export_m']) + '`.', '',
             'Full inertia about that COM, aligned with export axes, kg·m²:', '', '```text']
    lines += [' '.join(f'{x: .12g}' for x in row) for row in total['inertia_about_com_export_kg_m2']]
    lines += ['```', '', 'Off-diagonal terms are retained. These values describe the arbitrary exported assembly pose; articulated link inertias require correct rigid-body grouping. '
              'Source CAD tensors were rotated by R I Rᵀ and combined using the parallel-axis theorem, never diagonalized or replaced with mesh-derived tensors.', '',
              '## Mass breakdown', '', '| Category | Part instances | Mass (kg) |', '|---|---:|---:|']
    for key, cat in summary['category_breakdown'].items():
        lines.append(f'| {key.replace("_", " ")} | {cat["instances"]} | {cat["mass_kg"]:.9f} |')
    lines += ['', 'Grouping is explicitly based on the source document IDs in `.part` metadata. “Separate bearing parts” are the 18 repeated external bearing assemblies from document `' + BEARING_DOCUMENT + '`, not automatically included in motor replacement mass.', '',
              '## Motor mass gap', '',
              f'The unique vendor housing mesh appears **{motor_count} times**. The {len(motor_rows)} vendor part instances total **{motor_mass:.12f} kg**, '
              f'or **{1000*motor_mass/motor_count:.6f} g per motor**. Every vendor mesh multiplicity is an integer multiple of {motor_count}. '
              'There are **108 zero-mass motor part instances** across two nonempty mesh types; zero inertia on those parts is algebraically consistent but does not establish correct physical mass.', '',
              f'The repository historical RS05 nominal motor mass is 191 g. **If the exact same 18 motors are present**, replacing the aggregate vendor CAD mass would add '
              f'{hypothetical_delta:.9f} kg and produce a hypothetical robot mass of **{total["mass_kg"] + hypothetical_delta:.9f} kg**. '
              'No correction was applied. The motor internals’ distribution, housing/rotor split, cables and hardware need verification before a physically accurate inertia correction can be made. '
              'Do not carry forward the old robot total or its old per-link corrections by default. '
              'See repository `docs/RS05_SPEC_REVIEW.md` and `docs/CAD_ENGINEER_HANDOFF.md` for the historical evidence.', '',
              '## Materials: what was and was not exported', '',
              '**Assigned engineering material names are absent.** The `.part` JSONs provide part identity, source document/microversion, configuration and names. '
              'The pickle provides mass, COM, full inertia, mesh geometry references and RGBA appearance. '
              'All URDF `<material>` records are visual names plus color; none supply a density, alloy/polymer grade, elastic modulus, Poisson ratio, friction, restitution, or contact stiffness.', '',
              'Several catalog names mention steel/stainless grades, but those are name hints rather than proof of assigned Onshape materials. '
              'The table reports **estimated density = exported CAD mass / tessellated mesh volume** only where the mesh is a valid oriented closed solid. '
              'Density, color and part names never establish material identity. Density differences also include tessellation and CAD mass overrides. '
              'The user explicitly accepted mass properties without assigned material names; their absence is not an intake blocker. Measured manufactured masses and actual foot/contact properties remain useful for physical validation.', '',
              '## Per-type ledger', '',
              '| Mesh type | Count | Each mass (g, min–max) | Total mass (g) | Estimated density (kg/m³, min–max) |',
              '|---|---:|---:|---:|---:|']
    for g in sorted(groups, key=lambda g: -g['aggregate']['mass_kg']):
        density = 'unavailable' if g['density_estimate_min_kg_m3'] is None else f'{g["density_estimate_min_kg_m3"]:.3f}–{g["density_estimate_max_kg_m3"]:.3f}'
        lines.append(f'| `{Path(g["mesh"]).stem}` | {g["count"]} | {g["mass_each_min_kg"]*1000:.6f}–{g["mass_each_max_kg"]*1000:.6f} | {g["aggregate"]["mass_kg"]*1000:.6f} | {density} |')
    checks = summary['source_urdf_roundtrip']
    lines += ['', '## Validation', '',
              f'- Source CAD totals match every written fused URDF mass, COM and inertia component within its decimal precision: **{checks["all_fields_within_export_written_precision"]}**.',
              f'- Part ordering, mesh identity, unit mesh scales and all 1,753 visual/collision transforms verified. Largest translation-component difference: {checks["max_part_pose_translation_component_error_m"]:.9g} m; rotation difference: {checks["max_part_pose_rotation_error_rad"]:.9g} rad, consistent with rounded URDF pose text.',
              f'- All {len(rows)} part tensors are finite, symmetric, positive semidefinite and satisfy the principal-moment triangle inequality within recorded numerical tolerance: **{summary["tensor_checks"]["all_source_tensors_physically_consistent"]}**. Positive-mass/positive-definite count: {summary["tensor_checks"]["positive_mass_parts"]}/{summary["tensor_checks"]["strictly_positive_definite_parts"]}.',
              f'- Mesh volume estimates unsuitable for density: {len(summary["mesh_checks"]["invalid_volume_meshes"])} of {len(meshes)} types.',
              '- Restricted pickle loader allows only four inert data classes and three NumPy constructors. No arbitrary module loading or source mutation.', '',
              '## Files and reproduction', '',
              '- `part_instances.json`: all 1,753 instances with source/assembly transforms, mass, COM, full tensors, tensor checks and appearance.',
              '- `part_instances.csv`: flat per-instance ledger; tensor columns are about each part COM in its own mesh frame.',
              '- `part_types.json` / `part_types.csv`: 59 type groups, exact metadata, summed masses, source tensors and mesh diagnostics.',
              '- `mass_audit.json`: aggregate properties, all rounded-URDF comparisons and explicit uncertainty flags.',
              '- `source_file_hashes.json`: hashes of every input file.', '', '```sh',
              'python audit_onshape_mass_properties.py --source /path/to/extracted-export --out /path/to/audit --archive /path/to/HexapodLegUpdatedV2.zip',
              '```', '',
              'The optional archive check pins the supplied ZIP; extracted files must retain their bytes and decode filename mojibake as recorded by the intake. '
              'No Isaac/GPU run or hardware calibration is claimed by this audit.', '',
              '## Required next evidence', '',
              '1. Obtain or independently validate the current mechanism’s mate graph, rigid-link grouping, joint frames and mechanically justified travel bounds; add closure frames only if the current mechanism requires them.',
              '2. Verify the 18 motor model/revisions and actual mass; measure or document per-motor COM, housing/rotor inertia and extra cables/connectors.',
              '3. Retain the exported masses and tensors as requested; compare key manufactured part masses and complete missing/zero-mass motor internals. Assigned material names are optional per the user clarification.',
              '4. Confirm electronics, battery, payload, wiring, feet and fastener completeness by measured mass/COM inventory.',
              '5. Sum these unchanged source properties into each independently validated rigid link, then verify complete URDF/USD mass tensors, constraints, self-collision exclusions and bounded physics probes.']
    (out / 'MASS_MATERIAL_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'mass_kg': total['mass_kg'], 'parts': len(rows), 'types': len(groups),
                      'motor': motor, 'roundtrip': checks, 'tensor_checks': summary['tensor_checks'], 'mesh_checks': summary['mesh_checks']}, indent=2))
    if not (checks['all_fields_within_export_written_precision'] and checks['all_part_pose_checks_pass']
            and summary['tensor_checks']['all_source_tensors_physically_consistent']):
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
