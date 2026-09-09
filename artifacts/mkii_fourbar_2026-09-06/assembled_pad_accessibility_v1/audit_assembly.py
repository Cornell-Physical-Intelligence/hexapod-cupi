"""Read-only LF tibia occupancy of the frozen convex pad's sampled overfill."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CANDIDATE = OUT.parent/'pad_collision_candidate_v1'
sys.path.insert(0, str(CANDIDATE))
from build_candidate import read_stl, origin, closest_on_triangles

BOX_TOLERANCE_M = 1e-8
WINDING_TOLERANCE = 1e-6
PAD_DISTANCE_MIN_M = 1e-4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def winding(points, triangles):
    values = []
    for start in range(0, len(points), 16):
        offset = triangles[None, :, :, :]-points[start:start+16, None, None, :]
        a, b, c = np.moveaxis(offset, 2, 0)
        na, nb, nc = [np.linalg.norm(p, axis=2) for p in (a, b, c)]
        determinant = np.einsum('nfi,nfi->nf', a, np.cross(b, c))
        denominator = na*nb*nc + np.einsum('nfi,nfi->nf', a, b)*nc + np.einsum('nfi,nfi->nf', b, c)*na + np.einsum('nfi,nfi->nf', c, a)*nb
        values.extend(np.sum(2*np.arctan2(determinant, denominator), axis=1)/(4*np.pi))
    return np.asarray(values)


def mesh_topology(triangles):
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    ids = inverse.reshape(-1, 3)
    directed = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    _, edge_ids, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(edge_ids, weights=signs)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    return {'unique_vertices': len(vertices), 'triangles': len(triangles),
            'boundary_edges': int(np.sum(counts == 1)), 'nonmanifold_edges': int(np.sum(counts != 2)),
            'inconsistent_winding_edges': int(np.sum(balance != 0)), 'degenerate_triangles': int(np.sum(area <= 0)),
            'closed_consistently_wound_manifold': bool(np.all(counts == 2) and np.all(balance == 0) and np.all(area > 0)),
            'signed_volume_m3': float(np.einsum('ij,ij->', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))/6)}


def sample_candidate(triangles, divisions=8):
    samples = []
    for i in range(divisions+1):
        for j in range(divisions+1-i):
            samples.append(triangles[:, 0] + i/divisions*(triangles[:, 1]-triangles[:, 0])
                           + j/divisions*(triangles[:, 2]-triangles[:, 0]))
    return np.unique(np.concatenate(samples), axis=0)


def main():
    previous = json.loads((CANDIDATE/'geometry_report.json').read_text())
    for relative, record in previous['sources'].items():
        assert sha(ROOT/relative) == record['sha256']
    candidate_path = CANDIDATE/previous['candidate']['path']
    assert sha(candidate_path) == previous['candidate']['sha256']
    pad_path = ROOT/'robot/hexapod_mkii_assy/meshes/silicone_foot.stl'
    pad_triangles = read_stl(pad_path)
    pad_topology = mesh_topology(pad_triangles)
    assert pad_topology['closed_consistently_wound_manifold']
    samples = sample_candidate(read_stl(candidate_path))
    assert len(samples) == previous['concavity_fill']['sample_points']
    distances = []
    for start in range(0, len(samples), 16):
        d, _ = closest_on_triangles(samples[start:start+16], pad_triangles)
        distances.extend(d)
    distances = np.asarray(distances)
    relevant = np.flatnonzero(distances > PAD_DISTANCE_MIN_M)
    pad_winding = np.full(len(samples), np.nan)
    pad_winding[relevant] = winding(samples[relevant], pad_triangles)
    outside_pad = relevant[np.abs(pad_winding[relevant]) < WINDING_TOLERANCE]
    inside_pad = relevant[(np.abs(pad_winding[relevant]-np.rint(pad_winding[relevant])) < WINDING_TOLERANCE)
                          & (np.abs(np.rint(pad_winding[relevant])) >= 1)]
    ambiguous_pad = np.setdiff1d(relevant, np.r_[outside_pad, inside_pad])
    points = samples[outside_pad]
    urdf_path = ROOT/'robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf'
    link = ET.parse(urdf_path).getroot().find("link[@name='lf_tibia']")
    visuals = link.findall('visual')
    pad_visual = next(v for v in visuals if v.find('geometry/mesh').get('filename').endswith('silicone_foot.stl'))
    pad_transform = origin(pad_visual)
    points_in_tibia = points @ pad_transform[:3, :3].T + pad_transform[:3, 3]
    meshes = {}
    records = []
    containing_instances = [[] for _ in points]
    uncertain_instances = [[] for _ in points]
    box_hits = np.zeros(len(points), dtype=int)
    nearest_box_bound = np.full(len(points), np.inf)
    for index, visual in enumerate(visuals):
        mesh = visual.find('geometry/mesh')
        name = mesh.get('filename').split('/')[-1]
        if name == 'silicone_foot.stl':
            continue
        path = pad_path.parent/name
        if name not in meshes:
            tri = read_stl(path)
            verts = tri.reshape(-1, 3)
            meshes[name] = {'triangles': tri, 'bounds': np.array([verts.min(0), verts.max(0)]),
                            'topology': mesh_topology(tri), 'sha256': sha(path), 'bytes': path.stat().st_size}
        data = meshes[name]
        scale = np.array([float(v) for v in mesh.get('scale', '1 1 1').split()])
        if not np.array_equal(scale, [1, 1, 1]):
            raise ValueError('This audited assembly uses unit-scale meter meshes; review any changed scale explicitly')
        transform = origin(visual)
        inverse = np.linalg.inv(transform)
        local = points_in_tibia @ inverse[:3, :3].T + inverse[:3, 3]
        bounds = data['bounds']
        box_distance = np.linalg.norm(np.maximum(np.maximum(bounds[0]-local, local-bounds[1]), 0), axis=1)
        nearest_box_bound = np.minimum(nearest_box_bound, box_distance)
        possible = np.flatnonzero(np.all(local >= bounds[0]-BOX_TOLERANCE_M, axis=1) & np.all(local <= bounds[1]+BOX_TOLERANCE_M, axis=1))
        box_hits[possible] += 1
        inside_count = uncertain_count = 0
        if data['topology']['closed_consistently_wound_manifold']:
            w = winding(local[possible], data['triangles'])
            for pi, value in zip(possible, w):
                near_integer = np.isfinite(value) and abs(value-round(value)) < WINDING_TOLERANCE
                if near_integer and abs(round(value)) >= 1:
                    containing_instances[pi].append(index); inside_count += 1
                elif not (near_integer and round(value) == 0):
                    uncertain_instances[pi].append(index); uncertain_count += 1
        else:
            for pi in possible:
                uncertain_instances[pi].append(index); uncertain_count += 1
        records.append({'visual_index': index, 'mesh': name, 'source_visual_origin': visual.find('origin').attrib,
                        'mesh_scale': scale.tolist(), 'link_from_mesh': transform.tolist(), 'source_mesh_sha256': data['sha256'],
                        'sampled_overfill_points_in_mesh_aabb': len(possible), 'confirmed_occupied_sample_points': inside_count,
                        'uncertain_sample_points': uncertain_count})
    labels = np.array(['occupied' if occupied else 'uncertain' if uncertain else 'empty'
                       for occupied, uncertain in zip(containing_instances, uncertain_instances)])
    expected_witness = np.array(previous['concavity_fill']['witness_candidate_mesh_point_m'])
    wi = int(np.argmin(np.linalg.norm(points-expected_witness, axis=1)))
    assert np.linalg.norm(points[wi]-expected_witness) < 1e-12
    ranked = np.argsort(-distances[outside_pad])
    def row(i):
        return {'sample_index': int(outside_pad[i]), 'pad_mesh_point_m': points[i].tolist(), 'lf_tibia_point_m': points_in_tibia[i].tolist(),
                'distance_to_silicone_surface_m': float(distances[outside_pad[i]]), 'silicone_winding_number': float(pad_winding[outside_pad[i]]),
                'classification': labels[i], 'containing_visual_indices': containing_instances[i], 'uncertain_visual_indices': uncertain_instances[i],
                'non_silicone_aabb_hits': int(box_hits[i]), 'distance_lower_bound_to_other_cad_solids_m': float(nearest_box_bound[i])}
    report = {'schema': 'hexapod.assembled_pad_accessibility.v1', 'scope': 'LF tibia rigid assembly CAD occupancy only; no simulator, hardware measurement, approach-path or terrain accessibility proof',
              'runtime_assets_modified': False, 'published_candidate_modified': False,
              'sources': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in [urdf_path, candidate_path, CANDIDATE/'geometry_report.json', CANDIDATE/'build_candidate.py', pad_path]},
              'tolerances': {'expanded_mesh_aabb_m': BOX_TOLERANCE_M, 'winding_near_integer': WINDING_TOLERANCE,
                             'minimum_unsigned_pad_surface_distance_for_classification_m': PAD_DISTANCE_MIN_M},
              'sample_selection': {'candidate_surface_samples': len(samples), 'farther_than_0_1mm_from_silicone_surface': len(relevant),
                                   'outside_silicone_solid': len(outside_pad), 'inside_silicone_material': len(inside_pad),
                                   'ambiguous_silicone_winding': len(ambiguous_pad)},
              'assembly': {'link': 'lf_tibia', 'total_visual_instances': len(visuals), 'non_silicone_visual_instances': len(records),
                           'unique_non_silicone_meshes': len(meshes), 'instances': records,
                           'meshes': {name: {'sha256': m['sha256'], 'bytes': m['bytes'], 'bounds_mesh_m': m['bounds'].tolist(), 'topology': m['topology']} for name, m in meshes.items()}},
              'classification_counts': {name: int(np.sum(labels == name)) for name in ('empty', 'occupied', 'uncertain')},
              'outside_every_non_silicone_mesh_aabb': int(np.sum(box_hits == 0)),
              'prior_worst_overfill_witness': row(wi), 'largest_20_pad_overfill_samples': [row(i) for i in ranked[:20]],
              'interpretation': ['Empty means outside every other solid in this LF tibia assembly under the recorded AABB/winding checks; it does not establish an obstacle approach path.',
                                 'The prior worst witness is outside all 45 non-silicone visual AABBs, so its empty classification does not depend on watertightness or winding.',
                                 'Occupancy is sampled; do not interpret counts as area, volume, collision probability or a full assembly-wide bound.',
                                 'Moving neighboring links, wires absent from CAD and other legs were not included. No assumption that they permanently fill the pad is made.',
                                 'A closed consistently wound mesh can still self-intersect; edge checks alone do not establish an embedded manifold. Zero winding is used only with this caveat; the worst AABB witness is independent of it.']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (OUT/'sampled_overfill_classification.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'pad_x_m', 'pad_y_m', 'pad_z_m', 'tibia_x_m', 'tibia_y_m', 'tibia_z_m', 'distance_to_silicone_surface_m', 'silicone_winding', 'classification', 'occupied_visual_indices', 'uncertain_visual_indices', 'aabb_hits', 'distance_lower_bound_to_other_cad_solids_m'])
        for i in range(len(points)):
            r = row(i)
            writer.writerow([r['sample_index'], *r['pad_mesh_point_m'], *r['lf_tibia_point_m'], r['distance_to_silicone_surface_m'], r['silicone_winding_number'], r['classification'], json.dumps(r['containing_visual_indices']), json.dumps(r['uncertain_visual_indices']), r['non_silicone_aabb_hits'], r['distance_lower_bound_to_other_cad_solids_m']])
    print(json.dumps({'sample_selection': report['sample_selection'], 'classification_counts': report['classification_counts'],
                      'outside_every_aabb': report['outside_every_non_silicone_mesh_aabb'], 'worst': report['prior_worst_overfill_witness'],
                      'nonwatertight_meshes': [name for name, m in meshes.items() if not m['topology']['closed_consistently_wound_manifold']]}, indent=2))


if __name__ == '__main__':
    main()
