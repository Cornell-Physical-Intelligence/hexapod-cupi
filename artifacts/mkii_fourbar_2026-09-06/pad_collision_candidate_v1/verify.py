"""Verify source bytes and convex-distance optimality without relying on fit targets."""
import json
import numpy as np
from scipy.spatial import ConvexHull
from build_candidate import ROOT, OUT, MESH, sha, read_stl, closest_on_convex

report = json.loads((OUT/'geometry_report.json').read_text())
for relative, record in report['sources'].items():
    path = ROOT/relative
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
candidate_path = OUT/report['candidate']['path']
assert sha(candidate_path) == report['candidate']['sha256']
vertices = np.unique(read_stl(MESH).reshape(-1, 3), axis=0)
candidate_vertices = np.unique(read_stl(candidate_path).reshape(-1, 3), axis=0)
hull = ConvexHull(candidate_vertices)
distance, nearest = closest_on_convex(vertices, hull)
# Projection theorem: q in C and (p-q)·(x-q) <= 0 for every extreme x of C.
feasibility = float(np.max(nearest @ hull.equations[:, :3].T+hull.equations[:, 3]))
normal_inequality = np.einsum('nvi,ni->nv', candidate_vertices[None, :, :]-nearest[:, None, :], vertices-nearest)
optimality = float(normal_inequality.max())
assert feasibility <= 1e-11, feasibility
assert optimality <= 2e-15, optimality
assert abs(float(distance.max())-report['candidate']['maximum_cad_surface_to_candidate_distance_m']) <= 1e-12
assert len(candidate_vertices) == 64
assert all(any(np.array_equal(p, q) for q in vertices) for p in candidate_vertices)
samples = np.load(OUT/'support_samples.npz', allow_pickle=False)
deficit = samples['cad_support_m']-samples['candidate_support_m']
assert float(deficit.max()) <= float(distance.max())+1e-12
assert float(deficit.min()) >= -1e-12
result = {'pass': True, 'verified_source_files': len(report['sources']),
          'cad_vertices_with_convex_projection_certificates': len(vertices),
          'maximum_projection_feasibility_error_m': feasibility,
          'maximum_projection_optimality_inequality_m2': optimality,
          'maximum_outward_undercoverage_m': float(distance.max()),
          'candidate_sha256': sha(candidate_path),
          'scope': 'CPU geometry/serialization only; no simulator contact, cooking, material or hardware qualification'}
(OUT/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2))
