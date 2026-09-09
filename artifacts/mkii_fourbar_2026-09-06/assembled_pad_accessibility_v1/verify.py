"""Recheck source bytes and worst-witness emptiness independently of winding."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from audit_assembly import ROOT, OUT, CANDIDATE, sha
from build_candidate import read_stl, origin

report = json.loads((OUT/'report.json').read_text())
for relative, expected in report['sources'].items():
    path = ROOT/relative
    assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
mesh_dir = ROOT/'robot/hexapod_mkii_assy/meshes'
for name, expected in report['assembly']['meshes'].items():
    assert sha(mesh_dir/name) == expected['sha256']
for line in (CANDIDATE/'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert sha(CANDIDATE/name) == expected
link = ET.parse(ROOT/'robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf').getroot().find("link[@name='lf_tibia']")
pad = next(v for v in link.findall('visual') if v.find('geometry/mesh').get('filename').endswith('silicone_foot.stl'))
point = np.array(report['prior_worst_overfill_witness']['pad_mesh_point_m'])
t = origin(pad)
point_in_link = t[:3, :3] @ point+t[:3, 3]
bounds = []
count = 0
for visual in link.findall('visual'):
    name = visual.find('geometry/mesh').get('filename').split('/')[-1]
    if name == 'silicone_foot.stl':
        continue
    t = origin(visual)
    np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-12)
    local_point = t[:3, :3].T @ (point_in_link-t[:3, 3])
    vertices = read_stl(mesh_dir/name).reshape(-1, 3)
    distance_lower_bound = np.linalg.norm(np.maximum(np.maximum(vertices.min(0)-local_point, local_point-vertices.max(0)), 0))
    assert distance_lower_bound > 1e-8
    bounds.append(distance_lower_bound); count += 1
assert count == 45
assert abs(min(bounds)-report['prior_worst_overfill_witness']['distance_lower_bound_to_other_cad_solids_m']) < 1e-12
with (OUT/'sampled_overfill_classification.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == report['sample_selection']['outside_silicone_solid']
for label, expected in report['classification_counts'].items():
    assert sum(r['classification'] == label for r in rows) == expected
result = {'pass': True, 'source_files_rehashed': len(report['sources'])+len(report['assembly']['meshes']),
          'frozen_candidate_manifest_unchanged': True,
          'non_silicone_visual_bounds_excluding_worst_witness': count,
          'witness_distance_lower_bound_to_all_other_lf_tibia_cad_solids_m': float(min(bounds)),
          'csv_classification_rows': len(rows), 'worst_witness_conclusion_requires_winding': False}
(OUT/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2))
