"""Independent publication checks; run from the repository root."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

asset = Path('robot/hexapod_mkii_updated_v1')
selection = json.loads(Path('robot/active_model.json').read_text())
for key in ('urdf', 'model', 'raw_cad_urdf'):
    assert digest(Path(selection[key]['path'])) == selection[key]['sha256'], key
assert digest(Path(selection['usd'])) == selection['usd_sha256']
assert selection['native_physics_admitted'] is False
raw = json.loads((asset / 'model.json').read_text())
corrected = json.loads((asset / 'model_rs05_mass_corrected.json').read_text())
assert raw['parts'] == corrected['parts']
assert raw['joints'] == corrected['joints']
assert len(raw['parts']) == 1753
assert len(raw['links']) == 19 and len(raw['joints']) == 18
mapping = json.loads((asset / 'mesh_mapping.json').read_text())
assert len(mapping) == 59
for entry in mapping:
    assert digest(asset / 'meshes' / entry['mesh']) == entry['sha256']

expected_yaw = {'lf':74.295, 'lm':46.715, 'lr':74.750,
                'rf':74.750, 'rm':46.715, 'rr':74.295}
variants = []
for path in sorted((asset / 'urdf').glob('*.urdf')):
    root = ET.parse(path).getroot()
    links, joints = root.findall('link'), root.findall('joint')
    assert len(links) == 19 and len(joints) == 18
    assert len({j.find('child').get('link') for j in joints}) == 18
    mass = 0
    for link in links:
        inertial = link.find('inertial')
        mass += float(inertial.find('mass').get('value'))
        v = {k:float(v) for k,v in inertial.find('inertia').attrib.items()}
        tensor = np.array([[v['ixx'],v['ixy'],v['ixz']],
                           [v['ixy'],v['iyy'],v['iyz']],
                           [v['ixz'],v['iyz'],v['izz']]])
        eig = np.linalg.eigvalsh(tensor)
        assert np.all(np.isfinite(eig)) and eig[0] > 0
        assert eig[2] <= eig[0] + eig[1] + 1e-12
    for joint in joints:
        name = joint.get('name'); limit = joint.find('limit')
        assert joint.get('type') == 'revolute'
        assert np.allclose([float(x) for x in joint.find('axis').get('xyz').split()], [0,0,1])
        expected = (-120,80) if 'femur' in name else (-5,180) if 'tibia' in name else (-expected_yaw[name[:2]],expected_yaw[name[:2]])
        actual = tuple(math.degrees(float(limit.get(k))) for k in ('lower','upper'))
        assert np.allclose(actual, expected, atol=1e-10, rtol=0), (name,actual)
    expected_mass = 7.466088235225788 if 'mass_corrected' in path.name else 5.147603654203134
    assert abs(mass-expected_mass) < 1e-11
    variants.append({'path':str(path),'sha256':digest(path),'mass_kg':mass,'SPD_tensors':19,'joint_limits_checked':18})

bundle_root = Path('artifacts/mkii_updated_2026-09-10')
manifest_counts = {}
for name in ('joint_review_002','yaw_envelope_002'):
    directory = bundle_root/name; count = 0
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        sha, relative = line.split('  ',1)
        assert digest(directory/relative) == sha, relative
        count += 1
    manifest_counts[name] = count
directory = bundle_root/'usd_002'
entries = json.loads((directory/'INTEGRATION_MANIFEST.json').read_text())['files']
for entry in entries:
    assert digest(directory/entry['path']) == entry['sha256'], entry['path']
manifest_counts['usd_002'] = len(entries)

print(json.dumps({'passed':True,'source_parts_unchanged':1753,'mesh_hashes_verified':59,
                  'raw_corrected_kinematics_identical':True,'variants':variants,
                  'successor_evidence_files_verified':manifest_counts,
                  'selection_sha256':digest(Path('robot/active_model.json')),
                  'scope':'Offline source, URDF, mass/inertia, bounds and evidence integrity; no native physics claim.'},indent=2))
