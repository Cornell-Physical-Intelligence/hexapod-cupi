"""Freeze a fresh source from verified candidate003 plus reviewed physics-only files."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'tmp/omni_velocity_launch_003/source_003'
DEST = HERE / 'source_001'
RESIDUAL = ROOT / 'tmp/omni_reference_residual_002'
WAVE = ROOT / 'tmp/omni_reference_wave_001'
OWNED = ('physics_telemetry.py', 'reference_physics_env.py', 'run_reference_physics.py',
         'screen_contract.py', 'screen_metrics.py', 'launch_reference_physics_spark.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(directory):
    receipt = json.loads((directory / 'FREEZE_SHA256.json').read_text())
    mapping = receipt.get('files', receipt)
    for name, expected in mapping.items():
        if sha(directory / name) != expected:
            raise ValueError('Owner freeze changed: ' + str(directory / name))
    return sha(directory / 'FREEZE_SHA256.json')


def main():
    if DEST.exists():
        raise FileExistsError('Never replace a frozen source; make a new version')
    if sha(BASE / 'campaign_source_hashes.json') != '00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25':
        raise ValueError('Wrong root-deployed source003 baseline')
    residual_map = frozen(RESIDUAL)
    wave_map = frozen(WAVE)
    manifest = json.loads((BASE / 'campaign_source_hashes.json').read_text())
    for name, expected in manifest.items():
        if sha(BASE / name) != expected:
            raise ValueError('Changed baseline source: '+name)
    DEST.mkdir()
    for name in manifest:
        target = DEST / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    overlay = {}
    def copy(origin, name):
        target = DEST / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target); overlay[name] = sha(target)
    for name in OWNED:
        copy(HERE / name, 'tools/' + name)
    for name in ('reference_residual.py', 'reference_residual_env.py'):
        copy(RESIDUAL / name, 'tools/' + name)
    for name in ('wave_reference.py', 'serial_geometry.py', 'wave_math.py',
                 'geometry/candidate_c_reference.json', 'geometry/f050_t060.urdf'):
        copy(WAVE / name, 'tools/' + name)
    copy(ROOT / 'tmp/terrain_robot_smoke_005_preparation/source/tools/terrain_contact_evidence.py',
         'tools/terrain_contact_evidence.py')
    assert sha(DEST / 'tools/geometry/candidate_c_reference.json') == sha(DEST / 'robot/hexapod_mkii_length_study/candidate_c_reference.json')
    assert sha(DEST / 'tools/geometry/f050_t060.urdf') == sha(DEST / 'robot/hexapod_mkii_length_study/urdf/f050_t060.urdf')
    origin = dict(parent_source_manifest_sha256=sha(BASE / 'campaign_source_hashes.json'),
        parent_source_identity='fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e',
        parent_source_origin_sha256=sha(BASE / 'source_origin.json'),
        residual_owner_freeze_sha256=residual_map, wave_owner_freeze_sha256=wave_map,
        overlays=overlay, scope='32x1000zero-residualstanding then1x2400measuredcontactwave; noPPO/checkpoints/bodyposecontrol/automaticcontinuation',
        stage2_complete=False, terrain_qualified=False, original_asset_and_physical_gates_preserved=True,
        exact_plan_sha256=sha(DEST / 'robot/hexapod_mkii_length_study/training_plan.json'))
    (DEST / 'source_origin.json').write_text(json.dumps(origin, indent=2, sort_keys=True)+'\n')
    result = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST / 'campaign_source_hashes.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'source': str(DEST), 'files': len(result),
                     'manifest_sha256': sha(DEST / 'campaign_source_hashes.json')}, indent=2))


if __name__ == '__main__':
    main()
