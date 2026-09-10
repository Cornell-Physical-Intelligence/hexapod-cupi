"""Compact standalone fixture source; no full robot, PPO runtime or original30 clone."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];SOURCE=HERE/'source_mild_fixtures_001_final'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if SOURCE.exists():raise FileExistsError(SOURCE)
files=['validate_terrain_fixtures.py','terrain_fixture_checks.py','terrain_readiness.py','terrain_contact_evidence.py','mild_fixture_contract.py','gpu_ownership_helpers.py','launch_mild_fixtures_spark.py']
for f in files:ast.parse((HERE/f).read_text())
(SOURCE/'tools').mkdir(parents=True)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
catalog=Path('artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/terrain_catalog.json');entries=json.loads((ROOT/catalog).read_text())['fixtures'];target=SOURCE/catalog;target.parent.mkdir(parents=True);shutil.copy2(ROOT/catalog,target)
for e in entries:
 for f,h in [(Path(e['usda']),e['sha256']),(Path(e['usda']).with_suffix('.npz'),e['npz_sha256'])]:
  src=ROOT/catalog.parent/f;dst=target.parent/f
  assert sha(src)==h,(e['id'],str(f));dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
origin=json.loads((HERE/'SOURCE_INPUTS.json').read_text());origin.update(scope='Standalone48fixture smoke, original geometry/ray/contact gates, no robot or actor',
 catalog_sha256=sha(target),fixture_count=len(entries),catalog_geometry_payloads=97,owner_runtime_sha256={f:sha(HERE/f) for f in files})
(SOURCE/'source_origin.json').write_text(json.dumps(origin,indent=2)+'\n')
m={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(m,indent=2)+'\n')
r={'source':str(SOURCE),'source_files':len(m),'source_manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'catalog_geometry_files':97,'fixture_count':48,'source_not_full_robot_or_PPO':True,
 'harness_sha256':sha(SOURCE/'tools/validate_terrain_fixtures.py'),'host_sha256':sha(SOURCE/'tools/launch_mild_fixtures_spark.py')}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
