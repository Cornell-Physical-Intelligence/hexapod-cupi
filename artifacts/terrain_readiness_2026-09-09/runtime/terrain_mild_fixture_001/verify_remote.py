from pathlib import Path
import hashlib,json,subprocess,sys,time
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');source=base/'terrain_mild_fixture_source_001';run=base/'terrain_mild_fixture_smoke_001';pause=base/'forecast_pause_045'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected='245c20e9318229d59e7132a6bc30b1fd1179460ea2700a2192c1647721a59a91'
assert sha(source/'campaign_source_hashes.json')==expected
sys.path.insert(0,str(source/'tools'));from launch_mild_fixtures_spark import check_source
from mild_fixture_contract import validate_result,catalog_identity
identity=check_source(source);assert identity['source_files']==105
entries,geometry=catalog_identity(source);assert len(geometry)==97
before=json.loads((run/'inputs/fixture_before.sha256.json').read_text());assert before==geometry
state=json.loads((run/'fixtures/validation.json').read_text());gate=validate_result(state,source)
campaign=json.loads((run/'campaign.json').read_text());assert campaign['status']=='completed' and campaign['gate']==gate
raw={}
for label,path in [('run',run),('forecast_pause',pause)]:
 for p in sorted(path.rglob('*')):
  if p.is_file():raw[label+'/'+str(p.relative_to(path))]=sha(p)
absence={}
for p in (run/'jobs').glob('*.json'):
 data=json.loads(p.read_text())
 for token in (data.get('container_id'),data.get('container_name')):
  if not token:continue
  result=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert result.returncode and ('no such object' in result.stderr.lower() or 'no such container' in result.stderr.lower())
  absence[token]={'returncode':result.returncode,'stderr':result.stderr.strip()}
restored=json.loads((pause/'restored.json').read_text())
unit=subprocess.check_output(['systemctl','--user','show','hexapod-mild-fixtures-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result'],text=True)
assert 'ActiveState=active' not in unit and 'ActiveState=activating' not in unit
print(json.dumps(dict(verified_unix=time.time(),source_manifest_sha256=expected,source_files=105,source_unchanged=True,fixture_geometry_files=97,fixture_geometry_unchanged=True,fixture_gate=gate,raw_payloads=raw,owned_containers_absent=absence,pause_restoration=restored,unit=unit,no_robot_or_terrain_walking_qualification=True),indent=2))
