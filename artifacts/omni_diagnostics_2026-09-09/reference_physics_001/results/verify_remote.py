from pathlib import Path
import json,hashlib,subprocess,time
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
source=base/'reference_physics_source_001';run=base/'reference_physics_001';pause=base/'forecast_pause_027'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((source/'campaign_source_hashes.json').read_text())
for f,h in m.items():assert sha(source/f)==h,f
actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
assert actual==set(m)|{'campaign_source_hashes.json'}
assets=json.loads((run/'inputs/study_before.sha256.json').read_text())
for f,h in assets.items():assert sha(run/'inputs/study'/f)==h,f
assert {str(p.relative_to(run/'inputs/study')) for p in (run/'inputs/study').rglob('*') if p.is_file()}==set(assets)
raw={}
for prefix,root in [('run',run),('forecast_pause',pause)]:
 for p in sorted(root.rglob('*')):
  if p.is_file() and not (prefix=='run' and 'study' in p.relative_to(root).parts):raw[prefix+'/'+str(p.relative_to(root))]=sha(p)
absence={}
for p in (run/'jobs').glob('*.json'):
 j=json.loads(p.read_text())
 for token in (j.get('container_id'),j.get('container_name')):
  if not token:continue
  q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert q.returncode and ('no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower()),(token,q.stdout,q.stderr)
  absence[token]={'returncode':q.returncode,'stderr':q.stderr.strip()}
restored=json.loads((pause/'restored.json').read_text())
unit=subprocess.check_output(['systemctl','--user','show','hexapod-reference-physics-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result'],text=True)
assert 'ActiveState=active' not in unit and 'ActiveState=activating' not in unit
print(json.dumps(dict(verified_unix=time.time(),source_files=len(m),source_unchanged=True,source_manifest_sha256=sha(source/'campaign_source_hashes.json'),admitted_asset_files=len(assets),admitted_assets_unchanged=True,raw_payloads=raw,owned_containers_absent=absence,pause_restoration=restored,unit=unit),indent=2))
