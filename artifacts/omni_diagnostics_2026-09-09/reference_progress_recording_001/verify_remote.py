from pathlib import Path
import hashlib,json,subprocess,time
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
source=base/'reference_physics_source_009';run=base/'reference_progress_recording_001';pause=base/'forecast_pause_036';qualified=base/'reference_physics_009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,manifest):
    m=json.loads((root/manifest).read_text());m=m.get('files',m)
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/manifest}
    assert actual==m,root
    return dict(files=len(m),manifest_sha256=sha(root/manifest))
source_check=verify(source,'campaign_source_hashes.json')
adapter_check=verify(base/'reference_progress_recording_adapter_001','FREEZE_SHA256.json')
host_check=verify(base/'reference_progress_recording_launch_001','FREEZE_SHA256.json')
guard_check=verify(base/'reference_progress_recording_guard_001','FREEZE_SHA256.json')
assets=json.loads((qualified/'inputs/study_before.sha256.json').read_text())
assert {str(p.relative_to(qualified/'inputs/study')):sha(p) for p in (qualified/'inputs/study').rglob('*') if p.is_file()}==assets
raw={}
for prefix,root in [('run',run),('forecast_pause',pause)]:
    for p in sorted(root.rglob('*')):
        if p.is_file():raw[prefix+'/'+str(p.relative_to(root))]=sha(p)
absence={}
for p in (run/'jobs').glob('*.json'):
    j=json.loads(p.read_text())
    for token in (j.get('container_id'),j.get('container_name')):
        if not token:continue
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
        assert q.returncode and ('no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower()),(token,q.stdout,q.stderr)
        absence[token]={'returncode':q.returncode,'stderr':q.stderr.strip()}
restored=json.loads((pause/'restored.json').read_text())
unit=subprocess.check_output(['systemctl','--user','show','hexapod-reference-progress-recording-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result'],text=True)
assert 'ActiveState=active' not in unit and 'ActiveState=activating' not in unit
print(json.dumps(dict(verified_unix=time.time(),source=source_check,adapter=adapter_check,host=host_check,guard=guard_check,admitted_asset_files=len(assets),admitted_assets_unchanged=True,qualified_campaign_sha256=sha(qualified/'campaign.json'),raw_payloads=raw,owned_containers_absent=absence,pause_restoration=restored,unit=unit),indent=2))
