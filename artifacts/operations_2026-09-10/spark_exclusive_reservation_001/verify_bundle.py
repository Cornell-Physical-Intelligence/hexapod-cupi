from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'BUNDLE_SHA256.json').read_text())
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
assert actual==set(manifest)|{'BUNDLE_SHA256.json'}
for rel,expected in manifest.items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
r=json.loads((root/'verification_loaded_001.json').read_text())
assert r['verified'] and len(r['units'])==11 and not r['cuda_processes_raw'].strip()
assert not r['manual_launch_hardware_block_claimed'] and not r['mutation_performed']
for unit,state in r['units'].items():
    assert state['ActiveState'] in ('inactive','failed')
    assert state['NeedDaemonReload']=='no'
    assert '90-hexapod-exclusive-reservation-20260910.conf' in state['DropInPaths']
for path,info in r['files'].items():
    p=root/'remote_files'/path.removeprefix('/home/orionh/')
    assert p.read_text()==info['text'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
assert hashlib.sha256((root/'coordination_after.md').read_bytes()).hexdigest()==r['coordination_sha256']
print(json.dumps({'passed':True,'payloads':len(manifest),'deferred_units':11,'no_remote_actions':True}))
