"""Portable exact wrapper/nested-bundle/media/terminal verification."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'BUNDLE_SHA256.json';expected=json.loads(manifest.read_text())
actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
assert not any(p.is_symlink() for p in root.rglob('*'));assert actual==expected,'Wrapper inventory differs'
lineage=json.loads((root/'LINEAGE.json').read_text())
for name,bound in lineage['nested_freezes'].items():
 d=root/name;m=d/'FREEZE_SHA256.json';assert sha(m)==bound
 inventory={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file() and p!=m}
 assert inventory==json.loads(m.read_text()),name
subprocess.run([sys.executable,'-B',str(root/'terminal/verify_payload.py')],check=True)
halo=json.loads((root/'later_halo_restore/restored.json').read_text());assert sha(root/'later_halo_restore/restored.json')==(root/'later_halo_restore/remote_sha256.txt').read_text().split()[0]
assert halo['startup']['invocation']=='9091c23f9a954fa786ca0c7009cb2ab6' and halo['startup']['own_weather_lock_verified']
assert halo['restored_unix']==1789059356.8370728
assert halo['restored_unix']>json.loads((root/'terminal/forecast_pause/restored.json').read_text())['restored_unix']
print(json.dumps({'passed':True,'wrapper_payloads':len(actual),'bundle_sha256':sha(manifest),'media_complete':True,'host_failed_finalization_preserved':True,'later_halo_startup_only':True}))
