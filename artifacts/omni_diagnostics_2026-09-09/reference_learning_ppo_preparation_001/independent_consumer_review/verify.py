from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'FREEZE_SHA256.json').read_text())
files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}
assert files==set(m['files'])
for k,v in m['files'].items():assert sha(root/k)==v,k
if len(sys.argv)>1:
 owner=Path(sys.argv[1]);r=json.loads((root/'review.json').read_text())
 assert sha(owner/'FREEZE_SHA256.json')==r['owner_freeze_sha256']
 manifest=json.loads((owner/'FREEZE_SHA256.json').read_text())
 assert manifest==r['owner_payloads']
 for k,v in manifest.items():assert sha(owner/k)==v,k
 print('Verified all',len(manifest),'frozen owner payloads')
print('Verified',len(files),'independent review payloads')
