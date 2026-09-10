"""Read-only, portable exact-byte verifier. Does not import launch code."""
from pathlib import Path
import hashlib,json

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 root=Path(__file__).resolve().parent;manifest=root/'FREEZE_SHA256.json'
 if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic bundle path')
 expected=json.loads(manifest.read_text())
 for key in expected:
  p=Path(key)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe relative path')
 actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
 if actual!=expected:raise ValueError('Changed, missing or unlisted payload')
 print(json.dumps({'passed':True,'payloads':len(actual),'freeze_sha256':sha(manifest),'scope':'Local prepared bytes only; no remote or terminal admission'}))
if __name__=='__main__':main()
