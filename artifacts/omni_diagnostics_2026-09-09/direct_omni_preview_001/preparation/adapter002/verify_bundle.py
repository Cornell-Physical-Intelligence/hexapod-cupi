"""Read-only portable payload verifier; does not import simulator dependencies."""
from pathlib import Path
import hashlib,json

def main():
    root=Path(__file__).resolve().parent
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=root/'FREEZE_SHA256.json';expected=json.loads(manifest.read_text())
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink payload')
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=expected:raise ValueError('Missing, changed or unlisted payload')
    print(json.dumps({'passed':True,'payloads':len(actual),'freeze_sha256':sha(manifest)}))

if __name__=='__main__':main()
