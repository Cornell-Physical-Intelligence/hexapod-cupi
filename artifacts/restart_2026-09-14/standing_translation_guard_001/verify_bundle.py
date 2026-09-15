"""Offline byte verification only; no host calls or artifact-source imports."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
files = list(root.rglob('*'))
if any(path.is_symlink() for path in files):
    raise SystemExit('Symbolic bundle entry')
actual = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in files if path.is_file() and path.name != 'FREEZE_SHA256.json'}
expected = json.loads((root / 'FREEZE_SHA256.json').read_text())
if actual != expected:
    raise SystemExit('Changed or unlisted diagnostic guard input')
print(json.dumps({'verified': True, 'files': len(actual), 'native_calls': 0,
                  'root_bindings_still_required': True}, indent=2))
