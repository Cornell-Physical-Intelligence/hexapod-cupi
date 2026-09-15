"""Portable exact preparation verification; no imports of stored runtime."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((root/'BUNDLE_SHA256.json').read_text())
actual = {str(path.relative_to(root)): sha(path) for path in root.rglob('*')
          if path.is_file() and path.name != 'BUNDLE_SHA256.json' and '__pycache__' not in path.parts}
assert actual == manifest, 'Preparation payload changed or omitted'
readiness = json.loads((root/'READINESS.json').read_text())
for name in ('source','host'):
    assert sha(root/name/'FREEZE_SHA256.json') == readiness[name+'_freeze_sha256']
    entries = json.loads((root/name/'FREEZE_SHA256.json').read_text())
    for path, expected in entries.items(): assert sha(root/name/path) == expected, path
for folder in ('source005','host007','guard005'):
    entries=json.loads((root/'parents'/folder/'FREEZE_SHA256.json').read_text())
    for path, expected in entries.items(): assert sha(root/'parents'/folder/path)==expected,path
assert all(readiness[name] is False for name in ('native_executed','standing_admission',
                                               'batch_admission','training_allowed'))
print(json.dumps({'verified':True,'payloads':len(manifest),
                  'source_freeze_sha256':readiness['source_freeze_sha256'],
                  'host_freeze_sha256':readiness['host_freeze_sha256'],
                  'native_execution':False},indent=2))
