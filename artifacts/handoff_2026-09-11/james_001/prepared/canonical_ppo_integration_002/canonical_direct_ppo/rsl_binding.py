"""Installed RSL source identity, reused algorithm infrastructure only."""
from pathlib import Path
import hashlib,json

def verify():
 import rsl_rl
 root=Path(rsl_rl.__file__).resolve().parent
 expected=json.loads((Path(__file__).resolve().parent.parent/'RSL_SOURCE_SHA256.json').read_text())
 actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*.py')}
 if actual!=expected:raise ValueError('Installed RSL source differs from independently exercised algorithm')
 return {'source_files':len(expected),'map_sha256':hashlib.sha256((Path(__file__).resolve().parent.parent/'RSL_SOURCE_SHA256.json').read_bytes()).hexdigest()}
