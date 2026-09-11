#!/usr/bin/env python3
"""Portable verification of this CPU preparation; no simulator or admission implied."""
from pathlib import Path
import hashlib,json

def verify(root):
 root=Path(root).resolve();expected=json.loads((root/'FREEZE_SHA256.json').read_text())
 actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p!=root/'FREEZE_SHA256.json'}
 if actual!=expected:raise ValueError('Preparation payload inventory/hash mismatch')
 if any((root/n).is_symlink()or not(root/n).resolve().is_relative_to(root)for n in expected):raise ValueError('Unsafe preparation path')
 return {'payloads_verified':len(actual),'cpu_preparation_only':True,'native_admission_claimed':False}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).parent),sort_keys=True))
