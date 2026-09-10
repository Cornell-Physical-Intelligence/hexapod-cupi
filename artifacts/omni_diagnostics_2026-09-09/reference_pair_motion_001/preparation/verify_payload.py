"""Portable immutable preparation/source-delta verification; no physical pass."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify_map(root,raw):
 for rel,digest in raw.get('files',raw).items():
  p=(root/rel).resolve()
  assert root.resolve() in p.parents and sha(p)==digest,rel
b=json.loads((H/'BUNDLE_SHA256.json').read_text());verify_map(H,b)
bindings=json.loads((H/'BINDINGS.json').read_text())
for name,digest in bindings['owners'].items():
 p=H/name/'FREEZE_SHA256.json';assert sha(p)==digest;verify_map(H/name,json.loads(p.read_text()))
assert sha(H/'source_hashes.json')==bindings['new_source_manifest_sha256']
source=json.loads((H/'source_hashes.json').read_text());build=json.loads((H/'adapter/SOURCE_BUILD.json').read_text())
assert len(source)==946
for rel,change in build['exact_changed_paths'].items():assert source[rel]==change['after']==sha(H/'source_delta'/rel)
# Runtime subtree carries the original43-file owner identity but only selected
# runtime dependencies. Verify those against the complete upstreamowner here.
for p in (H/'source_delta/tools/paired_runtime').rglob('*'):
 if p.is_file() and p.name!='FREEZE_SHA256.json':
  rel=str(p.relative_to(H/'source_delta/tools/paired_runtime'));assert sha(p)==sha(H/'paired_cpu_owner'/rel),rel
assert not bindings['PPO_admitted'] and bindings['actual_GPU_controls']==0
print(json.dumps({'preparation_payloads_verified':len(b.get('files',b)),'source_payload_map_verified':946,'delta_payloads':len(build['exact_changed_paths']),
 'all_frozen_owner_maps_verified':True,'actual_physics_admitted':False},indent=2))
