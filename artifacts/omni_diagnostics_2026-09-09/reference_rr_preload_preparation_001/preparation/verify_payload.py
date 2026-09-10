"""Portable, read-only complete preparation/overlay verifier. No third-party deps."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,mapping,extra=()):
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 if actual!=set(mapping)|set(extra):raise ValueError('Extra or missing payloads')
 for f,h in mapping.items():
  p=(root/f).resolve()
  if root.resolve() not in p.parents or sha(p)!=h:raise ValueError('Hash/path mismatch: '+f)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-source',type=Path);args=ap.parse_args()
 m=json.loads((ROOT/'BUNDLE_SHA256.json').read_text());verify(ROOT,m,['BUNDLE_SHA256.json'])
 refs=json.loads((ROOT/'REFERENCED_INPUTS.json').read_text())
 for sub,key in [('owner','new_owner_freeze_sha256'),('guard','guard_freeze_sha256')]:
  p=ROOT/sub/'FREEZE_SHA256.json';assert sha(p)==refs[key];verify(ROOT/sub,json.loads(p.read_text()),['FREEZE_SHA256.json'])
 source=json.loads((ROOT/'source_overlay/campaign_source_hashes.json').read_text());assert len(source)==931
 assert sha(ROOT/'source_overlay/campaign_source_hashes.json')==refs['new_source_manifest_sha256']
 for f in ['wave_reference.py','rr_preload_diagnostic.py','directional_contract.py','launch_directional_physics_spark.py','solver_comparison.py']:
  assert sha(ROOT/'owner'/f)==source['tools/'+f]
 assert sha(ROOT/'source_overlay/source_origin.json')==source['source_origin.json']
 if args.parent_source:
  p=args.parent_source.resolve();assert sha(p/'campaign_source_hashes.json')==refs['source_parent']['sha256'];old=json.loads((p/'campaign_source_hashes.json').read_text());verify(p,old,['campaign_source_hashes.json'])
  expected=dict(old)
  for f in ['wave_reference.py','rr_preload_diagnostic.py','directional_contract.py','launch_directional_physics_spark.py','solver_comparison.py']:expected['tools/'+f]=sha(ROOT/'owner'/f)
  expected['source_origin.json']=sha(ROOT/'source_overlay/source_origin.json');assert expected==source
 print(json.dumps({'payloads_verified':len(m),'owner_payloads':26,'guard_payloads':2,'source_payloads_bound':931,'parent_full_source_verified':bool(args.parent_source),'physical_admission':False}))
if __name__=='__main__':main()
