"""Portable read-only compact002 and exact optional parent verification."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent
FILES=['rr_preload_contract.py','rr_preload_diagnostic.py','directional_contract.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,m,extra=()):
 if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(m)|set(extra):raise ValueError('Extra or missing payloads')
 for f,h in m.items():
  p=(root/f).resolve()
  if root.resolve() not in p.parents or sha(p)!=h:raise ValueError('Path/hash mismatch '+f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-source',type=Path);a=ap.parse_args();verify(ROOT,json.loads((ROOT/'BUNDLE_SHA256.json').read_text()),['BUNDLE_SHA256.json'])
 refs=json.loads((ROOT/'REFERENCED_INPUTS.json').read_text());assert sha(ROOT/'owner/FREEZE_SHA256.json')==refs['new_owner_freeze_sha256'];verify(ROOT/'owner',json.loads((ROOT/'owner/FREEZE_SHA256.json').read_text()),['FREEZE_SHA256.json'])
 p=ROOT/'source_overlay/campaign_source_hashes.json';new=json.loads(p.read_text());assert len(new)==932 and sha(p)==refs['new_source_manifest_sha256']
 for f in FILES:assert sha(ROOT/'owner'/f)==new['tools/'+f]
 assert sha(ROOT/'source_overlay/source_origin.json')==new['source_origin.json']
 if a.parent_source:
  oldp=a.parent_source.resolve();assert sha(oldp/'campaign_source_hashes.json')==refs['source_parent']['manifest_sha256'];old=json.loads((oldp/'campaign_source_hashes.json').read_text());verify(oldp,old,['campaign_source_hashes.json']);merged=dict(old)
  for f in FILES:merged['tools/'+f]=sha(ROOT/'owner'/f)
  merged['source_origin.json']=sha(ROOT/'source_overlay/source_origin.json');assert merged==new
 print(json.dumps({'owner_payloads':17,'source_payloads_bound':932,'parent_verified':bool(a.parent_source),'physical_admission':False}))
if __name__=='__main__':main()
