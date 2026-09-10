"""Reconstruct exact new source from a verified immutable directional002 parent."""
from pathlib import Path
import argparse,hashlib,json,shutil
from verify_payload import ROOT,verify,sha
ap=argparse.ArgumentParser();ap.add_argument('--parent-source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
p=args.parent_source.resolve();out=args.output.resolve()
if out.exists():raise FileExistsError('Fresh reconstruction output required')
if out==p or p in out.parents:raise ValueError('Output must be outside immutable parent source')
m=json.loads((ROOT/'BUNDLE_SHA256.json').read_text());verify(ROOT,m,['BUNDLE_SHA256.json'])
refs=json.loads((ROOT/'REFERENCED_INPUTS.json').read_text());assert sha(p/'campaign_source_hashes.json')==refs['source_parent']['sha256']
old=json.loads((p/'campaign_source_hashes.json').read_text());verify(p,old,['campaign_source_hashes.json'])
shutil.copytree(p,out)
for f in ['wave_reference.py','rr_preload_diagnostic.py','directional_contract.py','launch_directional_physics_spark.py','solver_comparison.py']:shutil.copy2(ROOT/'owner'/f,out/'tools'/f)
for f in ['source_origin.json','campaign_source_hashes.json']:shutil.copy2(ROOT/'source_overlay'/f,out/f)
new=json.loads((out/'campaign_source_hashes.json').read_text());verify(out,new,['campaign_source_hashes.json']);verify(p,old,['campaign_source_hashes.json'])
assert sha(out/'campaign_source_hashes.json')==refs['new_source_manifest_sha256']
print(json.dumps({'source_manifest_sha256':sha(out/'campaign_source_hashes.json'),'verified_source_payloads':len(new),'parent_unchanged':True,'physical_admission':False}))
