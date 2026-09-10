"""Reconstruct932 exact payloads from immutable931 source001 plus import delta."""
from pathlib import Path
import argparse,json,shutil
from verify_payload import ROOT,FILES,verify,sha
ap=argparse.ArgumentParser();ap.add_argument('--parent-source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=a.parent_source.resolve();out=a.output.resolve()
if out.exists() or out==p or p in out.parents:raise ValueError('Fresh output outside parent required')
verify(ROOT,json.loads((ROOT/'BUNDLE_SHA256.json').read_text()),['BUNDLE_SHA256.json']);r=json.loads((ROOT/'REFERENCED_INPUTS.json').read_text());assert sha(p/'campaign_source_hashes.json')==r['source_parent']['manifest_sha256'];old=json.loads((p/'campaign_source_hashes.json').read_text());verify(p,old,['campaign_source_hashes.json'])
shutil.copytree(p,out)
for f in FILES:shutil.copy2(ROOT/'owner'/f,out/'tools'/f)
for f in ['source_origin.json','campaign_source_hashes.json']:shutil.copy2(ROOT/'source_overlay'/f,out/f)
verify(out,json.loads((out/'campaign_source_hashes.json').read_text()),['campaign_source_hashes.json']);verify(p,old,['campaign_source_hashes.json']);assert sha(out/'campaign_source_hashes.json')==r['new_source_manifest_sha256'];print(json.dumps({'source_payloads_verified':932,'source_manifest_sha256':r['new_source_manifest_sha256'],'parent_unchanged':True,'physical_admission':False}))
