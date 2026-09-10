"""Read-only exact owner/source verification; no CPU receipt admits physics."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 owner=json.loads((H/'FREEZE_SHA256.json').read_text())['files']
 for rel,h in owner.items():assert sha(H/rel)==h,rel
 build=json.loads((H/'SOURCE_BUILD.json').read_text());source=H/'source_pair_motion_001'
 assert sha(source/'campaign_source_hashes.json')==build['manifest_sha256']
 mapping=json.loads((source/'campaign_source_hashes.json').read_text())
 assert len(mapping)==946
 assert {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}==set(mapping)|{'campaign_source_hashes.json'}
 for rel,h in mapping.items():assert sha(source/rel)==h,rel
 for rel in ['pair_motion_contract.py','pair_motion_measurements.py','pair_motion_metrics.py','run_pair_motion.py','launch_pair_motion_spark.py','sensor_freshness.py','sensor_source_contract.json']:
  assert sha(H/rel)==sha(source/'tools'/rel),rel
 partial=source/'tools/paired_runtime';full_owner=json.loads((partial/'FREEZE_SHA256.json').read_text())['files']
 # This is the full upstream43-file identity, intentionally bundled with only
 # its13 required runtime payloads, not a claim that all upstream evidence is here.
 for p in partial.rglob('*'):
  if p.is_file() and p.name!='FREEZE_SHA256.json':assert sha(p)==full_owner[str(p.relative_to(partial))],str(p)
 for name in ['run_owned','owned_container','tree_hashes']:
  pick=lambda p:ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
  assert pick(source/'tools/launch_reference_physics_spark.py')==pick(source/'tools/launch_pair_motion_spark.py'),name
 print(json.dumps({'owner_payloads_verified':len(owner),'source_payloads_verified':len(mapping),'source_sha256':build['manifest_sha256'],
  'runtime_matches_owner_tested_files':True,'owned_cleanup_AST_unchanged':True,'physics_admitted':False},indent=2))
if __name__=='__main__':main()
