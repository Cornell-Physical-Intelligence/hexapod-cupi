"""Reconstruct exact946-payload source from immutable009 plus21 declared deltas."""
from pathlib import Path
import argparse,json,hashlib,shutil
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,mapping):
 if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(mapping)|{'campaign_source_hashes.json'}:raise ValueError('Source file set differs')
 for rel,digest in mapping.items():
  p=(root/rel).resolve()
  if root.resolve() not in p.parents or sha(p)!=digest:raise ValueError('Source mismatch: '+rel)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 parent=a.parent.resolve();out=a.output.resolve();bindings=json.loads((H/'BINDINGS.json').read_text())
 if out.exists() or out==parent or parent in out.parents:raise ValueError('Fresh separate output required')
 if sha(parent/'campaign_source_hashes.json')!=bindings['parent_manifest_sha256']:raise ValueError('Exact original009 parent required')
 old=json.loads((parent/'campaign_source_hashes.json').read_text());verify(parent,old)
 mapping=json.loads((H/'source_hashes.json').read_text());delta=H/'source_delta'
 changed={rel for rel,digest in mapping.items() if old.get(rel)!=digest}
 if {str(p.relative_to(delta)) for p in delta.rglob('*') if p.is_file()}!=changed:raise ValueError('Exact delta set required')
 for rel in changed:
  if sha(delta/rel)!=mapping[rel]:raise ValueError('Delta mismatch: '+rel)
 shutil.copytree(parent,out)
 for rel in changed:
  target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(delta/rel,target)
 shutil.copy2(H/'source_hashes.json',out/'campaign_source_hashes.json');verify(out,mapping)
 if sha(out/'campaign_source_hashes.json')!=bindings['new_source_manifest_sha256']:raise ValueError('Reconstructed source identity differs')
 print(json.dumps({'files':len(mapping),'manifest_sha256':sha(out/'campaign_source_hashes.json'),'GPU_launched':False},indent=2))
if __name__=='__main__':main()
