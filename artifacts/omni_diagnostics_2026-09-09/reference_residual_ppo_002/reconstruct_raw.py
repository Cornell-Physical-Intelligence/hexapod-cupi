"""Restore the complete original80-file raw tree to a new directory; no overwrite."""
from pathlib import Path
import argparse,json,shutil,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
from verify_payload import main as verify,read,sha
sys.path.insert(0,str(ROOT/'chunk_tool'))
from chunk_artifact import reconstruct

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();output=args.output.resolve()
 if output.exists() or ROOT.resolve() in output.parents:raise ValueError('Fresh output directory outside publication required')
 verify();output.mkdir(parents=True,exist_ok=False)
 for name,entry in read(ROOT/'RAW_STORAGE.json').items():
  target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
  if entry['kind']=='file':shutil.copy2(ROOT/entry['path'],target)
  else:reconstruct((ROOT/entry['manifest']).parent,target)
  assert target.stat().st_size==entry['bytes'] and sha(target)==entry['sha256']
 print(json.dumps({'reconstructed_original_files':80,'output':str(output),'source_originals_unchanged':True,'GPU_launches':0}))
if __name__=='__main__':main()
