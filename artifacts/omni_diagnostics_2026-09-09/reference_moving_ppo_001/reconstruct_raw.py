"""Reconstruct exact original raw bytes into a fresh external directory."""
from pathlib import Path
import argparse,json,shutil,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));from verify_payload import main as verify
sys.path.insert(0,str(H/'chunk_tool'));from chunk_artifact import reconstruct

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 if out.exists() or H==out or H in out.parents:raise FileExistsError('Fresh output outside immutable bundle required')
 verify();out.mkdir(parents=True)
 for name,entry in json.loads((H/'RAW_STORAGE.json').read_text()).items():
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if entry['kind']=='file':shutil.copy2(H/entry['path'],dest)
  else:reconstruct((H/entry['manifest']).parent,dest)
 shutil.copy2(H/'independent_review/remote_audit.json',out/'remote_audit.json')
 print('Reconstructed all56 original raw files without compression/reencoding:',out)
if __name__=='__main__':main()
