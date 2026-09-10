"""Reconstruct every original raw byte to a fresh external folder; preserve all failures."""
from pathlib import Path
import argparse,json,shutil,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));from verify_payload import main as verify,sha
sys.path.insert(0,str(H/'chunk_tool'));from chunk_artifact import reconstruct

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 if out.exists() or H==out or H in out.parents:raise FileExistsError('Fresh output outside immutable bundle required')
 verify();out.mkdir(parents=True)
 storage=json.loads((H/'RAW_STORAGE.json').read_text())
 for name,r in storage.items():
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if r['kind']=='file':shutil.copy2(H/r['path'],dest)
  else:reconstruct((H/r['manifest']).parent,dest)
  assert dest.stat().st_size==r['bytes'] and sha(dest)==r['sha256'],name
 shutil.copy2(H/'terminal_audit/remote_audit.json',out/'remote_audit.json')
 print('Reconstructed and SHA-verified all 46 original raw files without compression/reencoding:',out)
if __name__=='__main__':main()
