"""Read-only stdlib check of the exact guard's persistent reservation on Spark."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--guard',type=Path,required=True);parser.add_argument('--expected-runtime',required=True);args=parser.parse_args()
 entry=args.guard/'launch_guarded_remote.py';policy=args.guard/'inputs/reservation_policy.json'
 if sha(entry)!=args.expected_runtime:raise RuntimeError('Unexpected guard runtime')
 before=(sha(entry),sha(policy));s=importlib.util.spec_from_file_location('actual_reservation_guard',entry);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
 g.require_final_bindings();result=g.verify_reservation()
 if before!=(sha(entry),sha(policy)):raise RuntimeError('Inputs changed during readback')
 assert not any(k in sys.modules for k in ('numpy','torch','isaaclab'))
 print(json.dumps({'verified':True,'read_only':True,'native_or_GPU_launched':False,'python':sys.version,'observed_unix':time.time(),'runtime_sha256':before[0],'policy_sha256':before[1],'reservation':result},indent=2))
if __name__=='__main__':main()
