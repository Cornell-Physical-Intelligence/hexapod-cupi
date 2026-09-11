"""Read-only CPU counterexamples against the frozen composition proposal."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
root=args.source.resolve();manifest=root/'FREEZE_SHA256.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads(manifest.read_text())['files']
assert sha(manifest)=='f6b239605e84389a9c3a9b292932b64304c54a853a28c85e83267234b1e637b9'
actual={p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()and p!=manifest};assert actual==m
sys.path.insert(0,str(root))
import numpy as np
from test_composition import rows
from reward_packet import pack_reward,neutral_interval_context
cases={}
def check(name,call):
 try:
  value=call();cases[name]={'rejected':False,'reward':value.get('reward',np.array([])).tolist(),'counter_interval':value.get('counter_interval')}
 except Exception as e:cases[name]={'rejected':True,'error':repr(e)}
b,s,k=rows(1);s[3]['time_s']=float('nan');check('single_NaN_substep_time',lambda:pack_reward(b,s,**k))
b,s,k=rows(1);b['time_s']=float('nan');check('NaN_before_time',lambda:pack_reward(b,s,**k))
b,s,k=rows(1);b['explicit_counter']+=.5
for row in s:row['explicit_counter']+=.5;row['substep_index']+=.5
check('fractional_counters_and_substep_indices',lambda:pack_reward(b,s,**k))
b,s,k=rows(1);s[3]['time_s']+=.01;check('finite_wrong_time_control',lambda:pack_reward(b,s,**k))
b,s,k=rows(1);last=s[-1];prior=copy.deepcopy(last);prior['explicit_counter']-=8;prior['time_s']-=.02;last['time_s']=float('nan');last['link_pose_xyzw']=np.tile([0.,0,0,0,0,0,1],(1,6,1));check('neutral_context_NaN_time',lambda:neutral_interval_context(prior,last))
assert not cases['single_NaN_substep_time']['rejected'] and not cases['NaN_before_time']['rejected'] and not cases['fractional_counters_and_substep_indices']['rejected'] and not cases['neutral_context_NaN_time']['rejected']
assert cases['finite_wrong_time_control']['rejected']
assert {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()and p!=manifest}==m
print(json.dumps({'frozen_source_sha256':sha(manifest),'source_payloads':len(m),'source_unchanged':True,'scope':'CPU proposal only; no native source, physics gate or policy modified','cases':cases},indent=2,allow_nan=False))
