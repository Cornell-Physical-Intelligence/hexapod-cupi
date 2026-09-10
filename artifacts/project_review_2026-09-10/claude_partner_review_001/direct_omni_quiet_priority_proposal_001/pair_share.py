"""NumPy-only mode census, aligned with actually recorded CAPS valid-pair fractions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,default=H.parent/'direct_omni_matched_pilots_publication_001');p.add_argument('--output',type=Path,default=H/'pair_share.json');args=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={}
for b in ['curriculum','caps']:
 d=args.campaign_root/b/'raw/run/train';f=d/'training_trace.npz';receipt=d/'training_receipt.json'
 with np.load(f) as z:c=z['command'];done=z['terminated']|z['truncated']
 if c.shape!=(1200,1024,3):raise RuntimeError('Unexpected coverage')
 valid=np.zeros(c.shape[:2],bool);valid[1:]=(c[1:]==c[:-1]).all(-1)&~done[:-1]
 quiet=(c==0).all(-1);v=valid.reshape(50,24,1024);q=quiet.reshape(50,24,1024)
 updates=[]
 r=json.loads(receipt.read_text())
 for i in range(50):
  total=int(v[i].sum());nq=int((v[i]&q[i]).sum());u={'update':i+1,'valid_pairs':total,'valid_quiet_pairs':nq,'quiet_share':nq/total,'valid_fraction':float(v[i].mean())}
  if b=='caps':
   observed=r['optimizer_updates'][i]['losses']['caps_valid_pair_fraction'];u['receipt_valid_fraction']=observed;u['abs_error']=abs(observed-u['valid_fraction'])
   if u['abs_error']>1e-6:raise RuntimeError('Actual receipt pairing differs')
  updates.append(u)
 result[b]={'inputs':{str(f):sha(f),str(receipt):sha(receipt)},'updates':updates,'valid_pairs':int(valid.sum()),'valid_quiet_pairs':int((valid&quiet).sum()),'quiet_share':float((valid&quiet).sum()/valid.sum()),'caps_receipt_max_abs_fraction_error':max(u.get('abs_error',0) for u in updates),'scope':'All1024 command/terminal rows; exact source PairState command/reset mask. Does not reconstruct unlogged noisy actor inputs or gradients.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ['inputs','updates']} for k,v in result.items()},indent=2))
