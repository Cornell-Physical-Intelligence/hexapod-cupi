"""Preserve the first catastrophic contacts independently of the first mild clip."""
import argparse,csv,gzip,hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise ValueError('Output must be new')
r=json.loads(a.report.read_text());record=next(row for row in r['trace_files'] if row['first_physics_sample']==36000)
path=a.report.parent/record['file'];sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(path)!=record['sha256']:raise ValueError('Trace hash mismatch')
with np.load(path,allow_pickle=False) as z:v,c=z['values'],z['columns'].tolist()
a.out.mkdir(parents=True);events=[];csv_path=a.out/'secondary_impacts_exact_samples.csv.gz'
with gzip.open(csv_path,'wt',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['physics_sample','environment',*c])
 for env,first in ((9,36226),(0,36408),(16,36422)):
  for sample in range(first-2,first+2):
   i=sample-36000;writer.writerow([sample,env,*[format(float(x),'.9g') for x in v[i,env]]])
   fields={key:float(v[i,env,c.index(key+'/lf_femur_pitch')]) for key in ('pre_q','post_q','pre_qd','post_qd','demand','applied','instantaneous_limit','headroom')}
   fields['reported_LF_foot_force_xyz_N']=[float(v[i,env,c.index('foot_force_w/lf_'+axis)]) for axis in 'xyz']
   events.append(dict(sample=sample,environment=env,reset_xyz_m=r['reset_root_positions_m'][env],LF_femur=fields))
result={'schema':'hexapod.prefix_secondary_impacts.v1','simulation_training_admission':False,'hardware_admission':False,
        'inputs':{str(path):sha(path),str(a.report):sha(a.report),str(Path(__file__)):sha(__file__)},'events':events,
        'csv':{'file':csv_path.name,'sha256':sha(csv_path),'rows':len(events),'encoding':'9 significant digits preserve float32 exactly'}}
(a.out/'secondary_impacts.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'events':len(events),'csv_sha256':sha(csv_path)}))
