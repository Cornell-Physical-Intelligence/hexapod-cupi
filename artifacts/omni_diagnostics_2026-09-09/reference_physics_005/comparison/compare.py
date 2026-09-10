"""Reproduce all-environment004→005 standing comparison with unchanged quiet scorer."""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
from substep_replay import review

HERE=Path(__file__).resolve().parent
SCORER=HERE/'frozen_quiet_scorer.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-run',type=Path,required=True);parser.add_argument('--candidate-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise FileExistsError('Fresh output required')
 tree=ast.parse(SCORER.read_text());nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QUIET_GATES' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='quiet_metrics')]
 assert len(nodes)==2
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SCORER),'exec'),ns)
 results={};quiet={};failures=[]
 for trial,path in [('004',args.baseline_run),('005',args.candidate_run)]:
  results[trial]=review(path/'standing');trace=path/'standing/trace.npz'
  with np.load(trace) as d:data={k:d[k] for k in d.files}
  assert data['joint_position_rad'].shape==(1000,32,18)
  rows=[ns['quiet_metrics'](data,e,200,data['joint_names'].tolist(),.02) for e in range(32)]
  quiet[trial]={'trace_sha256':sha(trace),'quiet_rows':rows,'pass_count':sum(r['pass'] for r in rows),'worst':{k:max(r[k] for r in rows) for k in ns['QUIET_GATES']}}
  if trial=='005':
   for e,row in enumerate(rows):
    if row['pass']:continue
    for name,jm in row['joints'].items():
     if jm['velocity_rms_rad_s']<=.03:continue
     j=data['joint_names'].tolist().index(name);q=data['joint_position_rad'][200:,e,j].astype(float);v=data['joint_velocity_rad_s'][200:,e,j].astype(float);fd=np.diff(q)/.02
     failures.append({'env':e,'joint':name,'failed_bounds':row['failed_bounds'],'reported_joint_RMS_rad_s':float(np.sqrt(np.mean(v*v))),'position_delta_50Hz_RMS_rad_s':float(np.sqrt(np.mean(fd*fd))),'position_range_rad':float(np.ptp(q)),'net_position_delta_rad':float(q[-1]-q[0]),'integrated_reported_joint_velocity_rad':float(v[1:].sum()*.02),'baseline004_reported_joint_RMS_rad_s':quiet['004']['quiet_rows'][e]['joints'][name]['velocity_rms_rad_s']})
 a=results['004']['post_settle'];b=results['005']['post_settle'];comparison={}
 for frame in ['link','com']:
  comparison[frame]={}
  for integrator in a[frame]['velocity_integrals']:
   av=np.array(a[frame]['velocity_integrals'][integrator]['displacement_difference_norm_m']);bv=np.array(b[frame]['velocity_integrals'][integrator]['displacement_difference_norm_m'])
   comparison[frame][integrator]={'baseline004_every_env_m':av.tolist(),'comparison005_every_env_m':bv.tolist(),'005_minus004_every_env_m':(bv-av).tolist(),'004_worst_m':float(av.max()),'005_worst_m':float(bv.max()),'improved_envs':int((bv<av).sum()),'increased_envs':int((bv>av).sum()),'004_count_above_5mm':int((av>.005).sum()),'005_count_above_5mm':int((bv>.005).sum())}
 args.output.mkdir(parents=True)
 for name,value in [('quiet_replay.json',{'scoring_source_sha256':sha(SCORER),'gates':ns['QUIET_GATES'],'trials':quiet}),('matched_comparison.json',comparison),('quiet_failures.json',{'failures':failures,'not_a_replacement_gate':True}),('all_substep_comparisons.json',results)]:
  (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({'quiet_pass004':quiet['004']['pass_count'],'quiet_pass005':quiet['005']['pass_count'],'output':str(args.output)},indent=2))
if __name__=='__main__':main()
