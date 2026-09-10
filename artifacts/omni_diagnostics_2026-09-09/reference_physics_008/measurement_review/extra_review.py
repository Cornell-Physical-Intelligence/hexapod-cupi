"""Reproduce incomplete-wave original progress and separate startup torque intervals."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from frozen_screen_metrics import measured_progress
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 result={}
 for phase in ['standing','wave']:
  root=a.run/phase
  with np.load(root/'trace.npz') as z:t={k:z[k] for k in z.files}
  with np.load(root/'physics_substeps.npz') as z:d={k:z[k] for k in z.files}
  n=len(t['time_s']);torque=[]
  for name,lo,hi in [('initial_before_first_observed_update',0,1),('actual_updates_before_scoring',1,1601),('scored_updates',1601,len(d['time_s']))]:
   v=d['computed_torque_nm'][lo:hi];k,e,j=np.unravel_index(np.abs(v).argmax(),v.shape);k+=lo
   torque.append({'interval':name,'sample_index':int(k),'env':int(e),'joint':str(d['joint_names'][j]),'time_s':float(d['time_s'][k]),'requested_signed_nm':float(d['computed_torque_nm'][k,e,j]),'applied_signed_nm':float(d['applied_torque_nm'][k,e,j]),'max_abs_requested_nm':float(np.abs(v).max()),'samples_above1p6':int((np.abs(v)>1.6).sum())})
  result[phase]={'controls':n,'trace_sha256':sha(root/'trace.npz'),'substeps_sha256':sha(root/'physics_substeps.npz'),'torque_intervals':torque,'hardware_startup_qualified':False}
  if phase=='wave':
   result[phase]['original_progress_incomplete_moving_prefix']=measured_progress(t,start_step=199,end_step=n-1)
   result[phase]['full24s_progress_or_quiet_stop_admitted']=False
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
