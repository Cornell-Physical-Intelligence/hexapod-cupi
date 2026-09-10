"""Keep initial actuator-buffer evidence separate from actual physics substeps."""
from pathlib import Path
import argparse,json,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Fresh output required')
 result={}
 for phase in ['standing','wave']:
  with np.load(a.run/phase/'physics_substeps.npz') as d:
   torque=d['computed_torque_nm'];applied=d['applied_torque_nm'];idx=np.unravel_index(np.abs(torque).argmax(),torque.shape);sample,e,j=idx
   peak={'sample':int(sample),'time_s':float(d['time_s'][sample]),'control_index':int(d['control_index'][sample]),'substep_index':int(d['substep_index'][sample]),'env':int(e),'joint_name':str(d['joint_names'][j]),'requested_nm':float(torque[idx]),'applied_same_sample_nm':float(applied[idx]),'captured_before_first_actual_physics_update':bool(sample==0)}
   pre=np.abs(torque[1:1601]);k=np.unravel_index(pre.argmax(),pre.shape);i=k[0]+1
   result[phase]={'all_recorded_computed_torque_peak':peak,'all_recorded_applied_max_abs_nm':float(np.abs(applied).max()),'first_16samples_computed_max_abs_nm':np.abs(torque[:17]).max(axis=(1,2)).tolist(),'first_16samples_applied_max_abs_nm':np.abs(applied[:17]).max(axis=(1,2)).tolist(),'postsettle_max_computed_abs_nm':float(np.abs(torque[1601:]).max()),'postsettle_max_applied_abs_nm':float(np.abs(applied[1601:]).max()),'above_rating_sample_indices':np.where((np.abs(torque)>1.6).any(axis=(1,2)))[0].tolist(),'actual_post_update_pre_settle_peak':{'sample':int(i),'time_s':float(d['time_s'][i]),'env':int(k[1]),'joint_name':str(d['joint_names'][k[2]]),'requested_abs_nm':float(pre[k]),'applied_same_sample_nm':float(applied[i,k[1],k[2]])}}
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
