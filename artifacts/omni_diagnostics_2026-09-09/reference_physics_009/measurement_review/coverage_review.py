"""Exact counter/timestamp/named control endpoint checks for actual009."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
rows=[]
for phase in ['standing','wave']:
 with np.load(a.run/phase/'physics_substeps.npz') as d,np.load(a.run/phase/'trace.npz') as t:
  n=len(t['time_s']);assert len(d['time_s'])==8*n+1
  np.testing.assert_array_equal(d['control_index'],np.r_[-1,np.repeat(np.arange(n),8)])
  np.testing.assert_array_equal(d['substep_index'],np.r_[0,np.tile(np.arange(1,9),n)])
  np.testing.assert_array_equal(d['relative_physics_index'],np.arange(8*n+1))
  np.testing.assert_array_equal(d['sim_step_counter'],d['sim_step_counter'][0]+np.arange(8*n+1))
  np.testing.assert_allclose(d['time_s'],np.arange(8*n+1)*.0025,rtol=0,atol=1e-12)
  np.testing.assert_allclose(d['sdk_sim_timestamp_s']-d['sdk_sim_timestamp_s'][0],d['time_s'],rtol=0,atol=1e-9)
  np.testing.assert_array_equal(d['joint_names'],t['joint_names'])
  for key in ['joint_position_rad','joint_velocity_rad_s','computed_torque_nm','applied_torque_nm']:np.testing.assert_array_equal(d[key][8::8],t[key])
  rows.append(dict(phase=phase,controls=n,samples=8*n+1,all_checks_passed=True,initial_counter=int(d['sim_step_counter'][0]),sdk_timestamp_max_error_s=float(np.abs(d['sdk_sim_timestamp_s']-d['sdk_sim_timestamp_s'][0]-d['time_s']).max())))
a.output.write_text(json.dumps(rows,indent=2)+'\n')
