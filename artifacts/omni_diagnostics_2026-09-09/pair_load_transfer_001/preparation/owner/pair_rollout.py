"""One original physics step, with every captured pre-reset row retained first."""
import torch
from physics_telemetry import require_single_pre_reset_sample

def emit_control(env,ref,captured,recorder,local_step,destination_rows,zero):
 before=len(captured);recorder.begin_control(local_step)
 with torch.inference_mode():
  env.set_reference_targets(torch.as_tensor(ref['q_ref'],device=env.device),
   torch.as_tensor(ref.get('analytic_velocity_rad_s',ref['v_ref']),device=env.device),
   torch.as_tensor(ref.get('analytic_acceleration_rad_s2',ref['a_ref']),device=env.device),
   torch.as_tensor(ref['valid'],device=env.device))
  env.set_evaluation_targets(torch.zeros((1,3),device=env.device,dtype=zero.dtype))
  obs,reward,term,trunc,_=env.step(zero)
 row=require_single_pre_reset_sample(captured,before,term,trunc)
 destination_rows.append(row)
 recorder.end_control(row)
 if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(reward).all():
  raise ValueError('Nonfinite observation/reward after preserved pre-reset physical row')
 return row,term,trunc

def check_measured_pair_state(generator,row):
 """Apply the frozen proposed measurement bounds to every completed row, including last."""
 import numpy as np
 generator._measure(row,generator.base._read(row))
 if np.abs(row['reference_to_executable_lag_rad']).max()>1e-12:
  raise ValueError('Zero-residual executable reference lag exceeds unchanged exact-knot contract')
 if np.abs(row['position_target_cast_error_rad']).max()>2e-7:
  raise ValueError('Actual position-target cast exceeds existing contract')
 return {'time_s':float(row['time_s'][0]),'all_existing_proposed_measurement_bounds_met':True,**generator.diagnostics}
