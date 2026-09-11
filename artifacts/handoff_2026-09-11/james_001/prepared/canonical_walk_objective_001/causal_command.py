"""Command attribution at a policy boundary, independent of actor/physics implementations."""
import numpy as np


def begin_control(requested_command,actor_frame81,actual_held_target,current_q,control_index):
 c=np.asarray(requested_command,float);frame=np.asarray(actor_frame81,float);held=np.asarray(actual_held_target,float);q=np.asarray(current_q,float)
 n=len(c)
 if c.shape!=(n,3)or frame.shape!=(n,81)or held.shape!=(n,18)or q.shape!=(n,18):raise ValueError('Malformed policy boundary')
 if not all(np.isfinite(a).all()for a in [c,frame,held,q])or type(control_index)is not int or control_index<0:raise ValueError('Invalid policy boundary clock/value')
 # Compare the explicitly declared float32 policy packet, not host float64 spelling.
 if not np.array_equal(frame[:,6:9].astype(np.float32),c.astype(np.float32)):raise ValueError('New requested command must be visible before its first action')
 if not np.array_equal(frame[:,63:81].astype(np.float32),(held-q).astype(np.float32)):raise ValueError('Actor frame substituted a future/unexecuted target')
 return {'control_index':control_index,'reward_requested_command':c.copy(),'previous_actual_held_target':held.copy(),
         'new_command_precedes_action':True,'history_advanced_by_checker':False}
