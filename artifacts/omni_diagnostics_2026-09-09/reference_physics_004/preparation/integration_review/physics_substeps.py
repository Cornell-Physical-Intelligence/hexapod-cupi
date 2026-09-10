"""Read-only post-scene-update telemetry; never replace DirectRLEnv.step.

The ordinary control-rate measurements and acceptance gates are retained.
Substep integration is diagnostic evidence, not a silently substituted velocity.
"""
from types import MethodType
from pathlib import Path
import json
import numpy as np
from physics_telemetry import array, rotation_xyzw


def measured_substep(env):
    d=env._robot.data
    return {'root_link_position_world_m':array(d.root_link_pos_w),
            'root_link_quaternion_world_xyzw':array(d.root_link_quat_w),
            'root_link_velocity_world_mps':array(d.root_link_lin_vel_w),
            'root_com_position_world_m':array(d.root_com_pos_w),
            'root_com_velocity_world_mps':array(d.root_com_lin_vel_w),
            'root_angular_velocity_world_rad_s':array(d.root_link_ang_vel_w),
            'computed_torque_nm':array(d.computed_torque),
            'applied_torque_nm':array(d.applied_torque),
            'joint_velocity_rad_s':array(d.joint_vel)}


class PhysicsSubstepRecorder:
    """Observe exactly eight real updates per control and restore the method."""
    def __init__(self,env,*,reader=measured_substep):
        if getattr(env,'_physics_handles_decimation',None) is not False:
            raise ValueError('Substep proof requires explicit backend-independent decimation')
        if env.cfg.decimation!=8 or abs(env.physics_dt-.0025)>1e-12 or abs(env.step_dt-.02)>1e-12:
            raise ValueError('Substep proof requires exact eight .0025s updates per .02s control')
        self.env,self.reader=env,reader
        self.original=None;self.rows=[];self.active=False;self.control=-1;self.count=0
        self.completed_controls=0;self.error=None
        self.base_counter=int(env._sim_step_counter)
        d=env._robot.data
        self.identity={'articulation_data_class':type(d).__module__+'.'+type(d).__qualname__,
                       'backend_name':getattr(d,'__backend_name__',None),
                       'joint_names_runtime':list(env._robot.joint_names),
                       'physics_handles_decimation':False,'decimation':8,'physics_dt_s':.0025,
                       'capture':'after original scene.update, before next simulation substep/reset',
                       'ordinary_control_metrics_unchanged':True}

    def _capture(self,control,substep):
        relative=len(self.rows)
        if int(self.env._sim_step_counter)!=self.base_counter+relative:
            raise RuntimeError('Simulation counter does not match every observed physics substep')
        row=self.reader(self.env)
        row.update(relative_physics_index=np.asarray(relative,dtype=np.int64),
                   control_index=np.asarray(control,dtype=np.int64),
                   substep_index=np.asarray(substep,dtype=np.int8),
                   time_s=np.asarray(relative*.0025),
                   sim_step_counter=np.asarray(self.env._sim_step_counter,dtype=np.int64),
                   sdk_sim_timestamp_s=np.asarray(self.env._robot.data._sim_timestamp,dtype=float))
        self.rows.append(row)
        if any(not np.isfinite(v).all() for v in row.values()):
            raise ValueError('Nonfinite substep state; retained raw failing sample')
        if len(self.rows)>1:
            delta=float(row['sdk_sim_timestamp_s']-self.rows[-2]['sdk_sim_timestamp_s'])
            if abs(delta-.0025)>1e-7:
                raise RuntimeError('SDK articulation time did not advance exactly one physics substep')

    def __enter__(self):
        if self.original is not None:raise RuntimeError('Substep observer cannot be nested/reused')
        self.original=self.env.scene.update
        recorder=self
        def update(scene,*args,**kwargs):
            if not recorder.active:
                raise RuntimeError('Unexpected scene update outside an active control')
            dt=kwargs.get('dt',args[0] if args else None)
            if dt is None or abs(float(dt)-.0025)>1e-12 or recorder.count>=8:
                raise RuntimeError('Expected exactly eight real physics-dt scene updates')
            result=recorder.original(*args,**kwargs)
            recorder.count+=1
            recorder._capture(recorder.control,recorder.count)
            return result
        try:
            self._capture(-1,0)
        except Exception as exc:
            self.error=repr(exc)
            raise
        self.env.scene.update=MethodType(update,self.env.scene)
        return self

    def begin_control(self,index):
        if self.active or index!=self.completed_controls:
            raise RuntimeError('Sequential controls required; do not omit/reorder/reset the observer')
        self.control=index;self.count=0;self.active=True

    def end_control(self,row):
        if not self.active or self.count!=8:
            raise RuntimeError('Missing physics-substep measurements')
        last=self.rows[-1]
        # Different property aliases must agree at the same post-step instant.
        for field,ordinary in [('root_link_position_world_m','position_world_m'),
                               ('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),
                               ('root_link_velocity_world_mps','velocity_world_mps'),
                               ('computed_torque_nm','computed_torque_nm'),
                               ('applied_torque_nm','applied_torque_nm')]:
            if not np.array_equal(last[field],row[ordinary]):
                raise RuntimeError('Control/substep endpoint mismatch: '+field)
        self.active=False;self.completed_controls+=1

    def __exit__(self,kind,value,traceback):
        self.env.scene.update=self.original
        if value is not None:self.error=repr(value)
        return False

    def data(self):
        return {key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]} if self.rows else {}

    def export(self,output):
        output=Path(output);d=self.data()
        if d:np.savez_compressed(output/'physics_substeps.npz',**d,
                                joint_names=np.asarray(self.identity['joint_names_runtime']))
        status={**self.identity,'samples_including_initial':len(self.rows),
                'completed_controls':self.completed_controls,'error':self.error,
                'method_restored':self.env.scene.update==self.original,
                'physical_acceptance_gate_changed':False}
        # Preserve partial/nonfinite raw traces even when derived metrics cannot
        # be formed; no missing window is converted to a passing gate.
        if d and all(np.isfinite(value).all() for value in d.values()):
            integrals=control_integrals(d,self.completed_controls)
            np.savez_compressed(output/'physics_control_integrals.npz',**integrals)
            if self.completed_controls>200:
                status['postsettle_prefix']=displacement_check(d,200,self.completed_controls)
                status['moving_window_or_prefix']=displacement_check(d,200,min(self.completed_controls,1400))
            status['all_recorded_substep_max_requested_torque_nm']=float(np.abs(d['computed_torque_nm']).max())
            status['all_recorded_substep_max_applied_torque_nm']=float(np.abs(d['applied_torque_nm']).max())
            if self.completed_controls>200:
                selected=d['computed_torque_nm'][200*8+1:(self.completed_controls*8)+1]
                status['postsettle_substep_max_requested_torque_nm']=float(np.abs(selected).max())
                status['postsettle_worst_joint_substep_saturation_fraction']=float((np.abs(selected)>1.6).mean(0).max())
        (output/'physics_substep_review.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
        return status


def control_integrals(data,controls):
    """All eight actual substeps: retain left/right/trapezoid values separately."""
    if len(data['time_s'])<controls*8+1:raise ValueError('Incomplete control/substep window')
    result={}
    for frame in ('link','com'):
        velocity=data[f'root_{frame}_velocity_world_mps'][:controls*8+1]
        position=data[f'root_{frame}_position_world_m'][:controls*8+1]
        shape=(controls,8)+velocity.shape[1:]
        left=velocity[:-1].reshape(shape).sum(1)*.0025
        right=velocity[1:].reshape(shape).sum(1)*.0025
        result[f'{frame}_velocity_left_integral_m']=left
        result[f'{frame}_velocity_right_integral_m']=right
        result[f'{frame}_velocity_trapezoid_integral_m']=.5*(left+right)
        result[f'{frame}_velocity_substep_mean_mps']=right/.02
        result[f'{frame}_position_displacement_m']=np.diff(position[::8],axis=0)
    for kind in ('computed','applied'):
        torque=data[f'{kind}_torque_nm'][1:controls*8+1]
        torque=torque.reshape((controls,8)+torque.shape[1:])
        result[f'{kind}_torque_abs_substep_max_nm']=np.abs(torque).max(1)
        result[f'{kind}_torque_substep_excess_count_1p6']=np.sum(np.abs(torque)>1.6,axis=1)
    return result


def displacement_check(data,start_control,end_control):
    if not 0<=start_control<end_control or len(data['time_s'])<=end_control*8:
        raise ValueError('Complete explicit physics interval required')
    lo,hi=start_control*8,end_control*8
    result={'start_control_boundary':start_control,'end_control_boundary':end_control,
            'duration_s':(end_control-start_control)*.02,'acceptance_verdict':None,
            'scope':'independent raw400Hz evidence; old50Hz gate unchanged'}
    rotation=rotation_xyzw(data['root_link_quaternion_world_xyzw'][lo])
    axis=-rotation[:,:,1];axis[:,2]=0.;axis/=np.linalg.norm(axis,axis=1)[:,None]
    for frame in ('link','com'):
        p=data[f'root_{frame}_position_world_m'];v=data[f'root_{frame}_velocity_world_mps'][lo:hi+1]
        displacement=p[hi]-p[lo]
        integrals={'left':v[:-1].sum(0)*.0025,'right':v[1:].sum(0)*.0025}
        integrals['trapezoid']=.5*(integrals['left']+integrals['right'])
        result[frame]={'measured_world_displacement_m':displacement.tolist(),
                       'measured_forward_displacement_m':np.sum(displacement*axis,axis=-1).tolist(),
                       'integrals':{name:{'world_m':value.tolist(),
                           'forward_m':np.sum(value*axis,axis=-1).tolist(),
                           'position_difference_norm_m':np.linalg.norm(displacement-value,axis=-1).tolist()}
                           for name,value in integrals.items()}}
    return result
