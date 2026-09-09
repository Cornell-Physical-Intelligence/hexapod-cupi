#!/usr/bin/env python3
"""Measure actual captured policy motion; never infer walking success from reward."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError('Expected finite Nx4 XYZW quaternions')
    length = np.linalg.norm(q, axis=1)
    if np.any(np.abs(length - 1) > 1e-3):
        raise ValueError('Captured quaternion is not unit length')
    x,y,z,w = (q / length[:,None]).T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1,3,3)


def summarize(states, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite control period required')
    names = ('pre_root_pos_w_m','post_root_pos_w_m','pre_root_quat_w_xyzw',
             'post_root_quat_w_xyzw','pre_command_navigation','done')
    missing = set(names)-set(states)
    if missing:
        raise ValueError('Missing capture fields: '+str(sorted(missing)))
    a = {k:np.asarray(states[k]) for k in names}
    n = len(a['done'])
    for k in names:
        shape = (n,) if k=='done' else (n,4 if 'quat' in k else 3)
        if a[k].shape != shape or not np.isfinite(a[k]).all():
            raise ValueError('Invalid shape/nonfinite field: '+k)
    if not np.isin(a['done'], [0,1]).all():
        raise ValueError('Done must contain only false/true values')
    keep = ~a['done'].astype(bool)
    if not keep.any():
        raise ValueError('No transition remains after excluding automatic resets')
    pre_r,post_r = rotation(a['pre_root_quat_w_xyzw']),rotation(a['post_root_quat_w_xyzw'])
    delta = a['post_root_pos_w_m']-a['pre_root_pos_w_m']
    body_velocity = np.einsum('nji,nj->ni',pre_r,delta)/dt
    # Relative rotation expressed in the preceding body frame. Its log Z is
    # equivalent short-interval rotation rate, not world heading or an exact instantaneous rate.
    rel = np.einsum('nji,njk->nik',pre_r,post_r)
    cosine = np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)
    angle = np.arccos(cosine)
    if np.any(angle[keep] > .5):
        raise ValueError('Non-reset orientation jump too large for short-step velocity estimate')
    factor = np.ones(n)*.5
    nonzero = angle>1e-7
    factor[nonzero] = angle[nonzero]/(2*np.sin(angle[nonzero]))
    yaw_rate = factor*(rel[:,1,0]-rel[:,0,1])/dt
    actual = np.column_stack((-body_velocity[:,1],body_velocity[:,0],yaw_rate))[keep]
    command = a['pre_command_navigation'][keep].astype(np.float64)
    error = actual-command
    planar_speed = np.linalg.norm(actual[:,:2],axis=1)
    command_speed = np.linalg.norm(command[:,:2],axis=1)
    commanded_moving = command_speed >= .02
    return {'schema':'hexapod.policy_rollout_motion_analysis.v1','scope':'descriptive motion metrics on the recorded flat rollout; no navigation, terrain or hardware admission',
        'control_dt_s':dt,'transitions_total':n,'transitions_used':int(keep.sum()),
        'excluded_reset_transition_indices':np.flatnonzero(~keep).tolist(),
        'observed_nonreset_duration_s':float(keep.sum()*dt),
        'navigation_axis_order':['forward (-body Y)','left (+body X)','yaw (+body Z)'],
        'velocity_estimator':'finite difference of plate-origin position and relative orientation over each control transition, expressed in its pre-action body frame',
        'training_reward_comparison':'These plate-origin interval estimates are not the exact instantaneous training reward velocities. Installed SDK root_pos_w aliases root_link_pos_w, while root_lin_vel_b aliases root_com_lin_vel_b. COM velocity additionally contains angular_velocity cross COM_offset; SO(3) log-Z is a finite-interval estimate.',
        'sdk_alias_provenance':{'source':'/home/orionh/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py',
            'sha256':'677d500672162d9bdc04b417c726b0d61edc839e0396cb3faf15316fbca28067',
            'root_pos_w_alias_line':1156,'root_lin_vel_b_alias_line':1186,'root_ang_vel_b_alias_line':1192},
        'mean_command':command.mean(axis=0).tolist(),'mean_measured_velocity':actual.mean(axis=0).tolist(),
        'tracking_rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),
        'tracking_mae':np.mean(np.abs(error),axis=0).tolist(),
        'tracking_absolute_error_p95':np.quantile(np.abs(error),.95,axis=0).tolist(),
        'actual_planar_path_length_m':float(np.linalg.norm(delta[keep,:2],axis=1).sum()),
        'commanded_planar_distance_m':float(command_speed.sum()*dt),
        'moving_command_transitions':int(commanded_moving.sum()),
        'near_stationary_fraction_during_moving_commands':float(np.mean(planar_speed[commanded_moving]<.01)) if commanded_moving.any() else None,
        'base_height_min_m':float(np.minimum(a['pre_root_pos_w_m'][keep,2],a['post_root_pos_w_m'][keep,2]).min()),
        'body_tilt_p95_deg':float(np.rad2deg(np.quantile(np.arccos(np.clip(post_r[keep,2,2],-1,1)),.95))),
        'acceptance_pass':None,'acceptance_reason':'No task-level numerical success threshold has been established; reward and finite inference are insufficient.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.capture_dir.resolve()
    report_bytes=(base/'report.json').read_bytes()
    report=json.loads(report_bytes)
    if report.get('pass') is not True:
        raise ValueError('Requires completed passing capture report')
    for name in ('states.npz','metadata.json'):
        expected=report['artifacts'][name]
        data=(base/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected['sha256'] or len(data)!=expected['bytes']:
            raise ValueError('Capture artifact bytes changed: '+name)
    metadata=json.loads((base/'metadata.json').read_text())
    if metadata.get('root_quaternion_order')!='XYZW':
        raise ValueError('Unsupported quaternion convention')
    with np.load(base/'states.npz',allow_pickle=False) as states:
        result=summarize(states,1/metadata['fps'])
    result['source_capture']=str(base)
    result['capture_report_sha256']=hashlib.sha256(report_bytes).hexdigest()
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['capture_input_sha256']=metadata.get('input_sha256')
    result['source_sha256']=metadata['source_sha256']
    result['input_artifacts']=report['artifacts']
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
