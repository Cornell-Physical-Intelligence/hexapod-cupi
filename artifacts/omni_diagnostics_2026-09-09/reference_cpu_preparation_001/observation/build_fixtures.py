"""Capture small exact-source CPU fixtures; no new physics or deployment claim."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WAVE=ROOT/'tmp/omni_reference_wave_002'
sys.path.insert(0,str(WAVE))
from test_landing import ActualPrefix,LandedFixture
from observation import WAVE_SHA256,RESIDUAL_SHA256,serial

KEYS=('time_s','position_world_m','rotation_world_from_body','joint_position_rad',
      'joint_velocity_rad_s','joint_target_rad','gyro_body_rad_s','projected_gravity_body',
      'velocity_body_mps','reference_point_world_m','reference_point_velocity_world_mps',
      'contact_point_world_m','contact_point_valid','distal_contact','shaft_contact',
      'coxa_contact','femur_contact','base_contact','terminated','truncated')


def packet(snapshot,reference,source):
    s={k:np.array(snapshot[k],copy=True) for k in KEYS}
    if source=='synthetic_fixture':
        # The upstream synthetic fixture replaces body rotation; derive this
        # vector in the same new frame rather than copy its old physical value.
        s['projected_gravity_body']=-s['rotation_world_from_body'][:,2,:]
    # Convert explicitly invalid contact-point storage to the documented finite
    # sentinel. Validity is retained; no unknown point becomes support.
    s['contact_point_world_m'][~s['contact_point_valid']]=0
    r=deepcopy(reference);epoch=float(s['time_s'][0]);s['time_s']-=epoch
    r['target_time_s']-=epoch
    for key in ('desired_time_s','hold_until_s','stop_requested_time_s','reference_quiet_time_s','landing_trigger_s'):
        if r['state'][key] is not None:r['state'][key]-=epoch
    for key in ('swing','landing'):
        if r['state'][key] is not None:r['state'][key]['start_s']-=epoch
    q=np.asarray(r['q_ref']);v=np.asarray(r['v_ref']);zero=np.zeros_like(q)
    c=dict(target_position_rad=q,target_velocity_rad_s=v,reference_position_rad=q,
        reference_velocity_rad_s=v,residual_position_rad=zero,residual_velocity_rad_s=zero,residual_goal_rad=zero)
    body_v=s['velocity_body_mps'][0]
    return serial(dict(episode_id='fixture-episode',step_index=0,time_s=0.,
        measurement_source=source,reference_source='virtual_desired_motion_not_prescribed_physics_pose',
        wave_source_sha256=WAVE_SHA256,residual_source_sha256=RESIDUAL_SHA256,
        joint_names_runtime=list(r['joint_names_runtime']),measurement=s,reference=r,controller=c,
        requested_twist=[.005,0.,0.],world_up_vector=[0.,0.,1.],
        contact_valid=np.ones(6,bool),contact_age_s=np.zeros(6),
        critic_true_navigation_velocity=[-body_v[1],body_v[0],body_v[2]]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'fixtures.json')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve frozen fixtures; choose a fresh --out path')
    if hashlib.sha256((WAVE/'wave_reference.py').read_bytes()).hexdigest()!=WAVE_SHA256:
        raise ValueError('Source changed; do not silently regenerate fixtures')
    p=ActualPrefix();standing=packet(p.snapshot(199),p.r.reset(p.snapshot(199)),'simulator_truth_instrumented')
    p=ActualPrefix();p.replay();f=LandedFixture(p)
    landing=packet(f.snapshot(),p.out,'synthetic_fixture')
    payload=dict(standing=standing,landing=landing,provenance={
        'standing':'Actual reference002 completed standing sample at t=4.0 with wave002 reset; time origin shifted to encoder episode zero.',
        'landing':'Real reference002 prefix replayed through wave002, followed by exactly one explicitly synthetic measured landed fixture; not observed physics.',
        'reference002_trace_sha256':hashlib.sha256((ROOT/'tmp/reference_physics_results_002/run/wave/trace.npz').read_bytes()).hexdigest(),
        'world_and_kinematics_unchanged_by_epoch_shift':True})
    args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
