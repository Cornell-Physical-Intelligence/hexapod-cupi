"""Independent numerical comparison of cold initial/final retention evidence."""
from pathlib import Path
import argparse
import json
import numpy as np
from audit import load, require, substeps, clocks, sha, CONSUMER

def summarize(path, *, legacy_fixture=False):
    path=Path(path);d=load(path/'trace.npz');s=load(path/'physics_substeps.npz');c=load(path/'sensor_clocks.npz')
    r=json.loads((path/'retention.json').read_text());state=json.loads((path/'state.json').read_text())
    expected='e9a1d6d687504fd1323d8addbcb962c3025745a258f71b9f076879778eedaa0b' if legacy_fixture else CONSUMER
    require(state['identity']['consumer_freeze_sha256']==expected,'Wrong exact consumer identity')
    checkpoint_key='checkpoint_sha256' if legacy_fixture else 'input_checkpoint_sha256'
    require(d['joint_position_rad'].shape==(2400,1,18),'Complete single-replica forward/stop trace required')
    _,physical=substeps(d,s);physical.update(clocks(c,np.zeros((2400,1),bool)))
    require(not d['terminated'].any() and not d['truncated'].any(),'Retention includes a native reset')
    quaternion=d['quaternion_world_xyzw'][199,0]
    require(np.isfinite(quaternion).all() and abs(np.linalg.norm(quaternion)-1)<=1e-4,'Invalid actual orientation')
    x,y,z,w=quaternion/np.linalg.norm(quaternion)
    forward=np.array([2*(w*z-x*y),-1+2*(x*x+z*z),0.]);forward/=np.linalg.norm(forward)
    delta=d['position_world_m'][1399,0]-d['position_world_m'][199,0]
    velocities=d['velocity_world_mps'][199:1400,0]
    integrated=np.sum((velocities[:-1]+velocities[1:])*.01,axis=0)
    measured=dict(measured_forward_mps=float(-d['velocity_body_mps'][200:1400,0,1].mean()),
        displacement_m=float(delta@forward),world_position_integral_discrepancy_m=float(np.linalg.norm(delta-integrated)))
    require(np.isclose(measured['measured_forward_mps'],r['measured_forward_mps'],rtol=0,atol=1e-12),'Reported forward speed differs')
    require(np.isclose(measured['displacement_m'],r['independent_progress']['measured_forward_displacement_m'],rtol=0,atol=1e-12),'Reported displacement differs')
    require(np.isclose(measured['world_position_integral_discrepancy_m'],r['independent_progress']['displacement_integral_difference_m'],rtol=0,atol=1e-12),'Reported old integration gate differs')
    quiet=r.get('final_quiet');require(quiet is not None,'No complete final quiet window')
    start=quiet['start_control_index'];require(2400-start>=500,'Less than ten seconds of quiet')
    require(not d['requested_command'][start:].any(),'Reported quiet window contains requested motion')
    q=d['joint_position_rad'][start:,0];target=d['joint_target_rad'][start:,0];v=d['joint_velocity_rad_s'][start:,0]
    measured.update(quiet_duration_s=(2400-start)*.02,
        quiet_raw_joint_rms_max_rad_s=float(np.sqrt(np.mean(v*v,axis=0)).max()),
        quiet_joint_range_max_rad=float(np.ptp(q,axis=0).max()),
        quiet_target_step_p95_max_rad=float(np.quantile(np.abs(np.diff(target,axis=0)),.95,axis=0).max()),
        # Interval averaged motion evidence; neither native rate nor old gate is replaced.
        quiet_interval_angle_rate_rms_max_rad_s=float(np.sqrt(np.mean((np.diff(q,axis=0)/.02)**2,axis=0)).max()))
    for ours,theirs in [('quiet_raw_joint_rms_max_rad_s','max_joint_velocity_rms_rad_s'),
                       ('quiet_joint_range_max_rad','max_joint_position_range_rad'),
                       ('quiet_target_step_p95_max_rad','max_target_step_abs_p95_rad_per_20ms')]:
        require(np.isclose(measured[ours],quiet['per_environment'][0][theirs],rtol=0,atol=1e-12),'Reported quiet numeric value differs '+theirs)
    return {'reported_retention_passed':r['passed'],'source_identity':state['identity'],
        'input_checkpoint_sha256':state[checkpoint_key],'measured':measured,'physical_substep_audit':physical,
        'complete_measured_legs_reported':r['completed_legs'],
        'source_claim':'No new admission: existing gate result retained; this independently replays motion/quiet/substep values',
        'files_sha256':{name:sha(path/name) for name in ('trace.npz','physics_substeps.npz','sensor_clocks.npz','retention.json','state.json')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'Comparison output must be new')
    first,last=summarize(a.initial),summarize(a.final)
    require(first['source_identity']==last['source_identity'],'Initial and final source/physical identity differ')
    result={'initial':first,'final':last,'final_minus_initial':{k:last['measured'][k]-v for k,v in first['measured'].items()},
        'Stage2_complete':False,'automatic_continuation_allowed':False,'new_admission_issued':False}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
