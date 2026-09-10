"""Compare actual400Hz angle increments with reported rates; no gate replacement."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def analyze(run):
    source=run/'physics_substeps.npz'; controls=run/'trace.npz'
    with np.load(source) as p,np.load(controls) as c:
        n=len(p['time_s']);assert n==8001
        np.testing.assert_array_equal(p['relative_physics_index'],np.arange(n))
        np.testing.assert_allclose(np.diff(p['time_s']),.0025,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(p['joint_names'],c['joint_names'])
        for key in ('joint_position_rad','joint_velocity_rad_s'):
            np.testing.assert_array_equal(p[key][8::8],c[key])
        q=p['joint_position_rad'][1600:].astype(np.float64); v=p['joint_velocity_rad_s'][1600:].astype(np.float64)
        dt=np.diff(p['time_s'][1600:]);delta=q[-1]-q[0]
        integral={name:np.einsum('t,tej->ej',dt,values) for name,values in [('left',v[:-1]),('right',v[1:]),('trapezoid',.5*(v[:-1]+v[1:]))]}
        interval_rate=np.diff(q,axis=0)/dt[:,None,None]
        sdk_interval=.5*(v[:-1]+v[1:]);sdk_rms=np.sqrt(np.mean(v[1:]**2,axis=0));difference_rms=np.sqrt(np.mean(interval_rate**2,axis=0))
        per=[]
        for e in range(32):
            for j,name in enumerate(p['joint_names'].tolist()):
                per.append(dict(environment=e,joint=name,world_root_position_at_start_m=p['root_link_position_world_m'][1600,e].astype(float).tolist(),actual_angle_delta_rad=float(delta[e,j]),actual_angle_range_rad=float(np.ptp(q[:,e,j])),integrated_reported_rate_rad={k:float(a[e,j]) for k,a in integral.items()},integral_minus_angle_delta_rad={k:float(a[e,j]-delta[e,j]) for k,a in integral.items()},reported_rate_rms_rad_s=float(sdk_rms[e,j]),angle_difference_interval_rate_rms_rad_s=float(difference_rms[e,j]),interval_rate_disagreement_rms_rad_s=float(np.sqrt(np.mean((sdk_interval[:,e,j]-interval_rate[:,e,j])**2)))))
        worst=max(per,key=lambda r:abs(r['integral_minus_angle_delta_rad']['trapezoid']))
        return dict(scope='Completed009 undisturbed standing only; actual angle evidence, no scoring substitution or native-cause claim',substep_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),control_sha256=hashlib.sha256(controls.read_bytes()).hexdigest(),control_boundaries=[200,1000],duration_s=float(dt.sum()),substep_samples=n,all_400Hz_control_endpoints_exact=True,rate_from_angles_semantics='interval-averaged angle differences, not an independently sensed instantaneous velocity',pairs=len(per),pairs_abs_integral_difference_gt_001rad=sum(abs(r['integral_minus_angle_delta_rad']['trapezoid'])>.01 for r in per),worst_pair=worst,per_environment_joint=per,stage2_complete=False,velocity_measurement_qualified=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--standing',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=analyze(args.standing);args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='per_environment_joint'},indent=2))
