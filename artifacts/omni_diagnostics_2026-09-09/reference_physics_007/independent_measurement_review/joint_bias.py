"""Exact16s joint position/velocity diagnostics for every matched joint; no gate substitution."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(run):
    phase=run/'standing'
    with np.load(phase/'trace.npz') as z:t={k:z[k] for k in z.files}
    with np.load(phase/'physics_substeps.npz') as z:d={k:z[k] for k in z.files}
    assert np.array_equal(t['joint_names'],d['joint_names'])
    assert np.array_equal(t['joint_velocity_rad_s'],d['joint_velocity_rad_s'][8::8])
    q=t['joint_position_rad'].astype(float);v=d['joint_velocity_rad_s'].astype(float)
    displacement=q[-1]-q[199];interval=v[1600:8001]
    integrals={'400Hz_trapezoid':.5*(interval[:-1]+interval[1:]).sum(0)*.0025,
               '400Hz_right':interval[1:].sum(0)*.0025,
               '50Hz_right':t['joint_velocity_rad_s'][200:].astype(float).sum(0)*.02}
    rows=[]
    for e in range(32):
        for j,name in enumerate(t['joint_names']):
            selected=q[200:,e,j];fd=np.diff(selected)/.02
            rows.append({'env':e,'joint':str(name),'world_root_at_start_m':t['position_world_m'][199,e].tolist(),
                         'joint_position_displacement_rad':float(displacement[e,j]),
                         'joint_position_range_rad':float(np.ptp(selected)),
                         'reported50Hz_joint_velocity_RMS_rad_s':float(np.sqrt(np.mean(t['joint_velocity_rad_s'][200:,e,j].astype(float)**2))),
                         'position_50Hz_difference_RMS_rad_s':float(np.sqrt(np.mean(fd**2))),
                         'reported_velocity_integrals_rad':{name:float(a[e,j]) for name,a in integrals.items()},
                         'position_minus_velocity_integral_rad':{name:float(displacement[e,j]-a[e,j]) for name,a in integrals.items()}})
    worst={k:max(rows,key=lambda r:abs(r['position_minus_velocity_integral_rad'][k])) for k in integrals}
    return {'interval_control_boundaries':[200,1000],'duration_s':16.,'raw_sha256':sha(phase/'physics_substeps.npz'),'trace_sha256':sha(phase/'trace.npz'),'rows':rows,'largest_abs_bias_rows':worst,'scope':'Diagnostic only; finite differences do not replace reported velocity RMS gates or rule out sub-control movement.'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--candidate-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r={'005':analyze(a.baseline_run),'007':analyze(a.candidate_run)}
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({run:{key:row['position_minus_velocity_integral_rad'][key] for key,row in value['largest_abs_bias_rows'].items()} for run,value in r.items()},indent=2))
if __name__=='__main__':main()
