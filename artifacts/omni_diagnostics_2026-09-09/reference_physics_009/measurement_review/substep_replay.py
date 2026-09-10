"""Independently inspect immutable raw physics measurements; no new acceptance gate.

Reads only. Writes a fresh review directory, never modifies source/raw results.
No simulator dependencies or GPU calls. NumPy is required.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def metrics(d,trace,start,end):
    assert 0 <= start < end <= len(trace['position_world_m'])
    lo,hi=start*8,end*8
    result={'start_control_boundary':start,'end_control_boundary':end,'duration_s':(end-start)*.02}
    # Control trace i equals substep boundary i+1. Use the exact start orientation.
    if start:
        axis=-trace['rotation_world_from_body'][start-1,:,:,1].astype(float)
    else:
        q=d['root_link_quaternion_world_xyzw'][0].astype(float)
        q=q/np.linalg.norm(q,axis=-1,keepdims=True);x,y,z,w=q.T
        axis=-np.stack((2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)),axis=-1)
    axis[:,2]=0;axis/=np.linalg.norm(axis,axis=-1,keepdims=True)
    result['initial_forward_axis_world']=axis.tolist()
    for frame in ('link','com'):
        p=d[f'root_{frame}_position_world_m'][lo:hi+1].astype(float)
        v=d[f'root_{frame}_velocity_world_mps'][lo:hi+1].astype(float)
        displacement=p[-1]-p[0]
        right=v[1:].sum(0)*.0025;left=v[:-1].sum(0)*.0025
        last=v[8::8].sum(0)*.02
        estimates={'400Hz_right':right,'400Hz_left':left,'400Hz_trapezoid':.5*(right+left),'50Hz_last_substep_right':last}
        residual=np.diff(p,axis=0)-v[1:]*.0025
        rn=np.linalg.norm(residual,axis=-1)
        top=np.argsort(rn.ravel())[-10:][::-1]
        extrema=[]
        for flat in top:
            step,env=np.unravel_index(flat,rn.shape);idx=lo+step+1
            extrema.append({'relative_physics_index':int(idx),'control_index':int(d['control_index'][idx]),'substep_index':int(d['substep_index'][idx]),'env':int(env),'position_minus_right_velocity_integral_m':residual[step,env].tolist(),'norm_m':float(rn[step,env])})
        shifted={}
        # Same position increments, shifted reported velocities; diagnostics only.
        globalp=d[f'root_{frame}_position_world_m'].astype(float)
        globalv=d[f'root_{frame}_velocity_world_mps'].astype(float)
        for shift in (-2,-1,0,1,2):
            first=max(lo+1,1-shift);lastindex=min(hi,len(globalv)-1-shift)
            idx=np.arange(first,lastindex+1)
            residual_shift=(globalp[idx]-globalp[idx-1])-globalv[idx+shift]*.0025
            shifted[str(shift)]={'substeps':len(idx),'increment_residual_rms_m':float(np.sqrt(np.mean(residual_shift**2))),'integral_difference_norm_m':np.linalg.norm(residual_shift.sum(0),axis=-1).tolist()}
        result[frame]={'world_displacement_m':displacement.tolist(),'forward_displacement_m':np.sum(displacement*axis,axis=-1).tolist(),
            'velocity_integrals':{k:{'world_m':v.tolist(),'forward_m':np.sum(v*axis,axis=-1).tolist(),'displacement_difference_norm_m':np.linalg.norm(displacement-v,axis=-1).tolist()} for k,v in estimates.items()},
            'substep_increment_difference_norm_quantiles_m':dict(zip(['p50','p95','p99','max'],np.quantile(rn,[.5,.95,.99,1]).tolist())),
            'mean_world_increment_residual_by_substep_m':residual.reshape((-1,8)+residual.shape[1:]).mean(axis=(0,2)).tolist(),
            'largest_increment_residuals':extrema,'velocity_index_shift_diagnostics':shifted}
    for kind in ('computed','applied'):
        values=np.abs(d[f'{kind}_torque_nm'][lo+1:hi+1].astype(float));control=np.abs(trace[f'{kind}_torque_nm'][start:end].astype(float))
        maximum=values.max(axis=(0,1));lastmax=control.max(axis=(0,1));counts=(values>1.6).sum(axis=(0,1))
        result[kind+'_torque']={'all_substeps_max_nm':float(values.max()),'last_substeps_max_nm':float(control.max()),
            'per_joint':[{'name':str(name),'substep_max_nm':float(maximum[j]),'last_substep_max_nm':float(lastmax[j]),'substep_count_above_1p6':int(counts[j]),'substep_total':int(values.shape[0]*values.shape[1])} for j,name in enumerate(d['joint_names'])]}
    return result


def review(phase):
    raw=phase/'physics_substeps.npz';ordinary=phase/'trace.npz'
    d=load(raw);t=load(ordinary);n=len(t['position_world_m'])
    if len(d['time_s'])!=n*8+1:raise ValueError('Partial control requires separate explicit review')
    assert np.array_equal(d['relative_physics_index'],np.arange(n*8+1))
    assert np.allclose(np.diff(d['time_s']),.0025,rtol=0,atol=1e-10)
    matches={}
    for a,b in [('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('root_link_velocity_world_mps','velocity_world_mps'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:
        matches[a]=bool(np.array_equal(d[a][8::8],t[b]));assert matches[a],a
    assert np.array_equal(d['joint_names'],t['joint_names'])
    predicted=d['root_com_velocity_world_mps']+np.cross(d['root_angular_velocity_world_rad_s'],d['root_link_position_world_m']-d['root_com_position_world_m'])
    relation=d['root_link_velocity_world_mps']-predicted
    result={'phase':phase.name,'raw_sha256':sha(raw),'control_trace_sha256':sha(ordinary),'controls':n,'raw_samples':len(d['time_s']),'raw_matches_ordinary_last_substeps':matches,
        'link_COM_velocity_identity_max_abs_error_mps':float(np.abs(relation).max()),
        'link_COM_velocity_identity_formula':'v_link = v_COM + omega cross (p_link-p_COM)',
        'old_control_gate_unchanged':True,'acceptance_verdict':None,'scope':'Independent measurement diagnosis, not admission or a substituted acceptance metric'}
    if n>200:result['post_settle']=metrics(d,t,200,n)
    if phase.name=='wave' and n>200:result['moving_prefix']=metrics(d,t,200,min(n,1400))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh review output required')
    if a.run.resolve()==a.output.resolve() or a.run.resolve() in a.output.resolve().parents:raise ValueError('Review output must be outside raw results')
    reports=[review(a.run/phase) for phase in ('standing','wave') if (a.run/phase/'physics_substeps.npz').exists()]
    if not reports:raise FileNotFoundError('No actual substep data yet')
    a.output.mkdir(parents=True)
    (a.output/'review.json').write_text(json.dumps({'reports':reports},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'phases':[r['phase'] for r in reports],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
