"""Read-only classification of saved first-rejected knots, not a new rollout."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));from study import IdealFixture,verify
from serial_geometry import tensor

def main():
    verify();report=json.loads((ROOT/'report.json').read_text());fixture=IdealFixture();g=fixture.g
    legs=('lf','lm','lr','rf','rm','rr');rows=[]
    for row in report['stage_one']:
        if row['failure'] is None:continue
        fail=row['failure'];candidate=fail['bound_candidate'];q=np.asarray(candidate['q_candidate'])[fixture.to_leg].reshape(6,3)
        foot,jac,_=g.fk(tensor(q));ik=g.ik(foot[None]);sigma=np.linalg.svd(jac.numpy(),compute_uv=False)[:,-1]
        crossing=[]
        for key,budget in [('v_candidate',1.75),('a_candidate',6.)]:
            data=np.asarray(candidate[key]);index=int(np.argmax(np.abs(data)));leg=legs[fixture.to_leg.index(index)//3]
            crossing.append(dict(channel=key,budget=budget,magnitude=float(abs(data[index])),joint_name=fixture.names[index],leg=leg,role='active_swing' if leg==fail['current_leg'] else 'planted_stance',exceeds=bool(abs(data[index])>budget+1e-5)))
        latest=[e for e in row['events'] if e['kind']=='synthetic_liftoff'][-1]
        phase=(fail['reference_time_s']-latest['time_s'])/row['swing_s']
        rows.append(dict(id=row['id'],swing_s=row['swing_s'],requested_speed_mps=row['requested_speed_mps'],confirmed_synthetic_touchdowns=row['confirmed_synthetic_touchdowns'],failure_mode=fail['mode'],failure_motion_phase='landing' if 'landing' in fail['mode'] else 'swing_unloading' if fail['mode']=='unloading' else 'swing',active_leg=fail['current_leg'],fraction_of_swing=phase,time_after_command_start_s=fail['reference_time_s']-2.,first_crossings=crossing,minimum_joint_margin_rad=candidate['minimum_joint_margin_rad'],required_joint_margin_rad=.02,minimum_jacobian_sigma_m=float(sigma.min()),minimum_sigma_leg=legs[int(np.argmin(sigma))],required_jacobian_sigma_m=.002,IK_valid_at_candidate=bool(ik['valid'].all()),FK_IK_error_m=float(ik['error_m'].max()),endpoint_stride_m=latest['stride_m'],classification='Executable swing path velocity/acceleration budget rejection, with valid nonsingular IK and retained joint margin. No measured motor/dynamic limit inferred.'))
    result=dict(scope='Classification of saved first rejected candidate only; no extra trajectory, geometry change or physics',rows=rows,failure_channels_sorted=sorted({x['channel'] for r in rows for x in r['first_crossings'] if x['exceeds']}),all_first_crossings_active_swing=all(x['role']=='active_swing' for r in rows for x in r['first_crossings'] if x['exceeds']),all_candidates_IK_valid=all(r['IK_valid_at_candidate'] for r in rows),minimum_rejected_knot_sigma_m=min(r['minimum_jacobian_sigma_m'] for r in rows),minimum_rejected_knot_joint_margin_rad=min(r['minimum_joint_margin_rad'] for r in rows))
    (ROOT/'failure_classification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:print(r['swing_s'],r['requested_speed_mps'],r['failure_mode'],r['active_leg'],round(r['fraction_of_swing'],3),[(x['joint_name'],x['leg'],round(x['magnitude'],4)) for x in r['first_crossings'] if x['exceeds']],round(r['minimum_jacobian_sigma_m'],5))
if __name__=='__main__':main()
