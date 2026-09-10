"""Repeatable target-space screen only; no contact, actuator or gait claims."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import torch
from velocity_action import JointTargetVelocity,VelocityActionConfig
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
def main():
    package=ROOT/'robot/hexapod_mkii_length_study'
    mapping=json.loads((package/'manifest.json').read_text())['link_joint_mapping']
    urdf=package/'urdf/f050_t060.urdf';tree=ET.parse(urdf)
    names=tuple(j.get('name') for j in tree.getroot().findall('joint'))
    bounds={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in tree.getroot().findall('joint')}
    lower={n:sum(bounds[n])/2-.475*(bounds[n][1]-bounds[n][0]) for n in names}
    upper={n:sum(bounds[n])/2+.475*(bounds[n][1]-bounds[n][0]) for n in names}
    pose={mapping[leg]['joints'][part]:np.deg2rad(deg) for leg in mapping for part,deg in [('coxa',0),('femur',40),('tibia',120)]}
    q=torch.tensor([[pose[n] for n in names]],dtype=torch.float32).expand(32,18).clone()
    rows=[]
    for profile in ['diagnostic_003','formal_004']:
        for std in [0.,.02,.05,.1]:
            torch.manual_seed(7057)
            c=JointTargetVelocity(names,lower,upper,32,VelocityActionConfig(profile),dtype=torch.float32);c.reset(q)
            positions=[];velocities=[];accelerations=[];limited=[]
            for _ in range(500):
                result=c.step(torch.randn_like(q)*std)
                positions.append(result.target_position_rad.numpy());velocities.append(result.target_velocity_rad_s.numpy())
                accelerations.append(result.target_acceleration_rad_s2.numpy());limited.append(result.acceleration_limited.numpy())
            p=np.stack(positions);v=np.stack(velocities);a=np.stack(accelerations)
            rows.append({'profile':profile,'sampled_normalized_action_std':std,'duration_s':10,'replicas':32,
                'target_velocity_rms_rad_s':float(np.sqrt(np.mean(v*v))),
                'max_target_step_rad':float(np.abs(v*.02).max()),'max_target_acceleration_rad_s2':float(np.abs(a).max()),
                'final_target_drift_rms_rad':float(np.sqrt(np.mean((p[-1]-q.numpy())**2))),
                'final_target_drift_abs_max_rad':float(np.abs(p[-1]-q.numpy()).max()),
                'acceleration_intervention_fraction':float(np.stack(limited).mean()),'physics_admitted':False})
    report={'complete':True,'kind':'CPU_target_state_noise_screen_not_physical_admission','stage2_complete':False,
            'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),'runtime_order_status':'URDF joint names used for CPU screen; simulator order must be observed at import',
            'joint_names':names,'rows':rows,'notes':['Profiles are separate software feasibility cases, not claimed loaded motor speed limits.',
                'IID velocity noise integrates into target drift. Zero action holds exactly; learned quiet feedback is not yet admitted.',
                'Acceleration is measured between executable 20ms knots. No continuous jerk, body tracking or contact guarantee.']}
    (HERE/'cpu_noise_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
