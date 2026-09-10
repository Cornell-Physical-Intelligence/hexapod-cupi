"""Exact rigid-link floating-base inertia at zero pose; bounded local pulse planning."""
from pathlib import Path
import json,numpy as np
from inertia_probe import calculate

def skew(v):
 x,y,z=v
 return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def floating_mass(model,order=None):
 r=calculate(model,order=order);M=np.zeros((24,24))
 for link in model['links']:
  d=r['details'][link['name']];p=d['com'];R=d['rotation']
  V=np.c_[np.eye(3),-skew(p),d['Jv']]
  W=np.c_[np.zeros((3,3)),np.eye(3),d['Jw']]
  M+=link['mass']*V.T@V+W.T@(R@np.asarray(link['inertia'])@R.T)@W
 return M,r

def prediction(model,amplitude=.005,pulse_steps=8,coast_steps=8,dt=.0025):
 M,r=floating_mass(model);cases=[]
 for j,name in enumerate(r['order']):
  u=np.zeros(24);u[6+j]=amplitude;a=np.linalg.solve(M,u)
  v=a*(pulse_steps*dt)
  # Constant-M semi-implicit substep integration, including motion during pulse.
  x=a*dt*dt*(pulse_steps*(pulse_steps+1)/2+pulse_steps*coast_steps)
  cases.append({'joint_name':name,'acceleration_generalized':a.tolist(),'end_velocity_generalized':v.tolist(),'end_displacement_generalized':x.tolist(),
                'maximum_all_joint_displacement_rad':float(max(abs(x[6:]))),'maximum_all_joint_speed_rad_s':float(max(abs(v[6:]))),
                'applied_positive_joint_response_rad':float(x[6+j])})
 return {'amplitude_nm':amplitude,'pulse_steps':pulse_steps,'coast_steps':coast_steps,'dt_s':dt,'cases':cases,
   'maximum_all_cases_joint_excursion_rad':max(x['maximum_all_joint_displacement_rad']for x in cases),
   'maximum_all_cases_joint_speed_rad_s':max(x['maximum_all_joint_speed_rad_s']for x in cases)}

if __name__=='__main__':
 D=Path(__file__).parent;ROOT=D.resolve().parents[1]
 model=json.loads((ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())
 M,r=floating_mass(model);old=prediction(model,.02,8,40);new=prediction(model)
 report={'scope':'Full floating rigid-link local zero-pose prediction only, no selfcontact/constraint/numerical/native proof','generalized_order':['root_vx','root_vy','root_vz','root_wx','root_wy','root_wz']+r['order'],
  'mass_matrix':M.tolist(),'minimum_eigenvalue':float(min(np.linalg.eigvalsh(M))),'reviewed_draft001':old,'selected002':new,
  'unchanged_diagnostic_bounds':{'joint_excursion_rad':.05,'joint_speed_rad_s':2},
  'selected_excursion_margin_ratio':.05/new['maximum_all_cases_joint_excursion_rad']}
 out=D/'floating_pulse_report.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
