"""CPU rigid-link inertia/coordinate study; no motor dynamics or controller selected."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

MODEL_SHA='7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881'

def rotation(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def frames(model,q,order):
 T={'body':np.eye(4)};axes={};origins={};ancestors={'body':[]}
 pending=list(model['joints'])
 while pending:
  ready=[j for j in pending if j['parent'] in T]
  if not ready:raise ValueError('Disconnected model')
  for j in ready:
   F=np.eye(4);F[:3,:3]=rotation(j['quaternion_xyzw']);F[:3,3]=j['xyz']
   B=T[j['parent']]@F;a=q[order.index(j['name'])]
   origins[j['name']]=B[:3,3];axes[j['name']]=B[:3,2]
   Z=np.eye(4);Z[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
   T[j['child']]=B@Z;ancestors[j['child']]=ancestors[j['parent']]+[j['name']];pending.remove(j)
 return T,axes,origins,ancestors

def calculate(model,q=None,order=None,gravity=np.array([0.,0.,-9.81])):
 order=order or [j['name']for j in model['joints']]
 if len(order)!=18 or set(order)!={j['name']for j in model['joints']}:raise ValueError('Wrong joint names')
 q=np.zeros(18)if q is None else np.asarray(q,dtype=float)
 if q.shape!=(18,)or not np.isfinite(q).all():raise ValueError('Invalid q')
 T,axes,origins,ancestors=frames(model,q,order)
 M=np.zeros((18,18));hold=np.zeros(18);U=0.;details={}
 for link in model['links']:
  n=link['name'];R=T[n][:3,:3];p=R@link['com']+T[n][:3,3];Jv=np.zeros((3,18));Jw=np.zeros((3,18))
  for j in ancestors[n]:
   k=order.index(j);Jv[:,k]=np.cross(axes[j],p-origins[j]);Jw[:,k]=axes[j]
  I=R@np.asarray(link['inertia'])@R.T
  M+=link['mass']*(Jv.T@Jv)+Jw.T@I@Jw
  hold-=Jv.T@(link['mass']*gravity);U-=link['mass']*gravity@p
  details[n]={'com':p,'rotation':R,'Jv':Jv,'Jw':Jw}
 return {'M':M,'hold':hold,'potential':U,'details':details,'frames':T,'order':order}

def report(model):
 r=calculate(model);M=r['M'];eig=np.linalg.eigvalsh(M);effective=1/np.diag(np.linalg.inv(M))
 return {'scope':'Rigid-link fixed-root zero-pose CPU calculation; no identified motor/reflected inertia/friction/contact or selected gains',
   'model_sha256':MODEL_SHA,'joint_names':r['order'],'q_rad':[0.]*18,
   'mass_matrix_kg_m2':M.tolist(),'eigenvalues_kg_m2':eig.tolist(),
   'locked_other_joint_inertia_kg_m2':np.diag(M).tolist(),
   'free_other_joint_effective_inertia_kg_m2':effective.tolist(),
   'suspended_gravity_compensation_nm':r['hold'].tolist(),
   'NOT_ground_support_stance_torques':True,
   'formulas':{'simulation_PD_candidate_only':'Kp=I_eff*omega_n^2; Kd=2*zeta*I_eff*omega_n. omega_n/zeta are declared numerical design choices, not fitted hardware gains.',
     'coupling':'M_jj is locked-others inertia;1/(M^-1)_jj is unforced-other-joints effective inertia. Neither includes unknown gearbox/motor armature.'}}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert hashlib.sha256(a.model.read_bytes()).hexdigest()==MODEL_SHA
 if a.output.exists():raise SystemExit('Preserve prior report; choose a new output')
 a.output.write_text(json.dumps(report(json.loads(a.model.read_text())),indent=2)+'\n')
