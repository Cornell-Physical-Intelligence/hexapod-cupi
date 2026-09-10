"""Independent read-only review of the frozen sensor-agent CPU prototype."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'tmp/omni_reference_prototype'
sys.path.insert(0,str(SOURCE))
from reference import SerialGeometry,TwistReference,Config,FilterState,tensor,nav_to_body,DTYPE
OUT=Path(__file__).parent
torch.set_num_threads(1)
source_sha=hashlib.sha256((SOURCE/'reference.py').read_bytes()).hexdigest()
g=SerialGeometry();r=TwistReference(g,Config(stance_travel_m=.06));f=FilterState(r)
segments=[('start',2,[0,0,0]),('forward',5,[.1,0,0]),('reverse',5,[-.1,0,0]),
 ('left',5,[0,.1,0]),('left_arc',5,[0,.1,.2]),('right_arc',5,[.1,0,-.2]),('yaw',5,[0,0,.3]),('stop',12,[0,0,0])]
rows=[];lastq=None
for name,duration,target in segments:
 invalid=0;delta_max=0.;qvel_max=0.;slip_max=0.;at_cap=0;n=0
 for step in range(round(duration/.02)):
  s=f.step([target],.02);q=s['q_checked'];invalid+=int((~s['valid']).sum());qvel_max=max(qvel_max,float(s['q_velocity'].abs().max()))
  if lastq is not None:
   delta=(q-lastq).abs();delta_max=max(delta_max,float(delta.max()));at_cap+=int((delta>.03).sum());n+=delta.numel()
  lastq=q
  p=torch.remainder(f.phase[:,None]+r.offsets,1);mask=p<r.config.duty;xy=s['feet'][...,:2]
  body=nav_to_body(f.command)[:,None,:]+f.command[:,None,2,None]*torch.stack((-xy[...,1],xy[...,0]),-1)
  slip=(s['foot_velocity'][...,:2]+body).norm(dim=-1)[mask]
  slip_max=max(slip_max,float(slip.max()))
 rows.append({'name':name,'invalid_leg_steps':invalid,'max_reference_step_rad_per_20ms':delta_max,
              'fraction_reference_steps_above_0p03':at_cap/max(n,1),'max_qdot_rad_s':qvel_max,'max_stance_reference_slip_mps':slip_max})
# Necessary stance-only velocity feasibility at the neutral pose; not full-cycle/dynamic admission.
feet,J,_=g.fk(g.q0);Jp=torch.stack((-feet[:,:2][:,1],feet[:,:2][:,0]),-1)
commands=[('forward',[.2,0,0]),('left',[0,.2,0]),('turn_left',[0,0,.4]),('forward_left_arc',[.2,0,.4]),('strafe_left_arc',[0,.2,.4])]
envelope=[]
for name,c in commands:
 c=tensor([c]);horizontal=-nav_to_body(c)[:,None,:]-c[:,None,2,None]*Jp
 vel=torch.cat((horizontal,torch.zeros(1,6,1,dtype=DTYPE)),-1)
 dq=torch.linalg.solve(J,vel[...,None]).squeeze(-1);rate=float(dq.abs().max())
 envelope.append({'command':name,'twist':c[0].tolist(),'max_neutral_stance_joint_speed_rad_s':rate,
  'scale_to_1p2_rad_s_necessary_not_sufficient':min(1.,1.2/rate),
  'per_joint_qdot':dict(zip([n for group in g.names for n in group],dq.flatten().tolist()))})
report={'source_sha256':source_sha,'configuration':r.config.__dict__,'transition':rows,
 'neutral_stance_envelope':envelope,'stage2_complete':False,
 'notes':['Kinematic-only, exact named serial C URDF. No torque, contacts, support, terrain or actuator admission.',
          '20% residual velocity reserve is an illustrative design budget, not a qualified setting.',
          '60mm transition sequence independently measured; do not substitute100mm transition metrics.']}
assert source_sha==hashlib.sha256((SOURCE/'reference.py').read_bytes()).hexdigest()
(OUT/'cpu_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('configuration','transition')},indent=2))
print('ENVELOPE',[(x['command'],x['max_neutral_stance_joint_speed_rad_s']) for x in envelope])
