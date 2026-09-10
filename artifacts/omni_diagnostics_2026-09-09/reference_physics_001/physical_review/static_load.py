"""Independent ideal quasistatic point-contact torque feasibility, not physics."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp/reference_physics_adapter_001/source_001/tools'
sys.path.insert(0,str(SOURCE))
from serial_geometry import SerialGeometry,tensor
from wave_math import MassGeometry
G=SerialGeometry(); M=MassGeometry(G)

def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def geometry(q):
    foot,J,T=G.fk(tensor(q))
    foot=foot.numpy();J=J.numpy();T=np.stack([v.numpy() for v in T],1)
    com=(T[:,:,:3,:3]@M.link_com[:,:,:,None])[:,:,:,0]+T[:,:,:3,3]
    totalcom=(M.body_mass*M.body_com+(M.link_mass[:,:,None]*com).sum((0,1)))/M.mass
    tau=np.zeros((6,3))
    for leg in range(6):
        for joint in range(3):
            axis=T[leg,joint,:3,:3]@G.axes[leg,joint].numpy()
            pivot=T[leg,joint,:3,3]
            for downstream in range(joint,3):
                tau[leg,joint]+=axis@np.cross(com[leg,downstream]-pivot,[0,0,-9.81*M.link_mass[leg,downstream]])
    return foot,J,tau,totalcom

def solve(q,support,mu=1.):
    foot,J,taug,com=geometry(q)
    # All 18 contact force coordinates plus minimax torque t. Excluded feet fixed zero.
    C=np.zeros(19);C[-1]=1
    Aeq=np.zeros((6,19));beq=np.r_[0,0,M.mass*9.81,np.cross(com,[0,0,M.mass*9.81])]
    Aub=[];bub=[]
    for leg in range(6):
        sl=slice(3*leg,3*leg+3)
        Aeq[:3,sl]=np.eye(3);Aeq[3:,sl]=skew(foot[leg])
        for joint in range(3):
            for sign in (-1,1):
                row=np.zeros(19);row[sl]=sign*J[leg,:,joint];row[-1]=-1
                Aub.append(row);bub.append(-sign*taug[leg,joint])
        # Inscribed octagonal friction cone approximation: |tangent| <= mu Fz.
        for a in np.arange(8)*np.pi/4:
            row=np.zeros(19);row[sl]=[np.cos(a),np.sin(a),-mu*np.cos(np.pi/8)]
            Aub.append(row);bub.append(0.)
    bounds=[]
    for leg in range(6):bounds.extend([(None,None),(None,None),(0,None)] if leg in support else [(0,0)]*3)
    bounds.append((0,None))
    r=linprog(C,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
    if not r.success:return dict(feasible=False,message=r.message)
    f=r.x[:18].reshape(6,3);tau=-(taug+np.einsum('lij,li->lj',J,f))
    return dict(feasible=True,minimax_torque_nm=float(r.x[-1]),force_n=f.tolist(),torque_nm=tau.tolist(),com_body_m=com.tolist(),support_margin_m=M.support_margin(foot,support,com),mass_kg=M.mass,feet_body_m=foot.tolist())

if __name__=='__main__':
    legs=['lf','lm','lr','rf','rm','rr'];q=G.q0.numpy()
    rows={}
    for mu in [0.,.5,1.]:
        rows[str(mu)]={'all_six':solve(q,list(range(6)),mu)}
        for i,leg in enumerate(legs):rows[str(mu)]['without_'+leg]=solve(q,[j for j in range(6) if i!=j],mu)
    out=dict(scope='Optimistic quasistatic force allocation at nominal frozen stance. Not actual measured pose, collision, drive or contact proof.',q_leg_major_rad=q.tolist(),results=rows)
    (Path(__file__).parent/'static_load.json').write_text(json.dumps(out,indent=2)+'\n')
    for mu,cases in rows.items():
        print('mu',mu,[(name,round(r.get('minimax_torque_nm',float('nan')),6)) for name,r in cases.items()])
