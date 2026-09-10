"""Ideal static force allocation for the frozen C serial model, not dynamics."""
from pathlib import Path
import itertools,json,math,hashlib
import numpy as np
from scipy.optimize import linprog
from serial_geometry import SerialGeometry,tensor
from wave_math import MassGeometry
LEGS=('lf','lm','lr','rf','rm','rr')
G=SerialGeometry();M=MassGeometry(G)


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def geometry(q):
    foot,J,T=G.fk(tensor(q));foot=foot.numpy();J=J.numpy();T=np.stack([x.numpy() for x in T],1)
    com=(T[:,:,:3,:3]@M.link_com[:,:,:,None])[:,:,:,0]+T[:,:,:3,3]
    total_com=(M.body_mass*M.body_com+(M.link_mass[:,:,None]*com).sum((0,1)))/M.mass
    tau_gravity=np.zeros((6,3))
    for leg in range(6):
        for joint in range(3):
            axis=T[leg,joint,:3,:3]@G.axes[leg,joint].numpy()
            pivot=T[leg,joint,:3,3]
            for downstream in range(joint,3):
                force=np.array([0.,0.,-9.81*M.link_mass[leg,downstream]])
                tau_gravity[leg,joint]+=axis@np.cross(com[leg,downstream]-pivot,force)
    return dict(foot=foot,J=J,link_com=com,total_com=total_com,tau_gravity=tau_gravity,
        potential_energy_j=9.81*(M.body_mass*M.body_com[2]+(M.link_mass*com[:,:,2]).sum()))


def solve(q,support,mu=0.,cone='outer',facets=16,minimum_support_normal_n=1.):
    """Outer friction polygon gives lower bound on ideal circular-cone optimum.

    Inner polygon gives a feasible exact-Coulomb witness and upper bound on that
    same ideal optimum. Both omit real actuator/control/dynamic limitations.
    """
    if cone not in ('outer','inner') or facets<4 or mu<0 or minimum_support_normal_n<0:raise ValueError('Declared positive friction model required')
    geo=geometry(q);foot,J,taug,com=(geo[k] for k in ('foot','J','tau_gravity','total_com'))
    support=tuple(support)
    if not 1<=len(support)<=6 or len(set(support))!=len(support) or min(support)<0 or max(support)>5:raise ValueError('Distinct named support indices required')
    objective=np.zeros(19);objective[-1]=1
    Aeq=np.zeros((6,19));beq=np.r_[0.,0.,M.mass*9.81,np.cross(com,[0,0,M.mass*9.81])]
    Aub=[];bub=[]
    friction_scale=1. if cone=='outer' else math.cos(math.pi/facets)
    for leg in range(6):
        sl=slice(3*leg,3*leg+3);Aeq[:3,sl]=np.eye(3);Aeq[3:,sl]=skew(foot[leg])
        for joint in range(3):
            for sign in (-1,1):
                row=np.zeros(19);row[sl]=sign*J[leg,:,joint];row[-1]=-1
                Aub.append(row);bub.append(-sign*taug[leg,joint])
        for a in np.arange(facets)*2*np.pi/facets:
            row=np.zeros(19);row[sl]=[math.cos(a),math.sin(a),-mu*friction_scale]
            Aub.append(row);bub.append(0.)
    bounds=[]
    for leg in range(6):bounds.extend([(None,None),(None,None),(minimum_support_normal_n,None)] if leg in support else [(0,0)]*3)
    bounds.append((0,None))
    result=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
    base=dict(support_indices=list(support),support_legs=[LEGS[i] for i in support],mu=mu,cone=cone,facets=facets,
        support_margin_m=M.support_margin(foot,support,com),minimum_support_normal_n=minimum_support_normal_n,feasible=bool(result.success),solver_status=int(result.status),solver_message=result.message)
    if not result.success:return base
    forces=result.x[:18].reshape(6,3);tau=-(taug+np.einsum('lij,li->lj',J,forces))
    ratios=np.linalg.norm(forces[:,:2],axis=1)/np.maximum(forces[:,2],1e-10)
    assert np.isfinite(result.x).all() and np.isfinite(tau).all()
    achieved=float(abs(tau).max())
    normres=float(np.max(abs(np.einsum('ij,j->i',Aeq,result.x)-beq)))
    ineqres=float(max(0.,np.max(np.einsum('ij,j->i',np.array(Aub),result.x)-np.array(bub))))
    assert normres<1e-6 and ineqres<1e-6 and abs(achieved-result.fun)<1e-6
    if cone=='inner':assert max(ratios)<=mu+1e-7
    idx=np.unravel_index(abs(tau).argmax(),tau.shape)
    return dict(**base,minimax_motor_torque_nm=float(result.fun),headroom_to1p6_nm=1.6-float(result.fun),
        contact_force_n=forces.tolist(),actuator_torque_nm=tau.tolist(),gravity_joint_torque_nm=taug.tolist(),
        maximum_friction_ratio=float(max(ratios)),circular_cone_witness=bool(max(ratios)<=mu+1e-7),
        equilibrium_residual_max=normres,linear_constraint_violation_max=ineqres,
        peak_leg=LEGS[idx[0]],peak_joint=G.names[idx[0]][idx[1]])


def run():
    q=G.q0.numpy();cases=[]
    for count in (3,4,5,6):
        for support in itertools.combinations(range(6),count):
            name='+'.join(LEGS[i] for i in support)
            rows=[]
            for mu in (0.,.3,.6,1.):
                outer=solve(q,support,mu,'outer');inner=solve(q,support,mu,'inner')
                if outer['feasible'] and inner['feasible']:assert outer['minimax_motor_torque_nm']<=inner['minimax_motor_torque_nm']+1e-7
                rows.append(dict(mu=mu,outer_lower_bound=outer,inner_feasible_witness=inner))
            cases.append(dict(name=name,count=count,alternating_tripod=support in ((0,2,4),(1,3,5)),rows=rows))
    geo=geometry(q)
    report=dict(scope='Nominal C serial fixed-pose optimistic static force allocation; not dynamic or hardware proof',
        mass_kg=float(M.mass),stance_q_rad=q.tolist(),joint_names=G.names,foot_positions_body_m=geo['foot'].tolist(),
        total_com_body_m=geo['total_com'].tolist(),gravity_joint_torque_nm=geo['tau_gravity'].tolist(),
        source_urdf_sha256=G.urdf_sha,cone_facets=16,assumptions=['All six legs keep the same nominal configuration; support subset alone changes.','Exact serial geometry and CAD-derived link mass/COM, body gravity and every downstream leg gravity moment included.','Point contacts on horizontal rigid ground; each declared support has at least1N normal force, other feet exactly0; no adhesion or contact moment.','Free optimal joint/contact force allocation; no PD controller, compliance, motor dynamics, acceleration or swing inertia.','No stance, mass, length, friction, torque or physical admission gate is changed.'],cases=cases)
    p=Path(__file__).parent;(p/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    thresholds=[]
    for support in ((0,2,4),(1,3,5)):
        row={'support_legs':[LEGS[i] for i in support],'minimum_support_normal_n':1.}
        for cone in ('outer','inner'):
            lo,hi=0.,.6
            for _ in range(25):
                mid=(lo+hi)/2;result=solve(q,support,mid,cone)
                if result['feasible'] and result['minimax_motor_torque_nm']<=1.6:hi=mid
                else:lo=mid
            row[cone+'_minimum_mu_bracket']=[lo,hi]
        thresholds.append(row)
    (p/'friction_thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    for case in cases:
        if case['alternating_tripod'] or case['count']==6:
            print(case['name'],[(r['mu'],r['outer_lower_bound'].get('minimax_motor_torque_nm'),r['inner_feasible_witness'].get('minimax_motor_torque_nm')) for r in case['rows']])
    return report

if __name__=='__main__':run()
