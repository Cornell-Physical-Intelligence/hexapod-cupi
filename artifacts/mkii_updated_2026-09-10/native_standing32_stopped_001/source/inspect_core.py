"""Read-only USD/native checks. No actuator, pose, or effort setters."""
import math
import numpy as np


def finite_array(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    if shape is not None and a.shape != shape:
        raise ValueError(f'Wrong shape {a.shape}, expected {shape}')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite native value')
    return a


def rotation(q):
    x,y,z,w = finite_array(q, (4,))
    if abs(x*x+y*y+z*z+w*w-1.) > 2e-5:
        raise ValueError('Quaternion is not normalized XYZW')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def close(a, b, atol, label, rtol=0.):
    a,b=finite_array(a),finite_array(b)
    if a.shape != b.shape or not np.allclose(a,b,rtol=rtol,atol=atol):
        raise ValueError(f'{label} differs')


def usd_readback(stage):
    from pxr import UsdGeom, UsdPhysics
    bodies,joints,colliders,drives=[],[],[],[]
    def qxyzw(q): return [*map(float,q.GetImaginary()),float(q.GetReal())]
    for p in stage.Traverse():
        path=str(p.GetPath())
        if not path.startswith('/Robot/'): continue
        # Preserve authored schema tokens even offline where PhysxSchema is unavailable.
        op=p.GetMetadata('apiSchemas')
        apis=list(op.GetAppliedItems()) if op else []
        drives += [path+':'+a for a in apis if a.startswith('PhysicsDriveAPI')]
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            mass=UsdPhysics.MassAPI(p);rb=UsdPhysics.RigidBodyAPI(p)
            M=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
            bodies.append({'name':p.GetName(),'path':path,'mass':mass.GetMassAttr().Get(),
                           'com':list(mass.GetCenterOfMassAttr().Get()),
                           'principal_moments':list(mass.GetDiagonalInertiaAttr().Get()),
                           'principal_axes_xyzw':qxyzw(mass.GetPrincipalAxesAttr().Get()),
                           'world_transform':M.tolist(),
                           'kinematic':rb.GetKinematicEnabledAttr().Get(),
                           'enabled':rb.GetRigidBodyEnabledAttr().Get(),
                           'self_collisions':p.GetAttribute('physxArticulation:enabledSelfCollisions').Get(),
                           'filtered_pairs':list(map(str,p.GetRelationship('physics:filteredPairs').GetTargets())) if p.HasRelationship('physics:filteredPairs') else [],
                           'articulation_root':p.HasAPI(UsdPhysics.ArticulationRootAPI)})
        if p.IsA(UsdPhysics.Joint):
            j=UsdPhysics.RevoluteJoint(p)
            if not j: raise ValueError('Non-revolute joint present')
            joints.append({'name':p.GetName(),'path':path,'axis':j.GetAxisAttr().Get(),
                           'body0':list(map(str,j.GetBody0Rel().GetTargets())),
                           'body1':list(map(str,j.GetBody1Rel().GetTargets())),
                           'local_pos0':list(j.GetLocalPos0Attr().Get()),
                           'local_pos1':list(j.GetLocalPos1Attr().Get()),
                           'local_rot0_xyzw':qxyzw(j.GetLocalRot0Attr().Get()),
                           'local_rot1_xyzw':qxyzw(j.GetLocalRot1Attr().Get()),
                           'limits_rad':[math.radians(j.GetLowerLimitAttr().Get()),math.radians(j.GetUpperLimitAttr().Get())],
                           'collision_enabled':j.GetCollisionEnabledAttr().Get()})
        if p.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({'path':path,'approximation':UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get(),
                              'schemas':apis,'enabled':UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get(),
                              'sdf_resolution':p.GetAttribute('physxSDFMeshCollision:sdfResolution').Get()})
    return {'bodies':bodies,'joints':joints,'colliders':colliders,'drives':drives,
            'up_axis':str(UsdGeom.GetStageUpAxis(stage)), 'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),
            'kg_per_unit':UsdPhysics.GetStageKilogramsPerUnit(stage)}


def validate_usd(r, model):
    if r['up_axis']!='Z' or r['meters_per_unit']!=1 or r['kg_per_unit']!=1: raise ValueError('Wrong stage units/frame')
    if r['drives']: raise ValueError('Unexpected drive API')
    bodies={x['name']:x for x in r['bodies']};joints={x['name']:x for x in r['joints']}
    if len(bodies)!=19 or len(r['bodies'])!=19 or set(bodies)!={x['name']for x in model['links']}: raise ValueError('Wrong body set')
    if len(joints)!=18 or len(r['joints'])!=18 or set(joints)!={x['name']for x in model['joints']}: raise ValueError('Wrong joint set')
    if [x['name']for x in r['bodies']if x['articulation_root']]!=['body']: raise ValueError('Wrong articulation root')
    if bodies['body']['self_collisions'] is not True:raise ValueError('Self-collision disabled')
    for m in model['links']:
        a=bodies[m['name']]
        if a['kinematic'] or not a['enabled']: raise ValueError('Body not dynamic')
        close(a['mass'],m['mass'],2e-7,'USD mass')
        close(a['com'],m['com'],2e-7,'USD COM')
        R=rotation(a['principal_axes_xyzw']);I=R@np.diag(a['principal_moments'])@R.T
        close(I,m['inertia'],1e-8,'USD full inertia',rtol=2e-5)
    for m in model['joints']:
        a=joints[m['name']]
        if a['collision_enabled'] is not False:raise ValueError('Authored adjacent joint collision filter changed')
        if a['axis']!='Z' or a['body0']!=[bodies[m['parent']]['path']] or a['body1']!=[bodies[m['child']]['path']]: raise ValueError('Wrong joint graph/axis')
        close(a['local_pos0'],m['xyz'],2e-7,'Joint frame position')
        close(rotation(a['local_rot0_xyzw']),rotation(m['quaternion_xyzw']),3e-7,'Joint frame rotation')
        close(a['local_pos1'],[0,0,0],1e-8,'Child joint frame')
        close(rotation(a['local_rot1_xyzw']),np.eye(3),1e-8,'Child joint rotation')
        close(a['limits_rad'],[m['lower'],m['upper']],2e-7,'Joint limits')
    c=r['colliders']
    if len(c)!=153 or len({x['path']for x in c})!=153:raise ValueError('Wrong collider inventory')
    if any(x['approximation']!='sdf' or not x['enabled'] or 'PhysxSDFMeshCollisionAPI' not in x['schemas'] for x in c): raise ValueError('Not exact enabled SDF colliders')
    return True


def validate_native(r, model):
    bn,jn=r['body_names'],r['joint_names']
    if r['count']!=1 or r['fixed_base'] or len(bn)!=19 or len(set(bn))!=19 or set(bn)!={x['name']for x in model['links']}: raise ValueError('Wrong native bodies/articulation')
    if len(jn)!=18 or len(set(jn))!=18 or set(jn)!={x['name']for x in model['joints']}:raise ValueError('Wrong native DOFs')
    mass=finite_array(r['masses'],(1,19))[0];com=finite_array(r['coms'],(1,19,7))[0]
    inertia=finite_array(r['inertias'],(1,19,9))[0]
    limits=finite_array(r['limits'],(1,18,2))[0]
    for m in model['links']:
        i=bn.index(m['name']);close(mass[i],m['mass'],2e-6,'Native mass')
        close(com[i,:3],m['com'],2e-6,'Native COM')
        rotation(com[i,3:])
        # Native tensor API107.3: COM-referenced tensor in rigid-body-prim frame.
        # Preserve raw tensor and principal-axes quaternion; never rotate it twice.
        close(inertia[i].reshape(3,3,order='F'),m['inertia'],2e-7,'Native full inertia',rtol=2e-4)
    for m in model['joints']:close(limits[jn.index(m['name'])],[m['lower'],m['upper']],2e-6,'Native limits')
    close(finite_array(r['stiffness'],(1,18)),np.zeros((1,18)),0.,'Native stiffness')
    close(finite_array(r['damping'],(1,18)),np.zeros((1,18)),0.,'Native damping')
    return True


def validate_sdf(r, usd):
    paths=[x['path']for x in usd['colliders']]
    if r['count']!=153 or r['valid'] is not True or len(r['paths'])!=153 or set(r['paths'])!=set(paths):raise ValueError('Missing native SDF shape representation')
    if r.get('contract')!='canonical_native_sdf_initialization_v2':raise ValueError('Wrong native SDF initialization contract')
    if r.get('initialization_barrier')!={'sim_reset_returned':True,'physics_view_valid':True,'articulation_view_valid':True}:
        raise ValueError('Native initialization barrier incomplete')
    counter=r['legacy_task_counter']
    if counter.get('available') is True:
        if type(counter.get('pending'))is not int or counter['pending']!=0:raise ValueError('Reported cooking unfinished')
    elif counter.get('available')is not False or counter.get('pending')is not None:
        raise ValueError('Unknown cooking count must stay null')
    return True


def validate_samples(rows):
    if len(rows)!=9 or [x['explicit_step']for x in rows]!=list(range(9)): raise ValueError('Wrong sample sequence')
    counters=[r['explicit_step_counter']for r in rows]
    if any(type(n) is not int for n in counters) or counters!=list(range(counters[0],counters[0]+9)):
        raise ValueError('Native explicit-step counter cadence differs')
    for row in rows:
        for k,s in [('link_pose_xyzw',(1,19,7)),('link_com_velocity',(1,19,6)),('joint_position',(1,18)),('joint_velocity_sdk',(1,18))]:finite_array(row[k],s)
    return True


def validate_scene(r):
    if r['gravity_magnitude']!=0. or r['time_steps_per_second']!=400 or r['manager_dt']!=.0025:
        raise ValueError('Native scene gravity/dt differs from declared inspection')
    if r['root_self_collisions'] is not True:raise ValueError('Native stage self-collision changed')
    return True


def validate_native_frames(row, body_names, joint_names, model):
    """Compare actual joint scalars against actual link transforms, in observed name order."""
    pose=finite_array(row['link_pose_xyzw'],(1,19,7))[0]
    q=finite_array(row['joint_position'],(1,18))[0]
    actual={}
    for name,p in zip(body_names,pose):
        T=np.eye(4);T[:3,:3]=rotation(p[3:]);T[:3,3]=p[:3];actual[name]=T
    expected={'body':actual['body']}
    pending=list(model['joints'])
    while pending:
        available=[j for j in pending if j['parent'] in expected]
        if not available:raise ValueError('Disconnected expected joint graph')
        for j in available:
            F=np.eye(4);F[:3,:3]=rotation(j['quaternion_xyzw']);F[:3,3]=j['xyz']
            a=q[joint_names.index(j['name'])];Z=np.eye(4)
            Z[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            expected[j['child']]=expected[j['parent']]@F@Z
            close(actual[j['child']][:3,3],expected[j['child']][:3,3],5e-6,'Native link/joint frame position')
            close(actual[j['child']][:3,:3],expected[j['child']][:3,:3],5e-5,'Native link/joint frame rotation')
            pending.remove(j)
    return True
