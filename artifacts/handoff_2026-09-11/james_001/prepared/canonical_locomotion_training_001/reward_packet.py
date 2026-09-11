"""Proposed CPU eight-substep packing; native acquisition/physical gates stay external."""
import numpy as np
from oracles.frames import rotations_xyzw,body_velocity_at_root_origin,native_to_navigation,world_to_native_body
from oracles.objective import reward,material_point_tangent_velocity
LEGS=('lf','lm','lr','rf','rm','rr')

def rigid(pose):
    p=np.asarray(pose,float)
    if p.shape[-1]!=7 or not np.isfinite(p).all():raise ValueError('Bad link pose')
    # Check the unchanged native quaternion tolerance before numeric normalization.
    flat=p.reshape(-1,7);rotations_xyzw(flat[:,3:])
    q=flat[:,3:]/np.linalg.norm(flat[:,3:],axis=1,keepdims=True)
    out=np.broadcast_to(np.eye(4),(len(flat),4,4)).copy()
    out[:,:3,:3]=rotations_xyzw(q);out[:,:3,3]=flat[:,:3]
    return out.reshape(p.shape[:-1]+(4,4))

def material_slip(previous_pose, current_pose, patches, active, *, body_names, classified_valid):
    """Max speed of nonzero-force toe patches on each admitted active foot.

    Caller must supply exact already-accepted source005 classifier output. This
    function does not normalize invalid raw normals into acceptance. A max is an
    explicit reward aggregation proposal, not a replacement support/force gate.
    """
    before=rigid(previous_pose);after=rigid(current_pose);a=np.asarray(active)
    if before.shape!=after.shape or after.ndim!=4 or a.shape!=(len(after),6)or a.dtype!=bool:
        raise ValueError('Wrong link/contact packet dimensions')
    if len(body_names)!=after.shape[1] or len(set(body_names))!=len(body_names):raise ValueError('Named body order missing/duplicated')
    if classified_valid is not True:raise ValueError('Original raw classifier acceptance required')
    index={v:i for i,v in enumerate(body_names)};speed=np.zeros(a.shape);seen=np.zeros(a.shape,bool)
    if not set(l+'_tibia' for l in LEGS)<=set(index):raise ValueError('Six named tibia links required')
    for patch in patches:
        e=patch['env'];body=patch['body'];f=float(patch['normal_force_n']);p=np.asarray(patch['point_world_m'],float);normal=np.asarray(patch['normal_world'],float)
        if type(e)is not int or not 0<=e<len(after)or body not in index or p.shape!=(3,)or normal.shape!=(3,)or not np.isfinite(np.r_[f,p,normal,patch['separation_m']]).all():
            raise ValueError('Malformed classified patch')
        inactive=patch['inactive_zero_normal']
        if type(inactive)is not bool:raise ValueError('Explicit inactive classification required')
        if inactive:
            if not(f==0 and np.all(normal==0)and patch['separation_m']==0):raise ValueError('Invalid inactive tuple')
            continue
        # Raw norm was accepted by source005. Reward uses a unit tangent, not a tighter gate.
        norm=np.linalg.norm(normal)
        if not norm>0:raise ValueError('Noninactive patch has zero normal')
        if patch['category']!='toe' or f==0:continue
        if body not in [l+'_tibia'for l in LEGS]:raise ValueError('Toe classification on wrong body')
        leg=LEGS.index(body[:-6])
        if not a[e,leg]:continue
        tangent=material_point_tangent_velocity(p,normal/norm,before[e,index[body]],after[e,index[body]],.0025)
        speed[e,leg]=max(speed[e,leg],float(np.linalg.norm(tangent)));seen[e,leg]=True
    if np.any(a & ~seen):raise ValueError('Active support lacks its classified force/point evidence')
    return speed

def pack_reward(before, samples, *, command, previous_held, emitted, requested, previous_delta, root_com_local, slip, slip_valid):
    """All eight rows retained; returns old objective001 mathematics with explicit provenance."""
    if len(samples)!=8:raise ValueError('Eight complete native samples required')
    n=len(command);initial=int(before['explicit_counter']);time=float(before['time_s'])
    emitted=np.asarray(emitted);previous_held=np.asarray(previous_held)
    if emitted.shape!=(n,18)or previous_held.shape!=(n,18)or not np.isfinite(emitted).all()or not np.isfinite(previous_held).all():raise ValueError('Bad actual held target')
    if np.any(abs(emitted.astype(float)-previous_held.astype(float))>.040):raise ValueError('Actual emitted target violates unchanged limiter')
    values={k:[]for k in ['linear','angular','gravity','computed','applied','active']}
    for i,row in enumerate(samples):
        if int(row['explicit_counter'])!=initial+i+1 or abs(float(row['time_s'])-(time+(i+1)*.0025))>1e-9 or int(row['substep_index'])!=i:
            raise ValueError('Substep clock/counter mismatch')
        for key in ['contact_valid','interval_valid']:
            valid=np.asarray(row[key])
            if valid.shape!=(n,)or valid.dtype!=bool or not valid.all():raise ValueError('Invalid native interval/contact evidence')
        if not np.array_equal(row['joint_target_rad'],emitted):raise ValueError('Substep used another held target')
        pose=np.asarray(row['root_pose_xyzw'],float);rotations_xyzw(pose[:,3:])
        pose=pose.copy();pose[:,3:]/=np.linalg.norm(pose[:,3:],axis=1,keepdims=True)
        linear,angular=body_velocity_at_root_origin(row['root_com_velocity'],pose,root_com_local)
        values['linear'].append(native_to_navigation(linear));values['angular'].append(native_to_navigation(angular))
        values['gravity'].append(native_to_navigation(world_to_native_body(np.tile([0.,0.,-1.],(n,1)),pose[:,3:])))
        for key,raw in [('computed','computed_torque_nm'),('applied','applied_torque_nm'),('active','distal_contact')]:values[key].append(np.asarray(row[raw]))
        # Terminal attribution is handled by a separate failure path; this no-reset integration
        # must not launder an earlier failure through a later clean endpoint.
        if np.asarray(row['terminated']).any()or np.asarray(row['truncated']).any():raise ValueError('Preserve failed prefix; abort no-auto-reset proposal')
    sv=np.asarray(slip_valid)
    if sv.shape!=(n,8)or sv.dtype!=bool or not sv.all():raise ValueError('No zero-filled missing slip evidence')
    last=samples[-1];delta=emitted.astype(float)-previous_held.astype(float)
    result=reward(command,np.stack(values['linear'],1),np.stack(values['angular'],1),np.stack(values['gravity'],1),
        np.stack(values['computed'],1),np.stack(values['applied'],1),delta,previous_delta,np.asarray(requested)-emitted,
        (np.asarray(last['joint_position_rad'],float)-np.asarray(before['joint_position_rad'],float))/.02,
        slip,np.stack(values['active'],1),np.ones(n,bool),np.zeros(n,bool),np.zeros(n,bool))
    result['packing_schema']='canonical_moving_reward_packet_proposal_v1'
    result['counter_interval']=[initial,initial+8];result['native_admission']=False
    return result

def pack_material_slip(before, samples, patch_packets, *, body_names):
    """Proposed read-only accessor ABI: copied complete post-classification packets.

    Source005 does not yet expose this API. Eight exact {sequence,
    explicit_counter,classifier_accepted,patches} records must be captured inside
    that source's original classifier call, not reconstructed from net forces.
    """
    if len(samples)!=8 or len(patch_packets)!=8:raise ValueError('Missing complete patch interval')
    previous=before['link_pose_xyzw'];values=[]
    for row,packet in zip(samples,patch_packets):
        if packet['sequence']!=int(row['sequence'])or packet['explicit_counter']!=int(row['explicit_counter']):
            raise ValueError('Stale or mismatched patch/pose clock')
        values.append(material_slip(previous,row['link_pose_xyzw'],packet['patches'],row['distal_contact'],
                                   body_names=body_names,classified_valid=packet['classifier_accepted']))
        previous=row['link_pose_xyzw']
    return np.stack(values,axis=1)

def neutral_interval_context(previous_control, final_control):
    """Initial reward memory comes from the measured constant-target neutral prefix.

    Does not score/admit that prefix; the unchanged standing scorer does. Never
    invent zero motion from missing poses/targets or reuse a failed native session.
    """
    if int(final_control['explicit_counter'])-int(previous_control['explicit_counter'])!=8:
        raise ValueError('Adjacent prelude controls required')
    if abs(float(final_control['time_s'])-float(previous_control['time_s'])-.02)>1e-9:
        raise ValueError('Prelude control clock mismatch')
    held=np.asarray(final_control['joint_target_rad'],float);old=np.asarray(previous_control['joint_target_rad'],float)
    if held.ndim!=2 or held.shape[1]!=18 or old.shape!=held.shape or not np.isfinite(held).all()or not np.array_equal(held,old):
        raise ValueError('Final neutral controls did not hold the same actual target')
    poses=np.asarray(final_control['link_pose_xyzw']);rigid(poses)
    return {'previous_held':held.copy(),'previous_delta':held-old,'pre_hold_link_pose_xyzw':poses.copy(),
            'native_counter':int(final_control['explicit_counter']),'native_time_s':float(final_control['time_s']),
            'physical_admission':False}
