"""Device-resident port of the frozen pre-reset telemetry reader.

No Isaac import and no guessed SDK API. It reads the same installed properties
as physics_telemetry_oracle.py. Contact age/validity and done flags are explicit
caller inputs: this module cannot silently invent a fresh sensor timestamp.
"""
import torch
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent


def tensor(value):
    value=value.torch if hasattr(value,'torch') else value
    if not isinstance(value,torch.Tensor):raise ValueError('SDK field must expose a Torch device tensor')
    return value.detach().clone()


def rotation_xyzw(q):
    norm=q.norm(dim=-1)
    valid=torch.isfinite(q).all(-1)&((norm-1).abs()<=1e-4)
    safe=q/norm.clamp_min(1e-30)[...,None]
    x,y,z,w=safe.unbind(-1)
    R=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)
    return R,valid


class DeviceTelemetry:
    """Bind exact named layout once; capture one complete batch before resets.

    The producer must call the original done predicate once and supply those
    exact flags, as in the frozen pre-reset hook. This adapter itself does not
    modify env.step, robot pose, sensor settings, or controller state.
    """
    def __init__(self,env,layout,reference_points_local_m):
        self.names=tuple(layout['joint_names_runtime'])
        urdf=HERE/'reference/oracle/geometry/f050_t060.urdf'
        constants=HERE/'reference/kernel/geometry_constants.json'
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c':raise ValueError('Wrong C URDF binding')
        if hashlib.sha256(constants.read_bytes()).hexdigest()!='38848208bee962704da39bd517c09821bc1481749758e5ce9ee0c5778300622e':raise ValueError('Wrong named geometry binding')
        root=ET.fromstring(urdf.read_bytes());children={j.attrib['name']:j.find('child').attrib['link'] for j in root.findall('joint')}
        leg_names=tuple(json.loads(constants.read_text())['names_leg_major'])
        self.bodies=tuple(env._robot.body_names)
        expected_bodies={l.attrib['name'] for l in root.findall('link')}
        if len(self.bodies)!=19 or len(set(self.bodies))!=19 or set(self.bodies)!=expected_bodies:raise ValueError('Exact19 C body names required')
        if set(self.names)!=set(leg_names):raise ValueError('Exact18 C joint names required')
        feet=tuple(children[leg_names[i]] for i in range(2,18,3))
        expected={'foot_body_ids':tuple(self.bodies.index(name) for name in feet), 'foot_link_names':feet,
                  'joint_runtime_to_leg_major':tuple(self.names.index(name) for name in leg_names)}
        for key,value in expected.items():
            if tuple(layout[key])!=value:raise ValueError('Incorrect named map: '+key)
        self.sensor_names=tuple((name,) for name in feet)
        if len(self.names)!=18 or len(set(self.names))!=18:raise ValueError('18 unique named joints required')
        if tuple(env._robot.joint_names)!=self.names:raise ValueError('Observed joint order differs from bound layout')
        if tuple(tuple(s.body_names) for s in env._feet_contact_sensors)!=tuple((n,) for n in layout['foot_link_names']):
            raise ValueError('Six named single-tibia contact sensors required')
        q=tensor(env._robot.data.joint_pos)
        self.device=q.device;self.n=q.shape[0];self.dtype=torch.float64
        if q.shape!=(self.n,18):raise ValueError('18 measured coordinates required')
        self.foot_ids=torch.tensor(layout['foot_body_ids'],device=self.device,dtype=torch.int64)
        self.leg_ids=torch.tensor(layout['joint_runtime_to_leg_major'],device=self.device,dtype=torch.int64)
        if self.foot_ids.shape!=(6,) or self.leg_ids.shape!=(18,):raise ValueError('Complete named foot/joint maps required')
        self.toes=torch.as_tensor(reference_points_local_m,device=self.device,dtype=self.dtype).clone()
        if self.toes.shape!=(6,3) or not torch.isfinite(self.toes).all():raise ValueError('Six finite local toe reference points required')
        self.position_is_float32=env._robot.data.root_pos_w.torch.dtype==torch.float32 if hasattr(env._robot.data.root_pos_w,'torch') else env._robot.data.root_pos_w.dtype==torch.float32

    def read(self,value,shape,name,boolean=False):
        value=tensor(value)
        if value.device!=self.device or value.shape!=shape:raise ValueError('Wrong device/shape: '+name)
        if boolean:
            if value.dtype!=torch.bool:raise ValueError('Boolean classification required: '+name)
            return value
        if not value.is_floating_point():raise ValueError('Floating SDK field required: '+name)
        return value.to(self.dtype)

    def capture(self,env,*,time_s,terminated,truncated,contact_valid,contact_age_s):
        n=self.n
        if tuple(env._robot.joint_names)!=self.names:raise ValueError('Joint order changed after static binding')
        if tuple(env._robot.body_names)!=self.bodies:raise ValueError('Body order changed after static binding')
        if tuple(tuple(s.body_names) for s in env._feet_contact_sensors)!=self.sensor_names:raise ValueError('Contact sensor order changed after binding')
        d=env._robot.data
        time=self.read(time_s,(n,),'sample time')
        term=self.read(terminated,(n,),'terminated',True);trunc=self.read(truncated,(n,),'truncated',True)
        valid_contact=self.read(contact_valid,(n,6),'contact estimate validity',True)
        age=self.read(contact_age_s,(n,6),'contact estimate age')
        p=self.read(d.root_pos_w,(n,3),'root link position')
        q=self.read(d.root_quat_w,(n,4),'root XYZW quaternion')
        R,rotation_valid=rotation_xyzw(q)
        bodyp=self.read(d.body_link_pos_w,(n,19,3),'body link pos').index_select(1,self.foot_ids)
        bodyq=self.read(d.body_link_quat_w,(n,19,4),'body link quat').index_select(1,self.foot_ids)
        bodyR,foot_rotation_valid=rotation_xyzw(bodyq)
        offset=(bodyR@self.toes[...,None]).squeeze(-1)
        foot=bodyp+offset
        bodyv=self.read(d.body_link_lin_vel_w,(n,19,3),'body link lin_vel').index_select(1,self.foot_ids)
        bodyw=self.read(d.body_link_ang_vel_w,(n,19,3),'body link ang_vel').index_select(1,self.foot_ids)
        footv=bodyv+torch.cross(bodyw,offset,dim=-1)
        rawcontact=torch.stack([self.read(s.data.contact_pos_w,(n,1,1,3),'contact point')[:,0,0] for s in env._feet_contact_sensors],1)
        point_valid=torch.isfinite(rawcontact).all(-1)
        contact=torch.where(point_valid[:,:,None],rawcontact,0.)
        normal=torch.stack([self.read(s.data.force_matrix_w,(n,1,1,3),'normal force')[:,0,0] for s in env._feet_contact_sensors],1)
        distal,shaft,slip,_,reaction,friction=env._get_foot_contact_state(include_ground_wrench=True)
        distal=self.read(distal,(n,6),'distal classification',True);shaft=self.read(shaft,(n,6),'shaft classification',True)
        histories=[tensor(env._coxa_contact_sensor.data.net_forces_w_history),
                   *(tensor(s.data.net_forces_w_history) for s in env._femur_contact_sensors),
                   tensor(env._base_contact_sensor.data.net_forces_w_history)]
        history_finite=torch.ones(n,device=self.device,dtype=torch.bool)
        if len(histories)!=8:raise ValueError('Six femur sensors required')
        for index,h in enumerate(histories):
            expected_bodies=6 if index==0 else 1
            if h.device!=self.device or h.ndim!=4 or h.shape[0]!=n or h.shape[1]<1 or h.shape[2:]!=(expected_bodies,3):raise ValueError('Bad force-history shape/device')
            history_finite &= torch.isfinite(h).reshape(n,-1).all(-1)
        coxa=histories[0].norm(dim=-1).amax(1)>1.
        femur=torch.cat([h.norm(dim=-1).amax(1)>1. for h in histories[1:7]],1)
        base=histories[7].norm(dim=-1).amax(1)[:,0]
        raw_link_v=self.read(d.root_link_lin_vel_w,(n,3),'root link velocity')
        gyro_w=self.read(d.root_link_ang_vel_w,(n,3),'root link angular velocity')
        snapshot=dict(time_s=time,position_world_m=p,quaternion_world_xyzw=q,
            quaternion_world_wxyz=q[:,[3,0,1,2]],rotation_world_from_body=R,
            velocity_world_mps=raw_link_v,gyro_world_rad_s=gyro_w,
            com_velocity_world_mps=self.read(d.root_lin_vel_w,(n,3),'SDK COM alias velocity'),
            velocity_body_mps=self.read(d.root_lin_vel_b,(n,3),'SDK COM body velocity'),
            root_link_velocity_body_mps=(raw_link_v[:,None,:]@R).squeeze(1),
            gyro_body_rad_s=self.read(d.root_ang_vel_b,(n,3),'SDK body angular velocity'),
            projected_gravity_body=self.read(d.projected_gravity_b,(n,3),'projected gravity'),
            joint_position_rad=self.read(d.joint_pos,(n,18),'joint position'),
            joint_velocity_rad_s=self.read(d.joint_vel,(n,18),'SDK reported joint velocity'),
            joint_target_rad=self.read(env._processed_actions,(n,18),'emitted target'),
            soft_joint_pos_limits_rad=self.read(d.soft_joint_pos_limits,(n,18,2),'joint soft limits'),
            computed_torque_nm=self.read(d.computed_torque,(n,18),'requested torque'),
            applied_torque_nm=self.read(d.applied_torque,(n,18),'applied torque'),
            reference_point_world_m=foot,reference_point_velocity_world_mps=footv,
            contact_point_world_m=contact,contact_point_world_m_raw=rawcontact,contact_point_valid=point_valid,
            normal_force_world_n=normal,reaction_force_world_n=self.read(reaction,(n,6,3),'reaction force'),
            friction_force_xy_n=self.read(friction,(n,6),'friction'),
            distal_contact=distal,shaft_contact=shaft,distal_contact_slip_mps=self.read(slip,(n,6),'slip'),
            coxa_contact=self.read(coxa,(n,6),'coxa contact',True),femur_contact=self.read(femur,(n,6),'femur contact',True),base_contact=self.read(base>1.,(n,),'base contact',True),terminated=term,truncated=trunc,
            contact_valid=valid_contact,contact_age_s=age,
            position_is_float32=torch.full((n,),self.position_is_float32,device=self.device,dtype=torch.bool))
        controller=getattr(env,'reference_residual_controller',None)
        if controller is None:raise ValueError('Bound residual002 controller state is required')
        snapshot['executable_target_velocity_rad_s']=self.read(controller.reference_velocity+controller.residual_velocity,(n,18),'executed target velocity')
        valid=history_finite&rotation_valid&foot_rotation_valid.all(-1)&torch.isfinite(time)&(time>=0)
        valid &= ((~(distal|shaft))|point_valid).all(-1)
        valid &= valid_contact.all(-1)&torch.isfinite(age).all(-1)&(age>=0).all(-1)&(age<=.04+1e-9).all(-1)
        for key,value in snapshot.items():
            if key!='contact_point_world_m_raw':valid &= torch.isfinite(value).reshape(n,-1).all(-1)
        snapshot['measurement_valid']=valid
        return {'measurement':snapshot,'measurement_source':'simulator_raw_instrumented',
                'velocity_semantics':'raw SDK rates retained; physical position-rate consistency unverified',
                'reported_velocity_physical_consistency_verified':False,'deployment_qualified':False,
                'joint_names_runtime':self.names,'contact_freshness_supplied_by_caller':True}
