"""Batched wave005 contact/reference FSM prototype. No physics or policy.

All dynamic state and per-row failure masks stay on the selected device. Fixed
joint/link and polynomial loops are allowed; there is no per-environment loop.
Source/schema validation is once per construction; dynamic invariants remain.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'parent/kernel'))
from tensor_kernel import TensorGeometry, finite_rows

LEGS=('lf','lm','lr','rf','rm','rr')
MODES=('hold','swing','awaiting_contact','contact_hold','stopping_reference_motion',
       'reference_quiet_hold','landing_blend','awaiting_landing_support','unloading')
ERRORS={0:'valid',1:'not reset',2:'malformed numeric measured state',3:'terminal or nonfoot contact',
 4:'fewer than five measured support contacts',5:'projected COM support margin insufficient',
 6:'actual body tracking error',7:'planted toe drift',8:'returned contact lacks flight/apex/descent',
 9:'returned contact speed',10:'landing endpoint exceeds12mm',11:'landing excursion exceeds12mm',
 12:'landing foot moved beyond contact region',13:'landing completion disagreement',
 14:'landing contact loss exceeds100ms',15:'liftoff not observed or touchdown missing',
 16:'next wave leg lacks initial contact',17:'IK invalid; no executable clipped target',
 18:'reference joint margin',19:'reference discrete rate budget',20:'time mismatch',
 21:'invalid requested command',22:'reset preload or measured FK invalid',23:'reset velocity or limits invalid',
 24:'reset episode identity is not fresh',25:'liftoff lacks measured2mm clearance by apex'}
FLOAT_FIELDS={'time':(), 'position':(3,), 'R0':(3,3), 'yaw':(), 'command':(3,), 'rate':(3,),
 'q':(18,), 'v':(18,), 'lower':(18,), 'upper':(18,), 'neutral':(6,3), 'anchors':(6,3),
 'measured_anchors':(6,3), 'preload_world':(6,3), 'preload_q':(6,3), 'hold_until':(),
 'requested':(3,), 'command_target':(3,), 'factor':(), 'flight_baseline_z':(), 'flight_peak_z':(),
 'landing_trigger':(), 'landing_origin':(3,), 'landing_preload':(3,), 'landing_correction':(),
 'landing_original_error':(), 'landing_excursion':(), 'stop_time':(), 'quiet_time':(),
 'sw_start':(), 'sw_duration':(), 'sw_coeff':(6,3), 'sw_hcoeff':(6,3), 'sw_end':(3,),
 'land_start':(), 'land_duration':(), 'land_coeff':(6,3), 'land_end':(3,),
 'support_margin':(), 'body_error':(), 'stance_drift':(6,), 'reference_points':(6,3),
 'a':(18,), 'unqualified_return_time':(), 'unqualified_lift':()}
BOOL_FIELDS=('ready','flight_seen','descent_seen','sw_active','land_active','flight_valid',
             'landing_metadata_valid','stop_valid','quiet_valid','position_float32','unqualified_return_valid')
INT_FIELDS=('episode','failure','mode','current_leg','order','flight_count','contact_count',
            'liftoffs','touchdowns','landing_gap','raw_force_free_samples','raw_force_free_runs',
            'unqualified_contact_returns','last_unqualified_run_samples')
SNAP_FLOAT={'time_s':(), 'position_world_m':(3,), 'quaternion_world_xyzw':(4,),
 'rotation_world_from_body':(3,3),'velocity_body_mps':(3,), 'gyro_body_rad_s':(3,),
 'joint_position_rad':(18,), 'joint_target_rad':(18,), 'reference_point_world_m':(6,3),
 'reference_point_velocity_world_mps':(6,3),'contact_point_world_m':(6,3)}
SNAP_BOOL={'contact_point_valid':(6,), 'distal_contact':(6,), 'shaft_contact':(6,),
 'coxa_contact':(6,), 'femur_contact':(6,), 'base_contact':(), 'terminated':(), 'truncated':(),
 'position_is_float32':()}


class BatchWave005:
    def __init__(self,joint_names_runtime,num_envs,*,device='cpu'):
        if type(num_envs) is not int or num_envs<1:raise ValueError('Positive replica count required')
        contract=json.loads((HERE/'source_contract.json').read_text())
        for name,sha in contract['source_files'].items():
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen input mismatch: '+name)
        self.identity={k:v for k,v in contract.items() if k!='source_files'}
        self.identity['implementation']='batched_qualified_contact_reference_prototype_v2'
        self.g=TensorGeometry(joint_names_runtime,binding='wave003_7mm',profile='formal_004',device=device)
        self.n=num_envs;self.device=self.g.device;self.dtype=self.g.dtype;self.names=tuple(joint_names_runtime)
        self.s={}
        with torch.inference_mode(False):
            for name,tail in FLOAT_FIELDS.items():self.s[name]=torch.zeros((num_envs,*tail),device=self.device,dtype=self.dtype)
            for name in BOOL_FIELDS:self.s[name]=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
            for name in INT_FIELDS:self.s[name]=torch.zeros(num_envs,device=self.device,dtype=torch.int64)
            self.s['episode'].fill_(-1);self.s['current_leg'].fill_(-1);self.s['failure'].fill_(1)
        self.order=torch.tensor([0,5,1,3,2,4],device=self.device)
        self.ids=torch.arange(num_envs,device=self.device)
        self.legs=torch.arange(6,device=self.device)[None]
        self.up=torch.tensor([0.,0.,1.],device=self.device,dtype=self.dtype).expand(num_envs,-1)
        self.inv=torch.linalg.inv(torch.tensor([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]],device=self.device,dtype=self.dtype))
        pairs=torch.combinations(torch.arange(6,device=self.device),r=2)
        self.edge_a,self.edge_b=pairs[:,0],pairs[:,1]
        self.sample_times=torch.linspace(0,1,51,device=self.device,dtype=self.dtype)

    @property
    def active(self):return self.s['ready'] & (self.s['failure']==0)

    def put(self,name,value,mask=None):
        old=self.s[name]; mask=self.active if mask is None else mask
        if not isinstance(value,torch.Tensor):value=torch.full_like(old,value)
        self.s[name].copy_(torch.where(mask.reshape(self.n,*([1]*(old.ndim-1))),value,old))

    def fail(self,condition,code):
        self.put('failure',code,self.active & condition)

    def _read(self,m):
        valid=torch.ones(self.n,device=self.device,dtype=torch.bool)
        for key,shape in SNAP_FLOAT.items():
            self.g.check(m[key],(self.n,*shape),key)
            if key!='contact_point_world_m':valid &= finite_rows(m[key])
        for key,shape in SNAP_BOOL.items():self.g.check(m[key],(self.n,*shape),key,torch.bool)
        contact=m['distal_contact'] & m['contact_point_valid']
        valid &= (torch.isfinite(m['contact_point_world_m']).all(-1) | ~contact).all(-1)
        x,y,z,w=m['quaternion_world_xyzw'].unbind(-1)
        R=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),
                       2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(self.n,3,3)
        valid &= (m['quaternion_world_xyzw'].norm(dim=-1)-1).abs()<=1e-4
        valid &= ((R-m['rotation_world_from_body']).abs() <= 2e-4+1e-5*m['rotation_world_from_body'].abs()).all((-1,-2))
        terminal=m['terminated']|m['truncated']|m['base_contact']|m['shaft_contact'].any(-1)|m['coxa_contact'].any(-1)|m['femur_contact'].any(-1)
        return contact,valid,terminal

    def R(self,yaw=None):
        yaw=self.s['yaw'] if yaw is None else yaw
        c,s=yaw.cos(),yaw.sin();zero=torch.zeros_like(c);one=torch.ones_like(c)
        rot=torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(self.n,3,3)
        return rot@self.s['R0']

    def _support(self,m,contact,exclude):
        support=contact & (self.legs!=exclude[:,None])
        points=torch.nan_to_num(m['contact_point_world_m'][...,:2])
        com=self.g.com(m['joint_position_rad'],m['position_world_m'],m['rotation_world_from_body'])
        a=points[:,self.edge_a];b=points[:,self.edge_b];edge=b-a
        d=points[:,None,:,:]-a[:,:,None,:]
        cross=edge[:,:,None,0]*d[:,:,:,1]-edge[:,:,None,1]*d[:,:,:,0]
        positive=torch.where(support[:,None,:],cross,torch.inf).amin(-1)>=-1e-12
        negative=torch.where(support[:,None,:],cross,-torch.inf).amax(-1)<=1e-12
        length=edge.norm(dim=-1)
        candidate=support[:,self.edge_a]&support[:,self.edge_b]&(length>1e-12)&(positive|negative)
        noncollinear=torch.where(support[:,None,:],cross.abs(),0.).amax((-1,-2))>1e-12
        delta=com['world_m'][:,None,:2]-a
        signed=(edge[:,:,0]*delta[:,:,1]-edge[:,:,1]*delta[:,:,0])/length.clamp_min(1e-30)
        signed=torch.where(positive,signed,-signed)
        margin=torch.where(candidate,signed,torch.inf).amin(-1)
        margin=torch.where(noncollinear & candidate.any(-1) & com['valid'],margin,-torch.inf)
        return margin,support.sum(-1)

    def _bounds(self,q,v,a):
        self.fail(((q<self.s['lower']+.02)|(q>self.s['upper']-.02)).any(-1),18)
        self.fail((v.abs().amax(-1)>1.75+1e-5)|(a.abs().amax(-1)>6.+1e-5)|~finite_rows(q)|~finite_rows(v)|~finite_rows(a),19)

    def reset(self,m,selected,episode):
        self.g.check(selected,(self.n,),'reset selected',torch.bool)
        self.g.check(episode,(self.n,),'episode',torch.int64)
        contact,valid,terminal=self._read(m)
        self.g.check(m['executable_target_velocity_rad_s'],(self.n,18),'reset executed velocity')
        self.g.check(m['soft_joint_pos_limits_rad'],(self.n,18,2),'reset soft limits')
        fresh=(episode>=0)&(episode>self.s['episode'])
        self.put('failure',0,selected);self.put('ready',True,selected)
        self.fail(selected & ~fresh,24)
        self.fail(selected & ~valid,2);self.fail(selected & terminal,3);self.fail(selected & (contact.sum(-1)<5),4)
        limits=m['soft_joint_pos_limits_rad'];target=m['joint_target_rad'];actual=m['joint_position_rad']
        lower=torch.maximum(limits[...,0],self.g.runtime(self.g.lower.expand(self.n,-1,-1)))
        upper=torch.minimum(limits[...,1],self.g.runtime(self.g.upper.expand(self.n,-1,-1)))
        v=m['executable_target_velocity_rad_s']
        self.fail(selected & (~finite_rows(limits)|~finite_rows(v)|(v.abs().amax(-1)>1e-6)|(lower>=upper).any(-1)),23)
        preload=self.g.leg(target-actual)
        neutral=self.g.fk(self.g.leg(target))['feet_body_m']
        anchors=neutral@m['rotation_world_from_body'].transpose(-1,-2)+m['position_world_m'][:,None,:]
        preloadworld=anchors-m['reference_point_world_m']
        actual_fk=self.g.fk(self.g.leg(actual))['feet_body_m']@m['rotation_world_from_body'].transpose(-1,-2)+m['position_world_m'][:,None,:]
        self.fail(selected & ((preload.abs().amax((-1,-2))>.15)|(preloadworld.norm(dim=-1).amax(-1)>.025)
                  |((actual_fk-m['reference_point_world_m']).norm(dim=-1).amax(-1)>.001)),22)
        good=selected & self.active
        # Full selected state reset; another replica's state is never touched.
        for key,tail in FLOAT_FIELDS.items():self.put(key,0.,good)
        for key in BOOL_FIELDS:self.put(key,False,good)
        for key in INT_FIELDS:
            if key not in ('episode','failure'):self.put(key,0,good)
        for key,value in {'ready':True,'episode':episode,'current_leg':-1,'time':m['time_s'],
                'position':m['position_world_m'],'R0':m['rotation_world_from_body'],'q':target,'lower':lower,'upper':upper,
                'neutral':neutral,'anchors':anchors,'measured_anchors':m['reference_point_world_m'],
                'preload_world':preloadworld,'preload_q':preload,'hold_until':m['time_s'],'factor':1.,
                'position_float32':m['position_is_float32']}.items():self.put(key,value,good)
        # Scalar reset checks bounds after setting the new target.
        self._bounds(self.s['q'],self.s['v'],self.s['v'])
        return self.output()

    def _advance(self,target,dt):
        e=self.s['command']-target; a=self.s['rate']+2.*e;decay=math.exp(-2.*dt)
        command=target+(e+a*dt)*decay;rate=(self.s['rate']-2.*a*dt)*decay
        mid=.5*(self.s['command']+command);dyaw=mid[:,2]*dt
        body=torch.stack((mid[:,1],-mid[:,0],torch.zeros_like(dyaw)),-1)
        velocity=(self.R(self.s['yaw']+.5*dyaw)@body[...,None]).squeeze(-1)
        velocity=torch.cat((velocity[:,:2],torch.zeros_like(velocity[:,2:])),1)
        position=self.s['position']+dt*velocity
        # NumPy scalar oracle preserves the reset position dtype on +=. Actual
        # Isaac snapshots arefloat32; synthetic fixtures arefloat64. This is
        # explicit executable state, not an unlabelled tolerance relaxation.
        position=torch.where(self.s['position_float32'][:,None],position.float().to(self.dtype),position)
        return position,self.s['yaw']+dyaw,command,rate

    def _predict(self,target):
        # The scalar repeats198 equal analytical filter steps over7.9s. Evaluate
        # its same closed-form command samples at once, then use the same
        # trapezoidal yaw/position recurrence via cumsum. No replica/198-step loop.
        horizon=7.9;n=math.ceil(horizon/.04);dt=horizon/n
        t=torch.arange(n+1,device=self.device,dtype=self.dtype)*dt
        e=self.s['command']-target;a=self.s['rate']+2.*e
        command=target[:,None,:]+(e[:,None,:]+a[:,None,:]*t[None,:,None])*torch.exp(-2*t)[None,:,None]
        mid=.5*(command[:,1:]+command[:,:-1]);dyaw=mid[:,:,2]*dt
        angle=self.s['yaw'][:,None]+dyaw.cumsum(-1)-.5*dyaw
        body=torch.stack((mid[:,:,1],-mid[:,:,0],torch.zeros_like(angle)),-1)
        initial=(self.s['R0'][:,None,:,:]@body[...,None]).squeeze(-1)
        vx=angle.cos()*initial[:,:,0]-angle.sin()*initial[:,:,1]
        vy=angle.sin()*initial[:,:,0]+angle.cos()*initial[:,:,1]
        increments=torch.stack((vx*dt,vy*dt,torch.zeros_like(vx)),-1)
        position=self.s['position'].clone()
        # Rounding after every scalar += is not equivalent to casting one final
        # cumsum. A fixed198-sample scan preserves that legacy behavior while
        # still batching every replica; later removal requires a new lineage.
        for i in range(198):
            candidate=position+increments[:,i]
            position=torch.where(self.s['position_float32'][:,None],candidate.float().to(self.dtype),candidate)
        yaw=self.s['yaw']+dyaw.sum(-1)
        return position,self.R(yaw)

    def _coeff(self,p0,p1,duration,v0=None,a0=None):
        v0=torch.zeros_like(p0) if v0 is None else v0;a0=torch.zeros_like(p0) if a0 is None else a0
        first=torch.stack((p0,duration[:,None]*v0,.5*duration[:,None].square()*a0),1)
        rhs=torch.stack((p1-first.sum(1),-first[:,1]-2*first[:,2],-2*first[:,2]),1)
        return torch.cat((first,self.inv@rhs),1)

    def _sample(self,which,time):
        prefix='land' if which=='landing' else 'sw';active=self.s[prefix+'_active']
        lift=torch.zeros_like(time) if prefix=='land' else torch.full_like(time,.007)
        # Inactive state may retain internal coefficients for audit. Supply the
        # numeric kernel's explicit finite-zero sentinel at this boundary.
        coeff=torch.where(active[:,None,None],self.s[prefix+'_coeff'],0.)
        start=torch.where(active,self.s[prefix+'_start'],0.);duration=torch.where(active,self.s[prefix+'_duration'],0.)
        lift=torch.where(active,lift,0.)
        out=self.g.polynomial(coeff,start,duration,lift,time,active,world_up=self.up)
        p,v,a=out['position_world_m'],out['velocity_world_mps'],out['acceleration_world_mps2']
        if prefix=='sw':
            h=self.g.polynomial(torch.where(active[:,None,None],self.s['sw_hcoeff'],0.),start,duration*.8,
                                torch.zeros_like(lift),time,active,world_up=self.up)
            done=time>=start+duration*.8
            hp=torch.where(done[:,None],self.s['sw_end'],h['position_world_m'])
            hv=torch.where(done[:,None],0.,h['velocity_world_mps']);ha=torch.where(done[:,None],0.,h['acceleration_world_mps2'])
            p=torch.cat((hp[:,:2],p[:,2:]),-1);v=torch.cat((hv[:,:2],v[:,2:]),-1);a=torch.cat((ha[:,:2],a[:,2:]),-1)
        return p,v,a

    def output(self):
        valid=self.active
        # Complete typed state; optional trajectory values are masked only in a
        # future encoder. No Python per-row serialization on the dynamic path.
        return {'valid':valid.clone(),'failure_code':self.s['failure'].clone(),'error_definitions':ERRORS,
                'q_ref':torch.where(valid[:,None],self.s['q'],torch.nan),
                'v_ref':torch.where(valid[:,None],self.s['v'],torch.nan),
                'a_ref':torch.where(valid[:,None],self.s['a'],torch.nan),
                'state':{k:v.clone() for k,v in self.s.items()},'joint_names_runtime':self.names,
                'source_identity':dict(self.identity),'policy_training_allowed':False,'physical_admission':False}

    def step(self,m,requested,dt=.02):
        if dt!=.02:raise ValueError('Exact50Hz control required')
        contact,numeric,terminal=self._read(m);self.g.check(requested,(self.n,3),'requested twist')
        self.put('failure',1,~self.s['ready'] & (self.s['failure']==0))
        self.fail(~numeric,2)
        self.fail(m['position_is_float32']!=self.s['position_float32'],2)
        self.fail((m['time_s']-self.s['time']).abs()>1e-6,20)
        self.fail(~finite_rows(requested),21)
        requested=torch.nan_to_num(requested)
        stop=(requested.norm(dim=-1)==0)&(self.s['requested'].norm(dim=-1)>0)
        moving=requested.norm(dim=-1)>0
        self.put('stop_time',self.s['time'],self.active&stop);self.put('stop_valid',True,self.active&stop)
        self.put('quiet_valid',False,self.active&(stop|moving));self.put('stop_valid',False,self.active&moving)
        self.put('requested',requested)
        factor=torch.minimum(torch.ones(self.n,device=self.device,dtype=self.dtype),
            torch.minimum(.005/requested[:,:2].norm(dim=-1).clamp_min(1e-20),.015/requested[:,2].abs().clamp_min(1e-20)))
        target=factor[:,None]*requested
        self.fail(terminal,3)
        margin,count=self._support(m,contact,self.s['current_leg'])
        self.fail(count<5,4);self.fail(margin<.025,5)
        body_error=(m['position_world_m'][:,:2]-self.s['position'][:,:2]).norm(dim=-1)
        self.fail(body_error>.035,6)
        drift=(m['reference_point_world_m']-self.s['measured_anchors']).norm(dim=-1)
        stance=contact&(self.legs!=self.s['current_leg'][:,None])
        self.fail(torch.where(stance,drift,0.).amax(-1)>.02,7)
        self.put('support_margin',margin);self.put('body_error',body_error);self.put('stance_drift',drift)
        leg=self.s['current_leg'].clamp_min(0); has_leg=self.s['current_leg']>=0
        z=m['reference_point_world_m'][self.ids,leg,2]
        foot=m['reference_point_world_m'][self.ids,leg]
        footv=m['reference_point_velocity_world_mps'][self.ids,leg];speed=footv.norm(dim=-1)
        touching=contact[self.ids,leg]; air=has_leg&~touching
        # Raw force-free history remains distinct from qualified airborne state.
        self.put('raw_force_free_runs',self.s['raw_force_free_runs']+1,self.active&air&(self.s['flight_count']==0))
        self.put('raw_force_free_samples',self.s['raw_force_free_samples']+1,self.active&air)
        self.put('flight_count',self.s['flight_count']+1,self.active&air)
        self.put('contact_count',0,self.active&air)
        self.put('flight_peak_z',torch.maximum(self.s['flight_peak_z'],z),self.active&air)
        qualified=air&(self.s['flight_count']>=2)&((self.s['flight_peak_z']-self.s['flight_baseline_z'])>=.002)
        self.put('flight_seen',True,self.active&qualified)
        self.put('mode',1,self.active&qualified&~self.s['land_active'])
        beforeflight=has_leg&touching&~self.s['flight_seen']
        rebound=beforeflight&(self.s['flight_count']>0)
        for key,value in {'unqualified_contact_returns':self.s['unqualified_contact_returns']+1,
                          'unqualified_return_time':self.s['time'],'last_unqualified_run_samples':self.s['flight_count'],
                          'unqualified_lift':self.s['flight_peak_z']-self.s['flight_baseline_z'],
                          'unqualified_return_valid':True}.items():self.put(key,value,self.active&rebound)
        self.put('flight_count',0,self.active&has_leg&touching)
        self.put('flight_baseline_z',z,self.active&beforeflight);self.put('flight_peak_z',z,self.active&beforeflight)
        self.fail(has_leg&~self.s['flight_seen']&(self.s['time']>=self.s['sw_start']+1.),25)
        self.put('flight_peak_z',torch.maximum(self.s['flight_peak_z'],z),self.active&self.s['flight_seen']&has_leg)
        descending=has_leg&self.s['flight_seen']&(self.s['time']>=self.s['sw_start']+1.)&(z<self.s['flight_peak_z']-.00025)&(footv[:,2]<-.0001)
        self.put('descent_seen',True,self.active&descending)
        trigger=has_leg&touching&self.s['flight_seen']&~self.s['land_active']
        self.fail(trigger&((self.s['time']<self.s['sw_start']+1.)|~self.s['descent_seen']|((self.s['flight_peak_z']-self.s['flight_baseline_z'])<.002)),8)
        self.fail(trigger&(speed>.04),9)
        p0,v0,a0=self._sample('swing',self.s['time'])
        endpoint=torch.cat((p0[:,:2],(z+self.s['preload_world'][self.ids,leg,2])[:,None]),-1)
        original=(self.s['sw_end']-foot-self.s['preload_world'][self.ids,leg]).norm(dim=-1)
        correction=(endpoint-self.s['sw_end']).norm(dim=-1)
        self.fail(trigger&(torch.maximum(original,correction)>.012),10)
        duration=(self.s['sw_start']+self.s['sw_duration']-self.s['time']).clamp(.10,.50)
        coeff=self._coeff(p0,endpoint,duration,v0,a0)
        u=self.sample_times;k=torch.arange(6,device=self.device,dtype=self.dtype)
        samples=torch.einsum('sk,nkj->nsj',u[:,None]**k,coeff)
        excursion=(samples-p0[:,None,:]).norm(dim=-1).amax(-1)
        self.fail(trigger&(excursion>.012),11)
        launch_landing=self.active&trigger
        for key,value in {'land_active':True,'land_start':self.s['time'],'land_duration':duration,'land_coeff':coeff,
                'land_end':endpoint,'landing_trigger':self.s['time'],'landing_origin':foot,'landing_preload':endpoint-foot,
                'landing_correction':correction,'landing_original_error':original,'landing_excursion':excursion,
                'landing_metadata_valid':True,'landing_gap':0,'contact_count':0,'mode':6}.items():self.put(key,value,launch_landing)
        landing=has_leg&self.s['land_active']
        self.put('landing_gap',0,self.active&landing&touching)
        self.fail(landing&touching&((foot-self.s['landing_origin']).norm(dim=-1)>.012),12)
        landend=self.s['land_start']+self.s['land_duration']
        finishing=landing&touching&(self.s['time']>=landend-1e-8)
        preload=self.s['land_end']-foot
        self.fail(finishing&(((preload-self.s['landing_preload']).norm(dim=-1)>.012)|(speed>.04)|(preload.norm(dim=-1)>.025)),13)
        self.put('contact_count',self.s['contact_count']+1,self.active&finishing)
        confirm=self.active&finishing&(self.s['contact_count']>=3)
        legmask=(self.legs==leg[:,None])&confirm[:,None]
        self.put('anchors',torch.where(legmask[:,:,None],self.s['land_end'][:,None,:],self.s['anchors']),confirm)
        self.put('measured_anchors',torch.where(legmask[:,:,None],foot[:,None,:],self.s['measured_anchors']),confirm)
        self.put('preload_world',self.s['anchors']-self.s['measured_anchors'],confirm)
        for key,value in {'current_leg':-1,'sw_active':False,'land_active':False,'mode':3,
                          'hold_until':self.s['time']+.30,'touchdowns':self.s['touchdowns']+1}.items():self.put(key,value,confirm)
        gap=landing&~touching
        self.put('landing_gap',self.s['landing_gap']+1,self.active&gap);self.put('contact_count',0,self.active&gap)
        self.fail(gap&(self.s['landing_gap']*.02>.10+1e-9),14)
        still=self.s['current_leg']>=0
        self.put('mode',torch.where(self.s['time']<landend,6,7),self.active&still&self.s['land_active'])
        end=torch.where(self.s['land_active'],landend,self.s['sw_start']+self.s['sw_duration'])
        overdue=still&(self.s['time']>=end)
        self.put('mode',2,self.active&overdue&~self.s['land_active'])
        pause=overdue&~touching
        target=torch.where(pause[:,None],0.,target);factor=torch.where(pause,0.,factor)
        self.fail(overdue&(~self.s['flight_seen']|((self.s['time']-end)>.60)),15)
        magnitude=torch.maximum(target[:,:2].norm(dim=-1),target[:,2].abs()*.30)
        liftoff=(self.s['current_leg']<0)&(self.s['time']>=self.s['hold_until'])&(magnitude>1e-6)
        nextleg=self.order[self.s['order']]
        nextmargin,nextcount=self._support(m,contact,nextleg)
        self.fail(liftoff&(nextcount<5),4);self.fail(liftoff&(nextmargin<.025),5)
        self.fail(liftoff&~contact[self.ids,nextleg],16)
        predicted_p,predicted_R=self._predict(target)
        endpoint=(predicted_R@self.s['neutral'][self.ids,nextleg,:,None]).squeeze(-1)+predicted_p
        endpoint=torch.cat((endpoint[:,:2],self.s['anchors'][self.ids,nextleg,2,None]),-1)
        start=self.s['anchors'][self.ids,nextleg];duration=torch.full_like(self.s['time'],2.)
        coeff=self._coeff(start,endpoint,duration);hcoeff=self._coeff(start,endpoint,duration*.8)
        launch=self.active&liftoff
        updates={'sw_active':True,'sw_start':self.s['time'],'sw_duration':duration,'sw_coeff':coeff,'sw_hcoeff':hcoeff,
            'sw_end':endpoint,'current_leg':nextleg,'mode':8,'flight_seen':False,'flight_count':0,'contact_count':0,
            'land_active':False,'landing_metadata_valid':False,'landing_gap':0,'descent_seen':False,
            'raw_force_free_samples':0,'raw_force_free_runs':0,'unqualified_contact_returns':0,
            'unqualified_return_time':0.,'last_unqualified_run_samples':0,'unqualified_lift':0.,'unqualified_return_valid':False,
            'flight_baseline_z':m['reference_point_world_m'][self.ids,nextleg,2],
            'flight_peak_z':m['reference_point_world_m'][self.ids,nextleg,2],'flight_valid':True,
            'order':(self.s['order']+1)%6,'liftoffs':self.s['liftoffs']+1}
        for key,value in updates.items():self.put(key,value,launch)
        self.put('factor',factor);self.put('command_target',target)
        finite_stop=(requested.norm(dim=-1)==0)&(self.s['current_leg']<0)&(self.s['command'].abs().amax(-1)<=1e-6)&(self.s['rate'].abs().amax(-1)<=2e-5)
        self.put('command',0.,self.active&finite_stop);self.put('rate',0.,self.active&finite_stop)
        pos,yaw,command,rate=self._advance(target,.02)
        for key,value in {'position':pos,'yaw':yaw,'command':command,'rate':rate,'time':self.s['time']+.02}.items():self.put(key,value)
        sw=self._sample('swing',self.s['time'])[0];land=self._sample('landing',self.s['time'])[0]
        movingpoint=torch.where(self.s['land_active'][:,None],land,sw)
        mask=self.legs==self.s['current_leg'][:,None]
        points=torch.where(mask[:,:,None],movingpoint[:,None,:],self.s['anchors'])
        local=(points-self.s['position'][:,None,:])@self.R()
        ik=self.g.ik(local);self.fail(~ik['valid'].all(-1),17)
        q=self.g.runtime(ik['q_checked'])
        quiet=(self.s['current_leg']<0)&(self.s['command'].abs().amax(-1)<1e-14)&(self.s['rate'].abs().amax(-1)<1e-12)
        q=torch.where(quiet[:,None],self.s['q'],q);v=(q-self.s['q'])/.02;a=(v-self.s['v'])/.02
        self._bounds(q,v,a)
        self.put('q',q);self.put('v',v);self.put('a',a);self.put('reference_points',points)
        stopping=(self.s['current_leg']<0)&(requested.norm(dim=-1)==0)
        self.put('mode',torch.where(finite_stop,5,4),self.active&stopping)
        firstquiet=self.active&stopping&finite_stop&~self.s['quiet_valid']
        self.put('quiet_time',self.s['time'],firstquiet);self.put('quiet_valid',True,firstquiet)
        return self.output()
