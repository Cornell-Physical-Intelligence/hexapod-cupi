"""Versioned batched wave004/residual002 observation and reset-safe history.

Instrumented simulator observations, not deployable proprioception. Raw SDK
velocities are preserved; finite-difference rates are separate interval averages.
No old checkpoint can load this contract. No learner or physics admission.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from batch_wave import FLOAT_FIELDS,BOOL_FIELDS,INT_FIELDS,MODES
from tensor_kernel import TensorGeometry,finite_rows

CONTRACT=json.loads((HERE/'source_contract.json').read_text())
SOURCES=('simulator_raw_instrumented','synthetic_fixture','future_estimator_unqualified')
CONTROLLER_KEYS=('target_position_rad','target_velocity_rad_s','target_acceleration_rad_s2',
 'reference_position_rad','reference_velocity_rad_s','reference_acceleration_rad_s2',
 'residual_position_rad','residual_velocity_rad_s','residual_acceleration_rad_s2','residual_goal_rad','raw_residual_action')
MEASURED_FLOAT={'time_s':(), 'position_world_m':(3,), 'rotation_world_from_body':(3,3),
 'velocity_body_mps':(3,), 'root_link_velocity_body_mps':(3,), 'gyro_body_rad_s':(3,),
 'projected_gravity_body':(3,), 'joint_position_rad':(18,), 'joint_velocity_rad_s':(18,),
 'joint_target_rad':(18,), 'reference_point_world_m':(6,3), 'reference_point_velocity_world_mps':(6,3),
 'contact_point_world_m':(6,3), 'contact_age_s':(6,)}
MEASURED_BOOL={'contact_point_valid':(6,), 'contact_valid':(6,), 'distal_contact':(6,),
 'shaft_contact':(6,), 'coxa_contact':(6,), 'femur_contact':(6,), 'base_contact':(),
 'terminated':(), 'truncated':(), 'measurement_valid':(), 'position_is_float32':()}
EXCLUDED_REFERENCE={'time':'represented by verified episode age and sample sequence',
 'episode':'identity guard, not a policy feature','ready':'invalid rows cannot be encoded',
 'failure':'invalid rows cannot be encoded', 'support_margin':'prior-sample diagnostic, not transition state',
 'body_error':'prior-sample diagnostic; current desired/measured error is encoded',
 'stance_drift':'prior-sample diagnostic; current feet and stored anchors are encoded',
 'reference_points':'prior diagnostic, including uninitialized reset zeros; reconstructable from trajectories/anchors'}


class ObservationBuilder:
    def __init__(self,joint_names,num_envs,*,device='cpu'):
        expected={'reference/FREEZE_SHA256.json':'reference_freeze_sha256',
                  'reference/batch_wave.py':'batch_reference_sha256',
                  'reference_residual_oracle.py':'residual_source_sha256',
                  'physics_telemetry_oracle.py':'telemetry_oracle_sha256'}
        for path,key in expected.items():
            if hashlib.sha256((HERE/path).read_bytes()).hexdigest()!=CONTRACT[key]:raise ValueError('Changed bound source: '+path)
        for relative,sha in json.loads((HERE/'reference/FREEZE_SHA256.json').read_text()).items():
            if hashlib.sha256((HERE/'reference'/relative).read_bytes()).hexdigest()!=sha:raise ValueError('Changed frozen reference dependency: '+relative)
        self.g=TensorGeometry(joint_names,binding='wave003_7mm',profile='formal_004',device=device)
        self.n=num_envs;self.names=tuple(joint_names);self.device=self.g.device;self.dtype=self.g.dtype
        if type(num_envs) is not int or num_envs<1:raise ValueError('Positive batch count required')
        self.fields=None;self.signature=None;self.cached=None;self.actor_width=None
        self.implementation_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with torch.inference_mode(False):
            self.history=torch.zeros((num_envs,5,63),device=self.device,dtype=self.dtype)
            self.history_valid=torch.zeros((num_envs,5),device=self.device,dtype=torch.bool)
            self.episodes=torch.full((num_envs,),-1,device=self.device,dtype=torch.int64)
            self.steps=torch.full_like(self.episodes,-1)
            self.reset_times=torch.zeros(num_envs,device=self.device,dtype=self.dtype)
            self.ready=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
            self.previous_q=torch.zeros((num_envs,18),device=self.device,dtype=self.dtype)
            self.last_fd=torch.zeros_like(self.previous_q)
            self.last_fd_valid=torch.zeros_like(self.ready)

    def reset(self,selected,episode_ids,time_s):
        self.g.check(selected,(self.n,),'reset selection',torch.bool)
        self.g.check(episode_ids,(self.n,),'episode ids',torch.int64)
        self.g.check(time_s,(self.n,),'reset time')
        good=selected&(episode_ids>self.episodes)&(episode_ids>=0)&torch.isfinite(time_s)
        self.history.copy_(torch.where(selected[:,None,None],0.,self.history))
        self.history_valid &= ~selected[:,None]
        self.episodes.copy_(torch.where(good,episode_ids,self.episodes))
        self.steps.copy_(torch.where(selected,-1,self.steps))
        self.reset_times.copy_(torch.where(good,time_s,self.reset_times))
        self.ready.copy_(torch.where(selected,good,self.ready))
        self.previous_q.copy_(torch.where(selected[:,None],0.,self.previous_q))
        self.last_fd.copy_(torch.where(selected[:,None],0.,self.last_fd));self.last_fd_valid &= ~selected
        if self.signature is not None:self.signature.copy_(torch.where(selected[:,None],0.,self.signature))
        if self.cached is not None:self.cached.copy_(torch.where(selected[:,None],0.,self.cached))
        return good

    def _encode(self,m,ref,c,requested,world_up,fd,fd_valid):
        n=self.n;g=self.g
        if (ref.get('source_identity',{}).get('scalar_controller_sha256')!=CONTRACT['scalar_reference_sha256']
                or ref.get('source_identity',{}).get('binding')!='wave004_7mm_horizontal_080'
                or ref.get('source_identity',{}).get('implementation')!='batched_contact_reference_prototype_v1'):
            raise ValueError('Wrong reference source binding')
        if tuple(ref.get('joint_names_runtime',()))!=self.names:raise ValueError('Reference joint order changed')
        state=ref['state'];valid=torch.ones(n,device=self.device,dtype=torch.bool)
        for key,tail in MEASURED_FLOAT.items():g.check(m[key],(n,*tail),key);valid &= finite_rows(m[key])
        for key,tail in MEASURED_BOOL.items():g.check(m[key],(n,*tail),key,torch.bool)
        for key in CONTROLLER_KEYS:g.check(c[key],(n,18),key);valid &= finite_rows(c[key])
        for key,tail in FLOAT_FIELDS.items():g.check(state[key],(n,*tail),'reference.'+key);valid &= finite_rows(state[key])
        for key in BOOL_FIELDS:g.check(state[key],(n,),key,torch.bool)
        for key in INT_FIELDS:g.check(state[key],(n,),key,torch.int64)
        g.check(requested,(n,3),'next requested command');g.check(world_up,(n,3),'world up')
        g.check(ref['valid'],(n,),'reference validity',torch.bool)
        valid &= finite_rows(requested)&finite_rows(world_up)&((world_up.norm(dim=-1)-1).abs()<=1e-6)
        valid &= ref['valid']&state['ready']&(state['failure']==0)&m['measurement_valid']&~m['terminated']&~m['truncated']
        valid &= (state['time']-m['time_s']).abs()<=1e-7
        valid &= m['position_is_float32']==state['position_float32']
        p,R=m['position_world_m'],m['rotation_world_from_body'];valid &= g.rotation_valid(R)&g.rotation_valid(state['R0'])
        upbody=(world_up[:,None,:]@R).squeeze(1)
        valid &= (m['projected_gravity_body']+upbody).abs().amax(-1)<=3e-4
        valid &= ((m['projected_gravity_body'].norm(dim=-1)-1).abs()<=1e-3)
        for left,right in [(c['target_position_rad'],c['reference_position_rad']+c['residual_position_rad']),
                           (c['target_velocity_rad_s'],c['reference_velocity_rad_s']+c['residual_velocity_rad_s']),
                           (c['target_acceleration_rad_s2'],c['reference_acceleration_rad_s2']+c['residual_acceleration_rad_s2']),
                           (c['reference_acceleration_rad_s2'],state['a']),
                           (c['reference_position_rad'],state['q']), (c['reference_velocity_rad_s'],state['v'])]:
            valid &= (left-right).abs().amax(-1)<=1e-9
        valid &= (c['target_position_rad']-m['joint_target_rad']).abs().amax(-1)<=3e-7
        valid &= (c['residual_position_rad'].abs().amax(-1)<=.02+1e-7)&(c['residual_velocity_rad_s'].abs().amax(-1)<=.25+1e-7)
        valid &= (c['residual_goal_rad']-.02*torch.tanh(c['raw_residual_action'])).abs().amax(-1)<=1e-9
        valid &= (c['target_velocity_rad_s'].abs().amax(-1)<=2.+1e-5)&(c['target_acceleration_rad_s2'].abs().amax(-1)<=8.+1e-5)
        valid &= (state['lower']<state['upper']).all(-1)
        valid &= (c['residual_acceleration_rad_s2'].abs().amax(-1)<=2.+1e-5)&(state['a'].abs().amax(-1)<=6.+1e-5)
        valid &= (state['v'].abs().amax(-1)<=1.75+1e-5)
        valid &= (c['target_position_rad']>=state['lower']-1e-10).all(-1)&(c['target_position_rad']<=state['upper']+1e-10).all(-1)
        valid &= (state['q']>=state['lower']+.02-1e-10).all(-1)&(state['q']<=state['upper']-.02+1e-10).all(-1)
        valid &= m['contact_valid'].all(-1)&(m['contact_age_s']>=0).all(-1)&(m['contact_age_s']<=.04+1e-9).all(-1)
        valid &= (~(m['distal_contact']|m['shaft_contact'])|m['contact_point_valid']).all(-1)
        valid &= torch.where(m['contact_point_valid'][:,:,None],True,m['contact_point_world_m']==0.).all((-1,-2))
        valid &= (state['anchors']-state['measured_anchors']-state['preload_world']).abs().amax((-1,-2))<=1e-8
        valid &= (state['sw_active']==(state['current_leg']>=0))&(~state['land_active']|state['sw_active'])
        valid &= (state['mode']>=0)&(state['mode']<len(MODES))&(state['current_leg']>=-1)&(state['current_leg']<6)&(state['order']>=0)&(state['order']<6)
        for key in ('flight_count','contact_count','liftoffs','touchdowns','landing_gap'):valid &= state[key]>=0
        valid &= ~state['land_active']|state['landing_metadata_valid']
        valid &= ~((state['mode']==6)|(state['mode']==7))|state['land_active']
        for prefix in ('sw','land'):
            active=state[prefix+'_active'];duration=state[prefix+'_duration'];elapsed=state['time']-state[prefix+'_start']
            valid &= ~active|((duration>0)&(elapsed>=-1e-7))
            valid &= ~active|((state[prefix+'_coeff'].sum(1)-state[prefix+'_end']).abs().amax(-1)<=1e-8)
        valid &= ~state['sw_active']|((state['sw_hcoeff'].sum(1)-state['sw_end']).abs().amax(-1)<=1e-8)
        valid &= ~state['sw_active']|(state['sw_duration']==2.)
        valid &= ~state['land_active']|((state['land_duration']>=.1)&(state['land_duration']<=.5))
        frame=torch.cat((m['gyro_body_rad_s']*.25,m['projected_gravity_body'],requested*requested.new_tensor([5,5,2.5]),
                         m['joint_position_rad']-g.observation_nominal,m['joint_velocity_rad_s']*.05,c['residual_goal_rad']/.02),-1)
        values=[];fields=[]
        def add(name,value,source):
            flat=value.to(self.dtype).reshape(n,-1);start=sum(v.shape[-1] for v in values)
            fields.append(dict(name=name,start=start,stop=start+flat.shape[-1],source=source));values.append(flat)
        add('executed_target_offset',c['target_position_rad']-g.observation_nominal,'known_controller')
        add('executed_target_velocity',c['target_velocity_rad_s']/2.,'known_controller')
        add('residual_position',c['residual_position_rad']/.02,'known_controller')
        add('residual_velocity',c['residual_velocity_rad_s']/.25,'known_controller')
        add('joint_position_interval_average_rate',fd*.05,'separate50Hz_position_difference_not_instantaneous_velocity')
        add('joint_position_interval_rate_valid',fd_valid,'encoder')
        for key in ('velocity_body_mps','root_link_velocity_body_mps'):add('sdk_reported_'+key,m[key]*5,'raw_simulator_unverified_velocity')
        add('world_up_body',upbody,'measurement_frame')
        points=lambda x:(x-p[:,None,:])@R
        point=lambda x:((x-p)[:,None,:]@R).squeeze(1)
        for key in ('reference_point_world_m','contact_point_world_m'):
            value=points(m[key]);value=torch.where(m['contact_point_valid'][:,:,None],value,0.) if key=='contact_point_world_m' else value
            add(key+'_body',value*5,'instrumented_simulator_contact_or_link_geometry')
        add('sdk_reported_toe_velocity_body',m['reference_point_velocity_world_mps']@R*5,'raw_link_velocity_derived_unverified')
        for key in ('distal_contact','contact_point_valid','contact_valid','shaft_contact','coxa_contact','femur_contact','base_contact'):
            add(key,m[key],'instrumented_simulator_contacts')
        add('contact_age_s',m['contact_age_s']/.1,'explicit_sensor_freshness')
        for key,tail in FLOAT_FIELDS.items():
            if key in EXCLUDED_REFERENCE:continue
            value=state[key];source='known_reference'
            if key=='position':value=point(value)*5;source='ideal_reference_vs_measured_pose'
            elif key=='R0':value=R.transpose(-1,-2)@value
            elif key=='yaw':value=value/torch.pi
            elif key in ('command','requested','command_target'):value=value*value.new_tensor([5,5,2.5])
            elif key=='q':value=value-g.observation_nominal
            elif key=='v':value=value/1.75
            elif key=='a':value=value/6.
            elif key=='neutral':value=value*5
            elif key in ('anchors','measured_anchors'):value=points(value)*5
            elif key=='preload_world':value=value@R*5
            elif key=='preload_q':value=g.runtime(value)
            elif key=='hold_until':value=(value-state['time']).clamp_min(0)/2.
            elif key in ('flight_baseline_z','flight_peak_z'):
                value=torch.where(state['flight_valid'],(value-(p*world_up).sum(-1))*5,0.)
            elif key in ('stop_time','quiet_time','landing_trigger'):
                presence={'stop_time':'stop_valid','quiet_time':'quiet_valid','landing_trigger':'landing_metadata_valid'}[key]
                age=state['time']-value;valid &= ~state[presence]|(age>=-1e-7)
                value=torch.where(state[presence],age.clamp_min(0)/10.,0.)
            elif key=='landing_origin':value=torch.where(state['landing_metadata_valid'][:,None],point(value)*5,0.)
            elif key=='landing_preload':value=torch.where(state['landing_metadata_valid'][:,None],(value[:,None,:]@R).squeeze(1)*5,0.)
            elif key in ('landing_correction','landing_original_error','landing_excursion'):
                value=torch.where(state['landing_metadata_valid'],value*100.,0.)
            elif key.startswith('sw_') or key.startswith('land_'):
                prefix='sw' if key.startswith('sw_') else 'land';active=state[prefix+'_active']
                if key.endswith('coeff'):
                    rotated=value@R;value=torch.cat((point(value[:,0])[:,None,:],rotated[:,1:]),1)*5
                elif key.endswith('end'):value=point(value)*5
                elif key.endswith('start'):value=(state['time']-value)/2.
                elif key.endswith('duration'):value=value/2.
                value=torch.where(active.reshape(n,*([1]*(value.ndim-1))),value,0.)
            add('reference.'+key,value,source)
        for key in BOOL_FIELDS:
            if key not in EXCLUDED_REFERENCE:add('reference.'+key,state[key],'known_reference_validity')
        for key in INT_FIELDS:
            if key in EXCLUDED_REFERENCE:continue
            if key=='mode':value=torch.nn.functional.one_hot(state[key].clamp(0,len(MODES)-1),len(MODES))
            elif key=='current_leg':value=torch.nn.functional.one_hot((state[key]+1).clamp(0,6),7)
            elif key=='order':value=torch.nn.functional.one_hot(state[key].clamp(0,5),6)
            else:value=state[key].to(self.dtype)/100.
            add('reference.'+key,value,'known_reference_transition_state')
        current=torch.cat(values,-1)
        valid &= finite_rows(frame)&finite_rows(current)
        return frame,current,fields,valid

    def build(self,packet):
        if packet.get('measurement_source') not in SOURCES:raise ValueError('Explicit measurement provenance required')
        if tuple(packet.get('joint_names_runtime',()))!=self.names:raise ValueError('Measurement joint order changed')
        m,ref,c=packet['measurement'],packet['reference'],packet['controller']
        episode=packet['episode_ids'];step=packet['step_indices'];n=self.n
        self.g.check(episode,(n,),'episode ids',torch.int64);self.g.check(step,(n,),'step indices',torch.int64)
        same=step==self.steps;following=step==self.steps+1
        q=m['joint_position_rad']
        fd_valid=torch.where(same,self.last_fd_valid,self.steps>=0)
        fd=torch.where(same[:,None],self.last_fd,(q-self.previous_q)/.02)
        fd=torch.where(fd_valid[:,None],fd,0.)
        frame,current,fields,numeric=self._encode(m,ref,c,packet['requested_twist'],packet['world_up'],fd,fd_valid)
        valid=numeric&self.ready&(episode==self.episodes)&(ref['state']['episode']==episode)&(step>=0)
        valid &= ((m['time_s']-self.reset_times-step.to(self.dtype)*.02).abs()<=1e-7)&(same|following)
        critic=self.g.check(packet['critic_simulator_reported_twist'],(n,3),'critic raw reported twist')*5.
        valid &= finite_rows(critic)
        provenance_index=SOURCES.index(packet['measurement_source'])
        signature=torch.cat((frame,current,critic,torch.full((n,1),provenance_index,device=self.device,dtype=self.dtype)),1)
        if self.signature is None:
            with torch.inference_mode(False):
                self.signature=torch.zeros_like(signature);self.cached=torch.zeros((n,315+5+1+current.shape[-1]),device=self.device,dtype=self.dtype)
            self.fields=deepcopy(fields);self.actor_width=self.cached.shape[-1]
        elif fields!=self.fields or self.signature.shape!=signature.shape:raise ValueError('Observation layout changed')
        valid &= ~same|(signature==self.signature).all(-1)
        new=valid&following
        history=torch.cat((self.history[:,1:],frame[:,None,:]),1)
        hvalid=torch.cat((self.history_valid[:,1:],torch.ones((n,1),device=self.device,dtype=torch.bool)),1)
        age=(m['time_s']-self.reset_times)[:,None]/90.
        actor=torch.cat((history.flatten(1),hvalid.to(self.dtype),age,current),1)
        actor=torch.where(same[:,None],self.cached,actor)
        valid &= finite_rows(actor.float())&finite_rows(critic.float());new &= valid
        self.history.copy_(torch.where(new[:,None,None],history,self.history))
        self.history_valid.copy_(torch.where(new[:,None],hvalid,self.history_valid))
        self.steps.copy_(torch.where(new,step,self.steps));self.signature.copy_(torch.where(new[:,None],signature,self.signature))
        self.cached.copy_(torch.where(new[:,None],actor,self.cached))
        self.previous_q.copy_(torch.where(new[:,None],q,self.previous_q))
        self.last_fd.copy_(torch.where(new[:,None],fd,self.last_fd));self.last_fd_valid.copy_(torch.where(new,fd_valid,self.last_fd_valid))
        self.ready &= valid
        return dict(policy=torch.where(valid[:,None],actor.float(),torch.nan),
                    critic=torch.where(valid[:,None],torch.cat((actor,critic),1).float(),torch.nan),valid=valid,
                    interval_joint_rate_rad_s=torch.where((fd_valid&valid)[:,None],fd,torch.nan),interval_rate_valid=(fd_valid&valid).clone(),
                    interval_dt_s=.02,interval_rate_semantics='interval_average_from_consecutive_valid50Hz_positions',
                    raw_sdk_joint_velocity_rad_s=m['joint_velocity_rad_s'].clone(),schema=self.spec(),
                    policy_training_allowed=False,deployment_qualified=False,
                    reported_velocity_physical_consistency_verified=False)

    def spec(self):
        if self.fields is None:raise ValueError('Encode one validated-shaped sample before exporting widths')
        fields=[dict(name='sensor_history_raw_SDK_velocity_preserved',start=0,stop=315,source='five63_value_frames'),
                dict(name='history_valid',start=315,stop=320,source='encoder'),
                dict(name='episode_age',start=320,stop=321,source='encoder')]
        fields += [{**field,'start':field['start']+321,'stop':field['stop']+321} for field in self.fields]
        spec=dict(**CONTRACT,implementation_sha256=self.implementation_sha,joint_names_runtime=self.names,
                  actor_width=self.actor_width,critic_width=self.actor_width+3,fields=fields,
                  excluded_reference_fields=EXCLUDED_REFERENCE,
                  critic_extra='privileged_simulator_reported_twist_unverified_not_physical_truth')
        spec['schema_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        return deepcopy(spec)


def reject_checkpoint(checkpoint):
    """No trained policy exists for this new schema; old widths/weights forbidden."""
    raise ValueError('Observation-only prototype: no checkpoint load or implicit transfer is supported')
