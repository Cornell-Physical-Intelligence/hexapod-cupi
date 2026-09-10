"""Strict CPU observation contract for wave002 + residual002. No learner or physics."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from pathlib import Path

SCHEMA = 'c_wave002_residual002_instrumented_observation_v2'
WAVE_SHA256 = 'ef92745daf4c3bf544b6c6f9f65e18f19745d9e84811c8888a464312f864a56b'
RESIDUAL_SHA256 = 'fbf65749c2d1ffc36b22986f71c0972ce7b1ecf82beebef57a0bd3e1be966e40'
LEGS = ('lf','lm','lr','rf','rm','rr')
MODES = ('hold','swing','awaiting_contact','contact_hold','stopping_reference_motion',
         'reference_quiet_hold','landing_blend','awaiting_landing_support')
SOURCES = ('simulator_truth_instrumented','synthetic_fixture','future_estimator_unqualified')
DT = .02
HISTORY = 5
FRAME = 63
SOURCE_CONTRACT = json.loads((Path(__file__).with_name('source_contract.json')).read_text())


def required(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError('Missing required field: '+key)
    return mapping[key]


def arr(value, shape, name, boolean=False):
    a = np.asarray(value)
    if a.shape == (1,*shape): a = a[0]
    if a.shape != shape or (boolean and a.dtype != np.bool_):
        raise ValueError('Wrong shape/type for '+name)
    if not boolean and (not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all()):
        raise ValueError('Nonfinite or nonnumeric field: '+name)
    return a.copy() if boolean else a.astype(np.float64,copy=True)


def scalar(value,name): return float(arr(value,(),name))
def bit(value,name): return float(arr(value,(),name,True))


def rotation(value,name):
    R=arr(value,(3,3),name)
    if not np.allclose(R.T@R,np.eye(3),atol=2e-4,rtol=0) or abs(np.linalg.det(R)-1)>2e-4:
        raise ValueError('Invalid proper rotation: '+name)
    return R


def integer(value,name,maximum=None):
    n=scalar(value,name)
    if n<0 or n!=int(n) or isinstance(value,(bool,np.bool_)) or (maximum is not None and n>maximum):
        raise ValueError('Invalid count/index: '+name)
    return int(n)


def finite_tree(value,path='packet'):
    if isinstance(value,dict):
        for k,v in value.items(): finite_tree(v,path+'.'+k)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): finite_tree(v,path+f'[{i}]')
    elif isinstance(value,np.ndarray):
        if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():
            raise ValueError('Nonfinite '+path)
    elif isinstance(value,(float,np.floating)) and not math.isfinite(value):
        raise ValueError('Nonfinite '+path)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


@dataclass(frozen=True)
class Limits:
    profile: str = 'formal_004'
    residual_radius_rad: float = .02
    residual_velocity_rad_s: float = .25
    total_velocity_rad_s: float = 2.

    def validate(self):
        expected={'formal_004':2.,'diagnostic_003':1.5}
        if self.profile not in expected or self.total_velocity_rad_s!=expected[self.profile]:
            raise ValueError('Explicit separate .04 formal or .03 diagnostic profile required')
        if self.residual_radius_rad!=.02 or self.residual_velocity_rad_s!=.25:
            raise ValueError('This schema binds the reviewed residual radius/rate')


class ObservationBuilder:
    """Batched current-state encoding with isolated five-frame histories.

    reset(ids, episode_ids, reset_times) precedes first build; first step is0.
    build takes one packet per replica, each with a current measurement,
    frozen reference output, controller output, next requested twist and
    explicit provenance/contact freshness. Validation is transactional.
    """
    def __init__(self,joint_names,nominal_joint_positions,num_envs,*,limits=Limits()):
        self.names=tuple(joint_names)
        if len(self.names)!=18 or len(set(self.names))!=18 or not all(isinstance(n,str) for n in self.names):
            raise ValueError('18 unique runtime joint names required')
        if set(self.names)!=set(SOURCE_CONTRACT['joint_names_leg_major']):
            raise ValueError('Joint names differ from the frozen C geometry')
        if type(num_envs) is not int or num_envs<1:raise ValueError('Positive replica count required')
        if set(nominal_joint_positions)!=set(self.names):raise ValueError('Complete named nominal stance required')
        self.nominal=arr([nominal_joint_positions[n] for n in self.names],(18,),'nominal')
        expected_nominal=arr([SOURCE_CONTRACT['nominal_joint_positions'][n] for n in self.names],(18,),'frozen nominal')
        if not np.array_equal(self.nominal,expected_nominal):raise ValueError('Nominal stance differs from frozen named source')
        limits.validate();self.limits=limits;self.n=num_envs
        self.implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.source_contract_sha256=hashlib.sha256(Path(__file__).with_name('source_contract.json').read_bytes()).hexdigest()
        self.history=np.zeros((num_envs,HISTORY,FRAME),np.float64)
        self.valid=np.zeros((num_envs,HISTORY),bool)
        self.episodes=[None]*num_envs;self.reset_times=np.zeros(num_envs)
        self.steps=np.full(num_envs,-1,int);self.signatures=[None]*num_envs
        self.cached=[None]*num_envs;self.fields=None
        self.seen_episodes=[set() for _ in range(num_envs)]

    def reset(self,ids,episode_ids,reset_times):
        ids=list(ids);episode_ids=list(episode_ids);times=list(reset_times)
        if len(ids)!=len(set(ids)) or len(ids)!=len(episode_ids) or len(ids)!=len(times):
            raise ValueError('Reset lists must match with unique replicas')
        for i,e,t in zip(ids,episode_ids,times):
            if type(i) is not int or not 0<=i<self.n or not isinstance(e,str) or not e or not math.isfinite(t):
                raise ValueError('Invalid explicit reset identity/time')
            if e in self.seen_episodes[i]:raise ValueError('Reset requires a fresh episode identity')
        for i,e,t in zip(ids,episode_ids,times):
            self.history[i]=0;self.valid[i]=False;self.episodes[i]=e;self.reset_times[i]=t
            self.steps[i]=-1;self.signatures[i]=None;self.cached[i]=None
            self.seen_episodes[i].add(e)

    def _encode(self,p):
        # Contact points may be unknown, but their representation is explicit
        # finite zeros + validity. NaNs are never accepted into this contract.
        finite_tree(p)
        if required(p,'wave_source_sha256')!=WAVE_SHA256 or required(p,'residual_source_sha256')!=RESIDUAL_SHA256:
            raise ValueError('Frozen source identity mismatch')
        if required(p,'measurement_source') not in SOURCES:raise ValueError('Explicit measurement provenance required')
        if required(p,'reference_source')!='virtual_desired_motion_not_prescribed_physics_pose':
            raise ValueError('Reference provenance must distinguish ideal desired motion')
        if tuple(required(p,'joint_names_runtime'))!=self.names:raise ValueError('Runtime joint order mismatch')
        measured=required(p,'measurement');ref=required(p,'reference');c=required(p,'controller')
        state=required(ref,'state')
        if serial(required(required(ref,'diagnostics'),'configuration'))!=SOURCE_CONTRACT['wave_configuration']:
            raise ValueError('Reference configuration differs from versioned source contract')
        if not bool(arr(required(ref,'valid'),(1,),'reference.valid',True)[0]) or required(ref,'failure_reason') is not None:
            raise ValueError('Invalid reference cannot produce a policy observation')
        if tuple(required(ref,'joint_names_runtime'))!=self.names:raise ValueError('Reference runtime order mismatch')
        now=scalar(required(measured,'time_s'),'measurement.time_s')
        if abs(scalar(required(ref,'target_time_s'),'reference.time')-now)>1e-7 or abs(scalar(required(state,'desired_time_s'),'state.time')-now)>1e-7:
            raise ValueError('Reference and completed physical sample must be time-aligned')
        if abs(scalar(required(p,'time_s'),'packet.time')-now)>1e-7:raise ValueError('Packet/sample time mismatch')
        if bit(required(measured,'terminated'),'terminated') or bit(required(measured,'truncated'),'truncated'):
            raise ValueError('Terminal sample requires explicit new reset, never continuation')
        position=arr(required(measured,'position_world_m'),(3,),'position')
        R=rotation(required(measured,'rotation_world_from_body'),'body rotation')
        up=arr(required(p,'world_up_vector'),(3,),'world up')
        if abs(np.linalg.norm(up)-1)>1e-6:raise ValueError('Unit world-up direction required')
        point=lambda x:(arr(x,(3,),'point')-position)@R
        points=lambda x:(arr(x,(6,3),'points')-position)@R
        vector=lambda x:arr(x,(3,),'vector')@R
        q=arr(required(measured,'joint_position_rad'),(18,),'joint q')
        dq=arr(required(measured,'joint_velocity_rad_s'),(18,),'joint dq')
        gyro=arr(required(measured,'gyro_body_rad_s'),(3,),'gyro')
        gravity=arr(required(measured,'projected_gravity_body'),(3,),'gravity')
        if abs(np.linalg.norm(gravity)-1)>1e-3 or not np.allclose(gravity,-up@R,atol=3e-4,rtol=0):
            raise ValueError('Gravity, world-up and body frame disagree')
        commanded=arr(required(p,'requested_twist'),(3,),'next command')
        total_q=arr(required(c,'target_position_rad'),(18,),'target q')
        total_v=arr(required(c,'target_velocity_rad_s'),(18,),'target v')
        residual_q=arr(required(c,'residual_position_rad'),(18,),'residual q')
        residual_v=arr(required(c,'residual_velocity_rad_s'),(18,),'residual v')
        goal=arr(required(c,'residual_goal_rad'),(18,),'residual goal')
        ref_q=arr(required(c,'reference_position_rad'),(18,),'reference q')
        ref_v=arr(required(c,'reference_velocity_rad_s'),(18,),'reference v')
        if (not np.allclose(total_q,ref_q+residual_q,atol=1e-9,rtol=0)
            or not np.allclose(total_v,ref_v+residual_v,atol=1e-9,rtol=0)
            or not np.allclose(ref_q,arr(required(ref,'q_ref'),(18,),'output q'),atol=1e-9,rtol=0)
            or not np.allclose(ref_v,arr(required(ref,'v_ref'),(18,),'output v'),atol=1e-9,rtol=0)
            or not np.allclose(total_q,arr(required(measured,'joint_target_rad'),(18,),'emitted q'),atol=3e-7,rtol=0)):
            raise ValueError('Reference/residual/executable state disagrees')
        if (abs(residual_q).max()>self.limits.residual_radius_rad+1e-7
            or abs(goal).max()>self.limits.residual_radius_rad+1e-7
            or abs(residual_v).max()>self.limits.residual_velocity_rad_s+1e-7
            or abs(total_v).max()>self.limits.total_velocity_rad_s+1e-5):
            raise ValueError('Controller state exceeds declared profile')
        frame=np.concatenate((gyro*.25,gravity,commanded*np.array([5,5,2.5]),q-self.nominal,dq*.05,goal/self.limits.residual_radius_rad))
        values=[];fields=[]
        def add(name,value,source):
            a=np.asarray(value,dtype=np.float64).reshape(-1)
            if not np.isfinite(a).all():raise ValueError('Nonfinite encoded '+name)
            values.extend(a);fields.append(dict(name=name,width=len(a),source=source))
        add('executable_target_offset',total_q-self.nominal,'known_controller')
        add('executable_target_velocity',total_v/self.limits.total_velocity_rad_s,'known_controller')
        add('residual_position',residual_q/self.limits.residual_radius_rad,'known_controller')
        add('residual_velocity',residual_v/self.limits.residual_velocity_rad_s,'known_controller')
        add('measured_body_linear_velocity',arr(required(measured,'velocity_body_mps'),(3,),'body velocity')*5,'measurement')
        add('world_up_body',up@R,'measurement_frame')
        add('measured_toes_body',points(required(measured,'reference_point_world_m'))*5,'measurement')
        add('measured_toe_world_velocity_body',arr(required(measured,'reference_point_velocity_world_mps'),(6,3),'toe velocities')@R*5,'measurement')
        contact=arr(required(measured,'distal_contact'),(6,),'contact',True)
        valid=arr(required(p,'contact_valid'),(6,),'contact validity',True)
        age=arr(required(p,'contact_age_s'),(6,),'contact age')
        contact_points_valid=arr(required(measured,'contact_point_valid'),(6,),'point validity',True)
        if (not valid.all() or (age<0).any() or (age>.04+1e-9).any()
            or np.any(contact & ~contact_points_valid)):
            raise ValueError('Unknown/stale contact estimate cannot admit control')
        cp=arr(required(measured,'contact_point_world_m'),(6,3),'contact points')
        if np.any(cp[~contact_points_valid]!=0):raise ValueError('Unknown contact points require explicit finite zero sentinel')
        cp_body=(cp-position)@R;cp_body[~contact_points_valid]=0
        add('distal_contact',contact,'measurement');add('contact_valid',valid,'measurement')
        add('contact_age',age/.1,'measurement');add('contact_point_valid',contact_points_valid,'measurement')
        add('contact_points_body',cp_body*5,'measurement')
        for key in ('shaft_contact','coxa_contact','femur_contact'):
            add(key,arr(required(measured,key),(6,),key,True),'measurement')
        add('base_contact',bit(required(measured,'base_contact'),'base_contact'),'measurement')
        for name,key in [('previous_reference_request','requested_command'),('admitted_target_twist','admitted_target_command'),('admitted_filtered_twist','admitted_command')]:
            add(name,arr(required(ref,key),(3,),key)*[5,5,2.5],'known_reference')
        filtered=arr(required(state,'command_filter_velocity'),(3,),'filter velocity')
        if not np.allclose(filtered,arr(ref['admitted_command'],(3,),'admitted'),atol=1e-12,rtol=0):raise ValueError('Filter state mismatch')
        add('admitted_twist_derivative',arr(required(state,'command_filter_rate'),(3,),'filter rate'),'known_reference')
        mode=required(state,'mode');active=required(state,'current_leg')
        if mode not in MODES or active not in (None,*LEGS):raise ValueError('Unknown versioned mode/leg')
        if tuple(required(state,'gait_order'))!=('lf','rr','lm','rf','lr','rm'):raise ValueError('Frozen gait order mismatch')
        add('reference_mode',np.eye(len(MODES))[MODES.index(mode)],'known_reference')
        add('active_leg',np.eye(7)[(None,*LEGS).index(active)],'known_reference')
        add('next_wave_order',np.eye(6)[integer(required(state,'next_wave_order_index'),'order',5)],'known_reference')
        for key in ('flight_count','contact_count','liftoffs','confirmed_touchdowns','landing_contact_gap_steps'):
            add(key,integer(required(state,key),key)/100.,'known_reference')
        for key in ('flight_seen','actual_descent_seen'):
            add(key,bit(required(state,key),key),'known_reference')
        add('hold_remaining',max(0.,scalar(required(state,'hold_until_s'),'hold deadline')-now)/2,'known_reference')
        for key in ('stop_requested_time_s','reference_quiet_time_s','landing_trigger_s'):
            stamp=required(state,key)
            if stamp is None:add(key+'_valid_and_elapsed',[0,0],'known_reference')
            else:
                elapsed=now-scalar(stamp,key)
                if elapsed < -1e-7:raise ValueError('Future event timestamp '+key)
                add(key+'_valid_and_elapsed',[1,max(0.,elapsed)/10],'known_reference')
        for name,key in [('reference_anchors_body','reference_anchors_world_m'),('measured_anchors_body','measured_anchors_world_m')]:
            add(name,points(required(state,key))*5,'known_reference_from_measurement')
        anchors=arr(state['reference_anchors_world_m'],(6,3),'anchors')
        measured_anchors=arr(state['measured_anchors_world_m'],(6,3),'measured anchors')
        preload=arr(required(state,'reference_minus_measured_preload_world_m'),(6,3),'preload')
        if not np.allclose(anchors-measured_anchors,preload,atol=1e-8,rtol=0):raise ValueError('Anchor/preload inconsistency')
        add('preload_body',preload@R*5,'known_reference_from_measurement')
        add('neutral_toes_initial_body',arr(required(state,'neutral_reference_toes_body_m'),(6,3),'neutral toes')*5,'known_reference')
        add('desired_position_error_body',point(required(state,'desired_position_world_m'))*5,'ideal_reference_vs_measurement')
        add('desired_rotation_relative',R.T@rotation(required(state,'desired_rotation_world_from_body'),'desired rotation'),'ideal_reference_vs_measurement')
        add('initial_desired_rotation_relative',R.T@rotation(required(state,'initial_desired_rotation_world_from_body'),'initial rotation'),'known_reference')
        add('desired_yaw_delta',scalar(required(state,'desired_yaw_delta_rad'),'desired yaw')/math.pi,'known_reference')
        lower=arr(required(state,'joint_lower_leg_major_rad'),(6,3),'lower')
        upper=arr(required(state,'joint_upper_leg_major_rad'),(6,3),'upper')
        if np.any(lower>=upper):raise ValueError('Invalid named joint bounds')
        for key in ('joint_lower_leg_major_rad','joint_upper_leg_major_rad'):
            add(key,arr(required(state,key),(6,3),key),'known_reference_limits')
        baseline=required(state,'flight_baseline_z_m');peak=required(state,'flight_peak_z_m')
        if (baseline is None)!=(peak is None):raise ValueError('Flight height validity mismatch')
        if baseline is None:add('flight_height_valid_relative',[0,0,0],'measurement_history')
        else:
            b=scalar(baseline,'baseline');z=scalar(peak,'peak')
            if z<b-1e-10 or abs((z-b)-scalar(required(state,'measured_flight_lift_m'),'lift'))>1e-8:raise ValueError('Invalid flight lift history')
            add('flight_height_valid_relative',[1,(b-position@up)*5,(z-position@up)*5],'measurement_history')
        for key,is_point in [('landing_contact_origin_world_m',True),('landing_preload_world_m',False)]:
            value=required(state,key)
            add(key+'_valid_body',np.zeros(4) if value is None else np.r_[1,(point(value) if is_point else vector(value))*5],'measurement_history')
        for key in ('landing_endpoint_correction_m','landing_original_contact_error_m','landing_target_excursion_m'):
            value=required(state,key)
            add(key+'_valid_value',[0,0] if value is None else [1,scalar(value,key)*100],'known_reference_audit')
        def trajectory(key):
            v=required(state,key)
            if v is None:add(key+'_trajectory',np.zeros(25),'known_reference');return
            duration=scalar(required(v,'duration_s'),key+' duration');elapsed=now-scalar(required(v,'start_s'),key+' start')
            if duration<=0 or elapsed< -1e-7:raise ValueError('Invalid trajectory timing')
            coeff=arr(required(v,'coefficients'),(6,3),key+' coefficients')
            endpoint=arr(required(v,'endpoint_world_m'),(3,),key+' endpoint')
            if not np.allclose(coeff.sum(0),endpoint,atol=1e-8,rtol=0):raise ValueError('Polynomial endpoint inconsistency')
            transformed=coeff@R;transformed[0]=(coeff[0]-position)@R
            lift=scalar(required(v,'lift_m'),key+' lift')
            if lift<0 or (key=='landing' and lift!=0):raise ValueError('Invalid trajectory lift')
            add(key+'_trajectory',np.r_[1,elapsed/duration,duration/2,(endpoint-position)@R*5,transformed.reshape(-1)*5,lift*100],'known_reference')
        trajectory('swing');trajectory('landing')
        if active is None and state['swing'] is not None or active is not None and state['swing'] is None:
            raise ValueError('Active leg/swing mismatch')
        if mode in ('landing_blend','awaiting_landing_support') and state['landing'] is None:
            raise ValueError('Landing mode has no serialized landing trajectory')
        critic=arr(required(p,'critic_true_navigation_velocity'),(3,),'critic privileged velocity')*5
        with np.errstate(over='ignore'):
            if not np.isfinite(critic.astype(np.float32)).all():raise ValueError('Float32 critic overflow')
        return frame,np.asarray(values),fields,critic

    def build(self,packets):
        if len(packets)!=self.n:raise ValueError('One complete packet per replica required')
        pending=[];layout=None
        for i,p in enumerate(packets):
            if required(p,'episode_id')!=self.episodes[i] or self.episodes[i] is None:
                raise ValueError('Episode mismatch; explicit reset required')
            step=integer(required(p,'step_index'),'step')
            now=scalar(required(p,'time_s'),'time')
            if abs(now-self.reset_times[i]-step*DT)>1e-7:raise ValueError('Stale/nonuniform history timestamp')
            frame,current,fields,critic=self._encode(p)
            signature=hashlib.sha256(json.dumps(serial(p),sort_keys=True,allow_nan=False).encode()).hexdigest()
            if step==self.steps[i]:
                if signature!=self.signatures[i]:raise ValueError('Same-step input changed; sensor/noise/history timing ambiguous')
                pending.append((i,step,signature,None,None,self.cached[i].copy(),critic));continue
            if step!=self.steps[i]+1:raise ValueError('Missing/stale history step; reset instead of leaking stale history')
            h=np.roll(self.history[i],-1,axis=0);v=np.roll(self.valid[i],-1)
            h[-1]=frame;v[-1]=True
            actor=np.r_[h.reshape(-1),v.astype(float),(now-self.reset_times[i])/90,current].astype(np.float32)
            if not np.isfinite(actor).all():raise ValueError('Float32 observation overflow')
            pending.append((i,step,signature,h,v,actor,critic));layout=fields
        if layout is not None:
            if self.fields is not None and layout!=self.fields:raise ValueError('Schema changed between samples')
        # Commit only after every replica is valid, so a bad row changes none.
        for i,step,signature,h,v,actor,_ in pending:
            if h is not None:
                self.history[i]=h;self.valid[i]=v;self.steps[i]=step;self.signatures[i]=signature;self.cached[i]=actor.copy()
        if layout is not None:self.fields=deepcopy(layout)
        actor=np.stack([r[5] for r in pending]);critic=np.stack([np.r_[r[5],r[6]] for r in pending]).astype(np.float32)
        if not np.isfinite(critic).all():raise ValueError('Float32 critic overflow')
        return {'policy':actor,'critic':critic,'schema':self.spec(),
                'measurement_sources':[p['measurement_source'] for p in packets],
                'deployment_qualified':False,'policy_training_allowed':False}

    def spec(self):
        if self.fields is None:raise ValueError('Build one validated sample before exporting field slices')
        fields=[dict(name='sensor_history',width=HISTORY*FRAME,source='measurement_and_known_commands'),
                dict(name='history_valid',width=HISTORY,source='encoder'),dict(name='episode_age',width=1,source='encoder')]+deepcopy(self.fields)
        offset=0
        for f in fields:f['start']=offset;offset+=f['width'];f['stop']=offset
        spec=dict(schema=SCHEMA,actor_width=offset,critic_width=offset+3,fields=fields,
            critic_extra={'name':'privileged_true_navigation_velocity','start':offset,'stop':offset+3},
            control_dt_s=DT,history_frames=HISTORY,sensor_frame_width=FRAME,joint_names_runtime=self.names,
            limits=dict(self.limits.__dict__),wave_source_sha256=WAVE_SHA256,residual_source_sha256=RESIDUAL_SHA256,
            nominal_joint_positions_rad_runtime=self.nominal.tolist(),
            encoder_implementation_sha256=self.implementation_sha256,
            source_contract_sha256=self.source_contract_sha256,
            policy_training_allowed=False,deployment_qualified=False,checkpoint_compatible_with_prior_lineages=False,
            terrain_features_present=False,reference_source='virtual_desired_motion_not_prescribed_physics_pose')
        spec['schema_identity_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        return spec
