"""CPU-only contact-aware opposing-pair forward reference; no physics state writes.

The original scalar wave005 source remains immutable in oracle/. This new state
machine has two independently qualified feet and a distinct state schema.
"""
from pathlib import Path
import sys,copy,math
from dataclasses import dataclass,asdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'oracle'))
from wave_reference import WaveContactReference,WaveConfig,AdvancedHorizontalSwing,LEGS,tensor
from wave_math import Swing

@dataclass(frozen=True)
class PairMotionConfig(WaveConfig):
    pair_order:tuple=(('lm','rm'),('lf','rr'),('lr','rf'))
    max_translation_mps:float=.01
    minimum_support_margin_m:float=.05

@dataclass
class FootCycle:
    leg:int
    current_leg:object=None
    mode:str='unloading'
    swing:object=None
    landing:object=None
    flight_seen:bool=False
    flight_count:int=0
    contact_count:int=0
    raw_force_free_runs:int=0
    raw_force_free_samples:int=0
    unqualified_contact_returns:int=0
    last_unqualified_return_time:object=None
    last_unqualified_run_samples:int=0
    last_unqualified_lift_m:object=None
    flight_baseline_z:object=None
    flight_peak_z:object=None
    descent_seen:bool=False
    landing_contact_gap_steps:int=0
    landing_trigger_s:object=None
    landing_contact_origin:object=None
    landing_preload_world_m:object=None
    landing_endpoint_correction_m:object=None
    landing_original_contact_error_m:object=None
    landing_target_excursion_m:object=None
    hold_until:float=0.

def curve_state(curve):
    if curve is None:return None
    d={'type':'advanced' if isinstance(curve,AdvancedHorizontalSwing) else 'swing','t0':curve.t0,'duration':curve.duration,'end':curve.end.copy(),'lift':curve.lift,'coeff':curve.coeff.copy()}
    if isinstance(curve,AdvancedHorizontalSwing):d.update(horizontal_fraction=curve.horizontal_fraction,horizontal_coeff=curve.horizontal.coeff.copy())
    return d

def foot_state(foot):
    if foot is None:return None
    return {k:(curve_state(v) if k in ('swing','landing') else copy.deepcopy(v)) for k,v in vars(foot).items()}

class PairContactReference(WaveContactReference):
    STATE_VERSION='pair_contact_motion_v001'
    def __init__(self,joint_names_runtime,config=PairMotionConfig()):
        if config.pair_order!=(('lm','rm'),('lf','rr'),('lr','rf')):raise ValueError('Exact three opposing-pair order required')
        if (config.swing_s!=2. or config.lift_m!=.007 or config.contact_hold_s!=.30 or config.horizontal_duration_fraction!=.80
                or not .01<=config.max_translation_mps<=.02 or config.minimum_support_margin_m!=.05
                or config.reference_velocity_rad_s!=1.75 or config.reference_acceleration_rad_s2!=6. or config.joint_margin_rad!=.02):
            raise ValueError('Keep declared2s/7mm geometry, target bounds and50mm support; speed comparison must be explicit .01..02')
        if asdict(config)!=asdict(PairMotionConfig(max_translation_mps=config.max_translation_mps)):
            raise ValueError('All inherited contact/landing/stop thresholds must remain exact; only declared forward cap may vary')
        super().__init__(joint_names_runtime,config)
    def reset(self,snapshot):
        if not self._read(snapshot)['contact'].all():raise ValueError('Paired reset requires all six measured distal supports')
        self.active_pair=None;self.foot_cycles=[None]*6;self.pair_index=0;self.completed_pairs=0
        self._initial_snapshot=copy.deepcopy(snapshot)
        return super().reset(snapshot)
    def _support(self,measured,exclude=None):
        excluded=tuple(exclude or ());ids=[i for i in range(6) if i not in excluded]
        if not measured['contact'][ids].all():raise ValueError('A required measured retained support is missing')
        if len(ids)<4:raise ValueError('Four measured retained supports required')
        com=self._com(self._leg(measured['joint_position_rad']),measured['position_world_m'],measured['rotation_world_from_body'])
        margin=self.helper.support_margin(measured['contact_point_world_m'],ids,com)
        if margin<self.cfg.minimum_support_margin_m:raise ValueError('Measured paired support margin below50mm')
        return margin
    def _update_foot(self,foot,measured,dt):
        missing=False
        i = foot.current_leg
        actual_z = float(measured['reference_point_world_m'][i, 2])
        actual_speed = np.linalg.norm(measured['reference_point_velocity_world_mps'][i])
        # A zero-force interval during unloading is not yet a measured
        # airborne step. Keep raw evidence separate from qualification.
        if not measured['contact'][i]:
            if foot.flight_count == 0:foot.raw_force_free_runs += 1
            foot.raw_force_free_samples += 1
            foot.flight_count += 1; foot.contact_count = 0
            foot.flight_peak_z = max(foot.flight_peak_z, actual_z)
            if (foot.flight_count >= 2
                    and foot.flight_peak_z-foot.flight_baseline_z >= self.cfg.minimum_measured_lift_m):
                foot.flight_seen = True
                if foot.landing is None:foot.mode = 'swing'
        elif not foot.flight_seen:
            if foot.flight_count:
                foot.unqualified_contact_returns += 1
                foot.last_unqualified_return_time = self.time
                foot.last_unqualified_run_samples = foot.flight_count
                foot.last_unqualified_lift_m = foot.flight_peak_z-foot.flight_baseline_z
            # Separate off/on blips cannot accumulate into qualified
            # flight or retain an earlier unqualified height peak.
            foot.flight_count = 0
            foot.flight_baseline_z = actual_z; foot.flight_peak_z = actual_z
        else:
            # The qualification latch survives contact; this counter
            # always describes the current raw consecutive off run.
            foot.flight_count = 0
        # No extra tuning parameter or unlimited unloading grace period:
        # the existing planned apex is the latest qualification time.
        if not foot.flight_seen and self.time >= foot.swing.t0+.5*foot.swing.duration:
            raise ValueError('Liftoff not observed with measured2mm clearance by swing apex')
        if foot.flight_seen:
            foot.flight_peak_z = max(foot.flight_peak_z, actual_z)
            if (self.time >= foot.swing.t0+.5*foot.swing.duration
                    and actual_z < foot.flight_peak_z-self.cfg.descent_drop_from_peak_m
                    and measured['reference_point_velocity_world_mps'][i, 2] < -.0001):
                foot.descent_seen = True
        if measured['contact'][i] and foot.flight_seen and foot.landing is None:
            lift = foot.flight_peak_z-foot.flight_baseline_z
            if (self.time < foot.swing.t0+.5*foot.swing.duration or not foot.descent_seen
                    or lift < self.cfg.minimum_measured_lift_m):
                raise ValueError('Returned contact lacks measured2mm flight, passed apex and actual descent')
            if actual_speed > .04:
                raise ValueError('Returned contact speed exceeds bounded landing admission')
            # Start at the EXISTING virtual foot P/V/A, never at measured
            # position. Preserve virtual XY to avoid a lateral target
            # snap or restoring obsolete horizontal standing preload.
            p0, v0, a0 = foot.swing.sample(self.time)
            endpoint = p0.copy()
            endpoint[2] = actual_z+self.preload_world[i, 2]
            original_error = float(np.linalg.norm(foot.swing.end-measured['reference_point_world_m'][i]-self.preload_world[i]))
            correction = float(np.linalg.norm(endpoint-foot.swing.end))
            if max(correction,original_error) > self.cfg.maximum_touchdown_error_m:
                raise ValueError('Measured landing endpoint exceeds12mm correction bound')
            duration = min(self.cfg.landing_blend_s,max(self.cfg.minimum_landing_blend_s,foot.swing.end_time-self.time))
            foot.landing = Swing(self.time,duration,p0,endpoint,0.,v0,a0)
            excursion = max(float(np.linalg.norm(foot.landing.sample(t)[0]-p0))
                for t in np.linspace(self.time,foot.landing.end_time,51))
            if excursion > self.cfg.maximum_touchdown_error_m:
                raise ValueError('Landing reference excursion exceeds12mm bound')
            foot.landing_original_contact_error_m = original_error
            foot.landing_target_excursion_m = excursion
            foot.landing_trigger_s = self.time
            foot.landing_contact_origin = measured['reference_point_world_m'][i].copy()
            foot.landing_preload_world_m = endpoint-foot.landing_contact_origin
            foot.landing_endpoint_correction_m = correction
            foot.landing_contact_gap_steps = 0; foot.contact_count = 0
            foot.mode = 'landing_blend'
        trajectory = foot.landing if foot.landing is not None else foot.swing
        end = trajectory.end_time
        if foot.landing is not None:
            if measured['contact'][i]:
                foot.landing_contact_gap_steps = 0
                if np.linalg.norm(measured['reference_point_world_m'][i]-foot.landing_contact_origin) > self.cfg.maximum_touchdown_error_m:
                    raise ValueError('Measured landing foot moved beyond bounded contact region')
                if self.time >= end-1e-8:
                    # Stable measured support only; a timer/first touch
                    # alone never completes the step or updates anchors.
                    new_preload = foot.landing.end-measured['reference_point_world_m'][i]
                    consistency_error = np.linalg.norm(new_preload-foot.landing_preload_world_m)
                    if (consistency_error > self.cfg.maximum_touchdown_error_m or actual_speed > .04
                            or np.linalg.norm(new_preload) > self.cfg.maximum_initial_preload_point_offset_m):
                        raise ValueError('Landing completion disagrees with measured position/speed')
                    foot.contact_count += 1
                    if foot.contact_count >= self.cfg.contact_confirm_steps:
                        self.anchors[i] = foot.landing.end.copy()
                        self.measured_anchors[i] = measured['reference_point_world_m'][i].copy()
                        self.preload_world[i] = self.anchors[i]-self.measured_anchors[i]
                        foot.current_leg = None; foot.swing = None; foot.landing = None
                        foot.mode = 'contact_hold'; foot.hold_until = self.time+self.cfg.contact_hold_s
                        self.touchdowns += 1
            else:
                foot.landing_contact_gap_steps += 1; foot.contact_count = 0
                if foot.landing_contact_gap_steps*dt > self.cfg.maximum_landing_contact_gap_s+1e-9:
                    raise ValueError('Landing contact loss exceeds bounded100ms reacquisition interval')
            if foot.current_leg is not None:
                foot.mode = 'landing_blend' if self.time < end else 'awaiting_landing_support'
        if foot.current_leg is not None and self.time >= end:
            if foot.landing is None:foot.mode = 'awaiting_contact'
            if not measured['contact'][i]:
                missing = True
            if not foot.flight_seen or self.time-end > self.cfg.max_touchdown_delay_s:
                raise ValueError('Liftoff not observed or measured touchdown missing')
        return missing
    def _output(self,q,v,a,measured):
        body=self._body()
        state={'version':self.STATE_VERSION,'mode':self.mode,'active_pair':None if self.active_pair is None else [LEGS[i] for i in self.active_pair],
            'pair_order':self.cfg.pair_order,'next_pair_index':self.pair_index,'completed_pairs':self.completed_pairs,
            'foot_cycles':[foot_state(f) for f in self.foot_cycles],
            'time_s':self.time,'desired_position_world_m':self.position.copy(),'desired_yaw_delta_rad':self.yaw,'initial_rotation_world_from_body':self.R0.copy(),
            'command_filter_velocity':self.command.copy(),'command_filter_rate':self.command_rate.copy(),
            'requested':self.requested.copy(),'command_target':self.command_target.copy(),'derating_factor':self.factor,
            'reference_anchors_world_m':self.anchors.copy(),'measured_anchors_world_m':self.measured_anchors.copy(),'preload_world_m':self.preload_world.copy(),'initial_joint_preload_rad':self.preload_q.copy(),
            'neutral_reference_toes_body_m':self.neutral.copy(),'joint_lower_rad':self.lower.copy(),'joint_upper_rad':self.upper.copy(),
            'q_leg_major':self.q.copy(),'v_leg_major':self.v.copy(),'hold_until_s':self.hold_until,'liftoffs':self.liftoffs,'confirmed_touchdowns':self.touchdowns,
            'stop_requested_time_s':self.stop_requested_time,'reference_quiet_time_s':self.reference_quiet_time,'failure':self.failure}
        return {'valid':np.array([self.failure is None]),'q_ref':None if self.failure else self._runtime(q)[None],'v_ref':None if self.failure else self._runtime(v)[None],'a_ref':None if self.failure else self._runtime(a)[None],
            'target_time_s':self.time,'joint_names_runtime':self.names,'failure_reason':self.failure,'state':state,
            'requested_command':self.requested.copy(),'admitted_target_command':self.command_target.copy(),'admitted_command':self.command.copy(),'command_derating_factor':self.factor,
            'actual_measured_command':np.array([-measured['velocity_body_mps'][1],measured['velocity_body_mps'][0],measured['gyro_body_rad_s'][2]]),
            'diagnostics':{**self.last_diagnostics,'configuration':asdict(self.cfg),'pose_written_to_robot':False,'physics_qualified':False,'PPO_packet_846_849_compatible':False}}
    def step(self,snapshot,requested_forward_left_yaw,dt=.02):
        if not self.ready:raise RuntimeError('Reset from settled measured state first')
        m=self._read(snapshot)
        if self.failure:return self._output(self.q,self.v,np.zeros_like(self.q),m)
        try:
            if dt!=.02 or abs(float(m['time_s'])-self.time)>1e-6:raise ValueError('Time-aligned50Hz measured snapshot required')
            if np.max(abs(m['joint_target_rad']-self._runtime(self.q)))>2e-7:raise ValueError('Executed target differs from previous paired reference')
            requested=np.asarray(requested_forward_left_yaw,dtype=float)
            if requested.shape!=(3,) or not np.isfinite(requested).all() or requested[1]!=0 or requested[2]!=0 or not 0<=requested[0]<=self.cfg.max_translation_mps:
                raise ValueError('This version admits only explicitly bounded forward or stop commands')
            if not requested.any() and self.requested.any():self.stop_requested_time=self.time;self.reference_quiet_time=None
            if requested.any():self.stop_requested_time=None;self.reference_quiet_time=None
            self.requested=requested.copy();target=requested.copy();factor=1.
            if m['terminal'] or m['base_contact'] or any(m[k].any() for k in ('shaft_contact','coxa_contact','femur_contact')):raise ValueError('Measured terminal or nonfoot contact')
            margin=self._support(m,self.active_pair)
            body_error=float(np.linalg.norm(m['position_world_m'][:2]-self.position[:2]))
            if body_error>self.cfg.maximum_body_tracking_error_m:raise ValueError('Actual body tracking exceeds declared35mm bound')
            retained=[i for i in range(6) if self.active_pair is None or i not in self.active_pair]
            drift=np.linalg.norm(m['reference_point_world_m']-self.measured_anchors,axis=-1)
            if drift[retained].max()>self.cfg.maximum_stance_drift_m:raise ValueError('Actual retained toe drift exceeds20mm')
            if self.active_pair is not None:
                missing=False
                for i in self.active_pair:
                    f=self.foot_cycles[i]
                    if f.current_leg is not None:missing=self._update_foot(f,m,dt) or missing
                for i in self.active_pair:
                    if self.foot_cycles[i].current_leg is None and not m['contact'][i]:
                        raise ValueError('Confirmed pair foot lost measured support before pair handoff')
                if missing:target=np.zeros(3);factor=0.
                if all(self.foot_cycles[i].current_leg is None for i in self.active_pair):
                    self.hold_until=max(self.foot_cycles[i].hold_until for i in self.active_pair)
                    self.active_pair=None;self.completed_pairs+=1;self.mode='contact_hold'
                else:self.mode='paired_motion'
            if self.active_pair is None and self.time>=self.hold_until and target[0]>1e-6:
                pair=tuple(LEGS.index(n) for n in self.cfg.pair_order[self.pair_index]);self._support(m,pair)
                if not m['contact'].all():raise ValueError('New pair requires all six measured supports')
                cycle=3*(self.cfg.swing_s+self.cfg.contact_hold_s);horizon=self.cfg.swing_s+.5*(cycle-self.cfg.swing_s)
                p,R=self._predict(target,horizon)
                for i in pair:
                    end=R@self.neutral[i]+p;end[2]=self.anchors[i,2]
                    curve=AdvancedHorizontalSwing(self.time,self.cfg.swing_s,self.anchors[i],end,self.cfg.lift_m,self.cfg.horizontal_duration_fraction)
                    z=float(m['reference_point_world_m'][i,2]);self.foot_cycles[i]=FootCycle(i,current_leg=i,swing=curve,flight_baseline_z=z,flight_peak_z=z)
                self.active_pair=pair;self.pair_index=(self.pair_index+1)%3;self.liftoffs+=2;self.mode='paired_unloading'
            self.command_target=target.copy();self.factor=factor
            finite_stop=(not requested.any() and self.active_pair is None and np.max(abs(self.command))<=self.cfg.stop_command_tolerance and np.max(abs(self.command_rate))<=self.cfg.stop_command_rate_tolerance)
            if finite_stop:self.command[:]=0.;self.command_rate[:]=0.
            self.position,self.yaw,self.command,self.command_rate=self._advance(target,dt);self.time+=dt
            points=self.anchors.copy()
            if self.active_pair is not None:
                for i in self.active_pair:
                    f=self.foot_cycles[i]
                    if f.current_leg is not None:points[i]=(f.landing if f.landing is not None else f.swing).sample(self.time)[0]
            local=(self._body().rotation_wb.T@(points-self.position).T).T;ik=self.g.ik(tensor(local[None]))
            if not bool(ik['valid'].all()):raise ValueError('Paired foot target unreachable; no IK clipping')
            q=ik['q_checked'][0].numpy()
            if self.active_pair is None and np.max(abs(self.command))<1e-14 and np.max(abs(self.command_rate))<1e-12:q=self.q.copy()
            v=(q-self.q)/dt;a=(v-self.v)/dt;self._bounds(q,v,a)
            self.last_diagnostics={'reference_point_world_m':points.copy(),'measured_support_margin_m':margin,'body_tracking_error_m':body_error,'stance_drift_m':drift,'max_reference_velocity_rad_s':float(abs(v).max()),'max_reference_acceleration_rad_s2':float(abs(a).max()),'minimum_joint_margin_rad':float(np.minimum(q-self.lower,self.upper-q).min())}
            self.q=q;self.v=v
            if self.active_pair is None and not requested.any():
                self.mode='reference_quiet_hold' if finite_stop else 'stopping_reference_motion'
                if finite_stop and self.reference_quiet_time is None:self.reference_quiet_time=self.time
            return self._output(q,v,a,m)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            self.failure=str(exc);return self._output(self.q,self.v,np.zeros_like(self.q),m)
