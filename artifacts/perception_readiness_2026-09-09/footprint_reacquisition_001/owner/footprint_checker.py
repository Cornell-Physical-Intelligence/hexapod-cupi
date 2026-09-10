"""Decision-only synthetic footprint evidence ledger; never emits robot actions."""
from dataclasses import dataclass
from pathlib import Path
import math,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from perception_replay import LocalHeightMap,WorldPoints
from terrain_readiness import terrain_channels
LEASE_S=.25
MAX_STD_M=.015

@dataclass(frozen=True)
class SensorSchedule:
    sensor_id:str
    calibration_id:str
    epoch_s:float
    period_s:float
    assumed_latency_s:float

@dataclass
class FrameReceipt:
    sensor_id:str
    sequence:int
    capture_time_s:float
    receive_time_s:float
    clock_id:str
    cloud:WorldPoints
    optical_mask:np.ndarray
    mesh_clear_mask:np.ndarray
    origin_ambiguous_mask:np.ndarray
    evidence_id:str

@dataclass(frozen=True)
class DriftUncertainty:
    """Explicit synthetic error assumptions; no real estimator is qualified."""
    additional_z_sigma_m:float
    z_sigma_growth_m_per_s:float
    xy_sigma_m:float
    xy_sigma_growth_m_per_s:float

@dataclass(frozen=True)
class FootstepProposal:
    proposal_id:str
    foot_id:str
    center_world_m:tuple
    proxy_radius_m:float
    created_time_s:float
    requested_use_time_s:float
    world_frame:str
    clock_id:str

@dataclass
class SupportGeometry:
    geometry_hit:np.ndarray
    inside_course:np.ndarray
    avoidance_hazard:np.ndarray
    support_geometry:np.ndarray
    world_frame:str
    clock_id:str
    query_time_s:float
    provenance:str
    source_mode:str

@dataclass(frozen=True)
class PlantedContact:
    foot_id:str
    time_s:float
    point_world_m:tuple
    load_bearing:bool
    point_valid:bool
    world_frame:str
    clock_id:str

class FootprintEvidenceLedger:
    def __init__(self,*,origin_xy,size_xy,resolution_m,world_frame,clock_id,sensors):
        if not world_frame or not clock_id:raise ValueError('Explicit world frame and acquisition clock required')
        self.map=LocalHeightMap(origin_xy=origin_xy,size_xy=size_xy,resolution_m=resolution_m,world_frame=world_frame)
        self.clock_id=clock_id;self.world_frame=world_frame;self.sensors={s.sensor_id:s for s in sensors}
        if not self.sensors or len(self.sensors)!=len(sensors):raise ValueError('Unique sensor schedules required')
        for s in sensors:
            if not s.sensor_id or not s.calibration_id or not np.isfinite([s.epoch_s,s.period_s,s.assumed_latency_s]).all() or s.epoch_s<0 or s.period_s<=0 or s.assumed_latency_s<0:
                raise ValueError('Finite identified acquisition schedule required')
        self.last_now=-math.inf;self.receipts={};self.frames=[]
    def _time(self,now):
        if not math.isfinite(now) or now<0 or now<self.last_now-1e-10:raise ValueError('Noncausal or invalid processing/query time')
    def ingest(self,frame,*,now_s):
        """Only received, frame/clock/calibration-matched sensor points enter map."""
        self._time(now_s)
        if frame.sensor_id not in self.sensors:raise ValueError('Unknown sensor')
        sensor=self.sensors[frame.sensor_id];cloud=frame.cloud
        if frame.clock_id!=self.clock_id or cloud.world_frame!=self.world_frame:raise ValueError('Frame/clock mismatch')
        if cloud.source_sensor!=frame.sensor_id or cloud.calibration_id!=sensor.calibration_id or not frame.evidence_id:raise ValueError('Sensor/calibration/evidence identity mismatch')
        if type(frame.sequence) is not int or frame.sequence<0:raise ValueError('Nonnegative integer frame sequence required')
        if not np.isfinite([frame.capture_time_s,frame.receive_time_s]).all() or not 0<=frame.capture_time_s<=frame.receive_time_s<=now_s:
            raise ValueError('Future or noncausal frame times')
        previous=self.receipts.get(frame.sensor_id)
        if previous and (frame.sequence<=previous['sequence'] or frame.capture_time_s<=previous['capture_time_s']):raise ValueError('Duplicate or reordered acquisition frame')
        masks=[np.asarray(v) for v in [frame.optical_mask,frame.mesh_clear_mask,frame.origin_ambiguous_mask]]
        if any(v.shape!=self.map.shape or v.dtype!=bool for v in masks):raise ValueError('Explicit optical/mesh/ambiguity masks must match grid')
        visible=masks[0]&masks[1]&~masks[2]
        points=np.asarray(cloud.points_m);variance=np.asarray(cloud.variance_z_m2);capture=np.asarray(cloud.capture_time_s)
        if points.shape!=(len(variance),3) or capture.shape!=variance.shape or not np.isfinite(points).all() or not np.isfinite(variance).all() or not np.isfinite(capture).all() or (variance<0).any():raise ValueError('Malformed world points or uncertainty')
        if (capture<0).any() or (capture>frame.capture_time_s).any() or (capture>frame.receive_time_s).any():raise ValueError('Future/noncausal per-point acquisition time')
        ij=np.floor((points[:,:2]-self.map.origin)/self.map.resolution).astype(int)
        inside=((ij>=0)&(ij<self.map.shape)).all(-1);ids=ij[inside]
        if not visible[ids[:,0],ids[:,1]].all():raise ValueError('Point cloud includes an optically invalid, occluded or ambiguous cell')
        # Do not insert already expired points, including delayed old frames.
        fresh=now_s-capture<=LEASE_S
        accepted=WorldPoints(points[fresh],variance[fresh],capture[fresh],cloud.world_frame,cloud.source_sensor,cloud.calibration_id,dict(cloud.diagnostics))
        result=self.map.integrate(accepted)
        receipt=dict(sensor_id=frame.sensor_id,sequence=frame.sequence,capture_time_s=frame.capture_time_s,receive_time_s=frame.receive_time_s,evidence_id=frame.evidence_id,accepted_points=int(fresh.sum()),expired_points=int((~fresh).sum()),visible_at_capture=visible.copy(),optical_at_capture=masks[0].copy(),mesh_clear_at_capture=masks[1].copy())
        self.receipts[frame.sensor_id]=receipt;self.frames.append({k:v for k,v in receipt.items() if not isinstance(v,np.ndarray)});self.last_now=now_s
        return result
    def sensor_health(self,now_s):
        result={}
        for key,s in self.sensors.items():
            expected=math.floor((now_s-s.assumed_latency_s-s.epoch_s)/s.period_s+1e-8)
            latest_expected=None if expected<0 else s.epoch_s+expected*s.period_s
            actual=self.receipts.get(key)
            missing=latest_expected is not None and (actual is None or actual['capture_time_s']<latest_expected-1e-8)
            result[key]=dict(missing_latest_scheduled_frame=missing,latest_expected_capture_s=latest_expected,last_received_capture_s=None if actual is None else actual['capture_time_s'],schedule_is_synthetic_assumption=True)
        return result
    def _footprint(self,proposal,radius):
        center=np.asarray(proposal.center_world_m,float)
        xy=np.moveaxis(np.indices(self.map.shape),0,-1)*self.map.resolution+self.map.origin+.5*self.map.resolution
        outside=bool(((center[:2]-radius<self.map.origin)|(center[:2]+radius>self.map.origin+np.array(self.map.shape)*self.map.resolution)).any())
        return np.linalg.norm(xy-center[:2],axis=-1)<=radius+self.map.resolution/math.sqrt(2),outside
    def query(self,proposal,geometry,uncertainty,*,now_s,planted_contacts=()):
        self._time(now_s)
        if (proposal.world_frame,proposal.clock_id)!=(self.world_frame,self.clock_id):raise ValueError('Proposal frame/clock mismatch')
        if not proposal.proposal_id or not proposal.foot_id or np.shape(proposal.center_world_m)!=(3,) or not np.isfinite(proposal.center_world_m).all() or not np.isfinite([proposal.proxy_radius_m,proposal.created_time_s,proposal.requested_use_time_s]).all() or proposal.proxy_radius_m<=0 or not 0<=proposal.created_time_s<=now_s<=proposal.requested_use_time_s:raise ValueError('Finite causal new-footstep proposal required')
        values=[uncertainty.additional_z_sigma_m,uncertainty.z_sigma_growth_m_per_s,uncertainty.xy_sigma_m,uncertainty.xy_sigma_growth_m_per_s]
        if not np.isfinite(values).all() or min(values)<0:raise ValueError('Explicit finite nonnegative drift/uncertainty required')
        if (geometry.world_frame,geometry.clock_id)!=(self.world_frame,self.clock_id) or geometry.query_time_s!=now_s:raise ValueError('Geometry frame/clock/time mismatch')
        if not geometry.provenance or geometry.source_mode not in ('synthetic_map_estimate','teacher_fixture_only'):raise ValueError('Explicit separate prototype support provenance required')
        fields=[np.asarray(getattr(geometry,k)) for k in ['geometry_hit','inside_course','avoidance_hazard','support_geometry']]
        if any(v.shape!=self.map.shape or v.dtype!=bool for v in fields):raise ValueError('Geometry masks must match map')
        eligible=fields[0]&fields[1]&~fields[2]&fields[3]
        horizon=proposal.requested_use_time_s-now_s
        # Worst acquisition-age registration drift is explicit, never assumed0.
        radius_now=proposal.proxy_radius_m+3*(uncertainty.xy_sigma_m+uncertainty.xy_sigma_growth_m_per_s*LEASE_S)
        radius_use=radius_now+3*uncertainty.xy_sigma_growth_m_per_s*horizon
        required_now,outside_now=self._footprint(proposal,radius_now);required_use,outside_use=self._footprint(proposal,radius_use)
        def usability(time):
            age=time-self.map.capture
            variance=self.map.variance+uncertainty.additional_z_sigma_m**2+(uncertainty.z_sigma_growth_m_per_s*np.maximum(age,0))**2
            channels=terrain_channels(self.map.height,variance,self.map.observed,self.map.capture,now_s=time,max_age_s=LEASE_S,max_std_m=MAX_STD_M)
            return channels[...,1].astype(bool),age,np.sqrt(variance)
        usable_now,age_now,std_now=usability(now_s);usable_use,age_use,std_use=usability(proposal.requested_use_time_s)
        complete_now=bool(required_now.any() and not outside_now and (usable_now&eligible)[required_now].all())
        complete_use=bool(required_use.any() and not outside_use and (usable_use&eligible)[required_use].all())
        decision='lease_covers_requested_use_time' if complete_use else 'fresh_now_only' if complete_now else 'blocked'
        reasons=dict(unobserved=required_use&~self.map.observed,stale_now=required_use&self.map.observed&(age_now>LEASE_S),expires_before_use=required_use&usable_now&~usable_use,uncertain_now=required_use&self.map.observed&(~np.isfinite(std_now)|(std_now>MAX_STD_M)),ineligible_support=required_use&~eligible,outside_course=required_use&~fields[1],known_hazard=required_use&fields[2])
        contacts=[]
        for contact in planted_contacts:
            if (contact.world_frame,contact.clock_id)!=(self.world_frame,self.clock_id) or contact.time_s!=now_s or not contact.foot_id or np.shape(contact.point_world_m)!=(3,):raise ValueError('Current contact frame/clock/time mismatch')
            valid=bool(contact.load_bearing and contact.point_valid and np.isfinite(contact.point_world_m).all())
            contacts.append(dict(foot_id=contact.foot_id,measured_contact_support=valid,point_world_m=list(map(float,contact.point_world_m)) if valid else None,updates_terrain_map=False,certifies_new_footprint=False))
        self.last_now=now_s
        capture=self.map.capture[required_use];observed=self.map.observed[required_use]
        result=dict(schema='new_footprint_evidence_checker_prototype_v1',proposal_id=proposal.proposal_id,foot_id=proposal.foot_id,query_time_s=now_s,requested_use_time_s=proposal.requested_use_time_s,decision=decision,complete_usable_eligible_now=complete_now,lease_covers_requested_use_time=complete_use,required_cells_now=int(required_now.sum()),required_cells_use=int(required_use.sum()),observed_use_cells=int(observed.sum()),usable_now_use_cells=int((usable_now&required_use).sum()),eligible_use_cells=int((eligible&required_use).sum()),usable_at_use_cells=int((usable_use&required_use).sum()),outside_map_now=outside_now,outside_map_use=outside_use,reason_cell_counts={k:int(v.sum()) for k,v in reasons.items()},minimum_capture_expiry_s=None if not observed.all() or not len(capture) else float(np.min(capture+LEASE_S)),maximum_current_age_s=None if not observed.any() else float(np.max(age_now[required_use][observed])),maximum_current_std_m=None if not observed.any() else float(np.max(std_now[required_use][observed])),raster_radius_now_m=radius_now,raster_radius_use_m=radius_use,max_capture_age_s=LEASE_S,max_std_m=MAX_STD_M,sensor_health=self.sensor_health(now_s),geometry_provenance=geometry.provenance,geometry_source_mode=geometry.source_mode,measured_current_contact_support=contacts,future_observations_assumed=False,physical_stop_or_abort_proven=False,hardware_qualified=False,terrain_traversal_qualified=False)
        result['masks']=dict(required_now=required_now,required_use=required_use,observed_ever=self.map.observed.copy(),usable_now=usable_now,usable_at_use=usable_use,eligible_geometry=eligible,**reasons)
        return result
