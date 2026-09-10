"""Synthetic CPU support-envelope supervisor; no actor/physics/hardware writes."""
from dataclasses import dataclass
from pathlib import Path
import sys, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'isaaclab'))
from terrain_readiness import terrain_channels

@dataclass
class MapSnapshot:
    height_m: np.ndarray
    variance_m2: np.ndarray
    observed: np.ndarray
    capture_time_s: np.ndarray
    origin_xy_m: tuple
    resolution_m: float
    world_frame: str
    clock_id: str
    query_time_s: float
    calibration_id: str
    source_mode: str

    def validate(self):
        shape=np.shape(self.height_m)
        if len(shape)!=2 or any(np.shape(v)!=shape for v in (self.variance_m2,self.observed,self.capture_time_s)):
            raise ValueError('Map arrays must share HxW shape')
        if np.asarray(self.observed).dtype!=bool or min(shape)<=0:
            raise ValueError('Nonempty boolean observation grid required')
        if not self.world_frame or not self.clock_id or not self.calibration_id:
            raise ValueError('Explicit frame, clock and calibration identifiers required')
        if self.source_mode not in ('ideal_teacher','synthetic_interface','simulated_depth','simulated_lidar','fused_simulated_map'):
            raise ValueError('Prototype has no real-sensor qualification')
        if np.shape(self.origin_xy_m)!=(2,) or not np.isfinite(self.origin_xy_m).all() or not math.isfinite(self.query_time_s) or not math.isfinite(self.resolution_m) or self.resolution_m<=0:
            raise ValueError('Invalid map grid/time')
        return shape

    def centers(self):
        shape=self.validate();a,b=np.indices(shape)
        return np.stack((a,b),-1)*self.resolution_m+np.asarray(self.origin_xy_m)+.5*self.resolution_m

@dataclass
class GeometryEligibility:
    support_geometry: np.ndarray
    inside_course: np.ndarray
    avoidance_hazard: np.ndarray
    geometry_hit: np.ndarray
    world_frame: str
    clock_id: str
    query_time_s: float
    provenance: str
    source_mode: str

@dataclass
class RequiredEnvelope:
    required: np.ndarray
    latest_use_time_s: np.ndarray
    outside_map: bool
    world_frame: str
    clock_id: str
    controller_id: str
    interpolation: str
    synthetic_only: bool=True


def map_eligibility(snapshot, *, reference_surface_height_m, inside_course, max_drop_m=.025,
                    max_step_m=.025, max_slope_deg=5., max_local_span_m=.015):
    """Conservative synthetic-map geometry screen, not learned traversability.

    The reference support surface is supplied explicitly, never inferred as0.
    Unknown3x3 neighborhoods are not filled; sharp boundaries remain ineligible.
    Frame-matched inside_course is an explicit mission boundary, not perception.
    """
    shape=snapshot.validate();inside=np.asarray(inside_course)
    reference=np.broadcast_to(np.asarray(reference_surface_height_m,float),shape)
    if inside.shape!=shape or inside.dtype!=bool or not np.isfinite(reference).all():
        raise ValueError('Explicit finite reference surface and boolean course grid required')
    if not all(math.isfinite(x) and x>0 for x in (max_drop_m,max_step_m,max_slope_deg,max_local_span_m)):
        raise ValueError('Positive declared geometry bounds required')
    channels=terrain_channels(snapshot.height_m,snapshot.variance_m2,snapshot.observed,snapshot.capture_time_s,now_s=snapshot.query_time_s)
    usable=channels[...,1].astype(bool);h=np.asarray(snapshot.height_m)
    hit=np.isfinite(h)&np.asarray(snapshot.observed)
    hazard=hit&((h<reference-max_drop_m)|(h>reference+max_step_m))
    neighborhood=np.zeros(shape,bool)
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            if not usable[i-1:i+2,j-1:j+2].all():continue
            block=h[i-1:i+2,j-1:j+2]
            gradient=np.array([h[i+1,j]-h[i-1,j],h[i,j+1]-h[i,j-1]])/(2*snapshot.resolution_m)
            neighborhood[i,j]=(np.ptp(block)<=max_local_span_m and np.linalg.norm(gradient)<=math.tan(math.radians(max_slope_deg)))
    return GeometryEligibility(hit&inside&~hazard&neighborhood,inside,hazard,hit,snapshot.world_frame,snapshot.clock_id,snapshot.query_time_s,
        'synthetic map height/drop/step/neighborhood bounds; no material or3D clearance proof',snapshot.source_mode)


def teacher_eligibility(snapshot, query_engine, fixture_index, *, course_origin=(0,0,0),course_yaw_rad=0.,boundary_margin_m=0.):
    """Explicit privileged truth adapter; forbidden for sensor-mode snapshots."""
    if snapshot.source_mode!='ideal_teacher':raise ValueError('Teacher geometry cannot fill a sensor-mode eligibility layer')
    xy=snapshot.centers();points=np.concatenate((xy,np.zeros((*xy.shape[:2],1))),-1)
    query=query_engine.query_world(points.reshape(-1,3),fixture_index,course_origins_world=course_origin,
        course_yaw_rad=course_yaw_rad,boundary_margin_m=boundary_margin_m)
    shape=xy.shape[:2]
    field=lambda name:getattr(query,name).detach().cpu().numpy().reshape(shape)
    return GeometryEligibility(field('support_geometry'),field('inside_course'),field('avoidance_hazard'),field('geometry_hit'),
        snapshot.world_frame,snapshot.clock_id,snapshot.query_time_s,'TerrainSupportQueries exact triangles; privileged teacher only','ideal_teacher')


def swept_foot_envelope(snapshot, *, poses_xy_yaw, relative_times_s, foot_centers_body_xy,
                        pad_radius_m, uncertainty_radius_m, world_frame, clock_id,controller_id):
    """Conservative raster of supplied piecewise-linear XY/unwrapped-yaw motion.

    All supplied body-fixed pad locations are considered possible support for the
    whole path. This synthetic bound is NOT an executable gait/braking predictor;
    real integration must supply controller-validated support trajectories.
    """
    snapshot.validate();poses=np.asarray(poses_xy_yaw,float);times=np.asarray(relative_times_s,float);feet=np.asarray(foot_centers_body_xy,float)
    if world_frame!=snapshot.world_frame or clock_id!=snapshot.clock_id:raise ValueError('Envelope frame/clock mismatch')
    if poses.ndim!=2 or poses.shape[1]!=3 or len(poses)<1 or times.shape!=(len(poses),) or feet.ndim!=2 or feet.shape[1]!=2 or not len(feet):
        raise ValueError('Nonempty pose/time and pad-center arrays required')
    if not np.isfinite(poses).all() or not np.isfinite(times).all() or not np.isfinite(feet).all() or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Finite poses and increasing relative times beginning0 required')
    if not controller_id or min(pad_radius_m,uncertainty_radius_m)<0 or not np.isfinite([pad_radius_m,uncertainty_radius_m]).all():raise ValueError('Named controller and finite nonnegative footprint bounds required')
    xy=snapshot.centers();required=np.zeros(xy.shape[:2],bool);latest=np.full(required.shape,np.nan);outside=False
    lo=np.asarray(snapshot.origin_xy_m);hi=lo+np.array(required.shape)*snapshot.resolution_m
    rmax=np.linalg.norm(feet,axis=1).max();sample_step=snapshot.resolution_m/4
    samples=[]
    if len(poses)==1:samples=[(poses[0],0.,0.,0.)]
    for k in range(len(poses)-1):
        # Yaw is deliberately unwrapped: a2pi turn is not mistaken for no motion.
        distance=np.linalg.norm(poses[k+1,:2]-poses[k,:2])+rmax*abs(poses[k+1,2]-poses[k,2])
        n=max(1,int(math.ceil(distance/sample_step)))
        for j in range(n+1):
            u=j/n;sample=(1-u)*poses[k]+u*poses[k+1]
            time=(1-u)*times[k]+u*times[k+1]
            samples.append((sample,time,.5*distance/n,.5*(times[k+1]-times[k])/n))
    for pose,time,spatial_extra,time_extra in samples:
        c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,-s],[s,c]])
        centers=feet@rotation.T+pose[:2]
        physical_radius=pad_radius_m+uncertainty_radius_m+spatial_extra
        raster_radius=physical_radius+snapshot.resolution_m/math.sqrt(2)
        for center in centers:
            outside |= bool((center-physical_radius<lo).any() or (center+physical_radius>hi).any())
            mask=np.linalg.norm(xy-center,axis=-1)<=raster_radius
            required|=mask
            use=snapshot.query_time_s+min(times[-1],time+time_extra)
            latest[mask]=np.fmax(latest[mask],use)
    return RequiredEnvelope(required,latest,outside,world_frame,clock_id,controller_id,
        'conservative sampled piecewise-linear XY/unwrapped-yaw; full-pad+cell+motion bound',True)


def assess(snapshot,geometry,envelope, *, requested_twist, max_age_s=.25,max_std_m=.015):
    """Decision-only: never emits joint actions or claims a physically safe stop.

    Existing evidence must remain usable at the last required future use; no
    future sensor reacquisition is assumed. A failed result requests a controller
    stop/replan and explicitly does not claim that stop is physically feasible.
    """
    shape=snapshot.validate();req=np.asarray(envelope.required)
    if req.shape!=shape or req.dtype!=bool or np.shape(envelope.latest_use_time_s)!=shape:raise ValueError('Envelope shape/mask mismatch')
    if any(np.shape(getattr(geometry,k))!=shape or np.asarray(getattr(geometry,k)).dtype!=bool for k in ('support_geometry','inside_course','avoidance_hazard','geometry_hit')):raise ValueError('Geometry boolean grids must match map')
    if (geometry.world_frame,geometry.clock_id)!=(snapshot.world_frame,snapshot.clock_id) or (envelope.world_frame,envelope.clock_id)!=(snapshot.world_frame,snapshot.clock_id):raise ValueError('Wrong frame/clock; no silent transform')
    if geometry.query_time_s!=snapshot.query_time_s or geometry.source_mode!=snapshot.source_mode:raise ValueError('Geometry time/provenance differs from map')
    twist=np.asarray(requested_twist,float)
    if twist.shape!=(3,) or not np.isfinite(twist).all():raise ValueError('Finite forward,left,yaw twist required')
    now=snapshot.query_time_s
    channels=terrain_channels(snapshot.height_m,snapshot.variance_m2,snapshot.observed,snapshot.capture_time_s,now_s=now,max_age_s=max_age_s,max_std_m=max_std_m)
    usable=channels[...,1].astype(bool)
    age_at_use=envelope.latest_use_time_s-np.asarray(snapshot.capture_time_s)
    future=usable&np.isfinite(age_at_use)&(age_at_use>=0)&(age_at_use<=max_age_s)&(envelope.latest_use_time_s>=now)
    eligible=geometry.geometry_hit&geometry.inside_course&~geometry.avoidance_hazard&geometry.support_geometry
    failures=dict(unobserved=req&~snapshot.observed,unusable_now=req&~usable,
        expires_before_required_use=req&usable&~future,ineligible_support=req&~eligible,
        outside_course=req&~geometry.inside_course,known_hazard=req&geometry.avoidance_hazard)
    counts={k:int(v.sum()) for k,v in failures.items()}
    reasons=[key for key,count in counts.items() if count]
    if not req.any():reasons.append('empty_required_envelope')
    if envelope.outside_map:reasons.append('envelope_outside_map')
    permit=not reasons
    valid_until=float(np.min(np.asarray(snapshot.capture_time_s)[req]+max_age_s)) if permit else now
    return dict(schema='synthetic_support_supervisor_v1',permit_candidate=permit,
        decision='permit_candidate' if permit else 'request_supported_stop_or_replan',
        stop_feasibility_proven=False,reason_codes=reasons,reason_cell_counts=counts,
        requested_twist=twist.tolist(),admitted_twist=twist.tolist() if permit else [0.,0.,0.],
        required_cells=int(req.sum()),observed_required_cells=int((req&snapshot.observed).sum()),
        usable_now_required_cells=int((req&usable).sum()),eligible_required_cells=int((req&eligible).sum()),
        future_usable_required_cells=int((req&future).sum()),valid_until_s=valid_until,
        source_mode=snapshot.source_mode,controller_id=envelope.controller_id,synthetic_only=True)


def snapshot_from_local_map(model, *, query_time_s,clock_id,calibration_id,source_mode):
    """Copy the existing LocalHeightMap arrays without inventing missing height."""
    snapshot=MapSnapshot(np.array(model.height,copy=True),np.array(model.variance,copy=True),
        np.array(model.observed,copy=True),np.array(model.capture,copy=True),tuple(model.origin),
        float(model.resolution),model.world_frame,clock_id,query_time_s,calibration_id,source_mode)
    snapshot.validate()
    return snapshot
