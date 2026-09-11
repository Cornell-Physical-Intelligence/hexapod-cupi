#!/usr/bin/env python3
"""CPU depth/point-cloud -> timestamped local terrain contract prototype.

Synthetic fixtures only. No ROS, device driver, trained perception or policy
injection. All timestamps must already share one calibrated acquisition clock.
World coordinates are a continuous local odom frame; map corrections stay out.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from experiments.terrain.tools.terrain_readiness import terrain_channels, support_region_observed
from experiments.terrain.tools.sensor_mount_math import apply_transform, look_outward, transform

ROOT=Path(__file__).resolve().parents[3]


def apply_rigid(points,matrix):
    # Explicit contraction also avoids BLAS floating-status warnings observed
    # with this host's NumPy/Accelerate build on large thin point arrays.
    return np.einsum('...i,ji->...j',points,matrix[:3,:3])+matrix[:3,3]


def validate_transform(matrix):
    a=np.asarray(matrix,dtype=float)
    if a.shape!=(4,4) or not np.isfinite(a).all():
        raise ValueError('Expected finite4x4transform')
    if not np.allclose(a[3],[0,0,0,1]) or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(a[:3,:3]),1.,atol=1e-7):
        raise ValueError('Expected a rigid right-handed transform')
    return a


@dataclass
class CameraCalibration:
    sensor_frame: str
    calibration_id: str
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    min_z_m: float
    max_z_m: float
    body_from_optical: np.ndarray
    depth_scale_m: float=1.
    rectified: bool=True
    pixel_sigma: float=.5
    calibrated: bool=False

    def __post_init__(self):
        self.body_from_optical=validate_transform(self.body_from_optical)
        values=[self.fx,self.fy,self.ppx,self.ppy,self.min_z_m,self.max_z_m,self.depth_scale_m,self.pixel_sigma]
        if not np.isfinite(values).all() or min(self.width,self.height,self.fx,self.fy,self.depth_scale_m)<=0 or not 0<self.min_z_m<self.max_z_m or self.pixel_sigma<0:
            raise ValueError('Invalid camera calibration')
        if not self.rectified or not self.calibrated or not self.calibration_id or not self.sensor_frame:
            raise ValueError('Requires identified calibrated rectified depth intrinsics; undistort externally')


@dataclass
class PoseSample:
    capture_time_s: float
    world_from_body: np.ndarray
    position_sigma_m: float=0.
    rotation_sigma_rad: float=0.


class PoseTimeline:
    def __init__(self,samples,*,clock_id='replay_clock',world_frame='odom',max_gap_s=.10):
        self.samples=list(samples);self.clock_id=clock_id;self.world_frame=world_frame;self.max_gap_s=max_gap_s
        self.times=np.array([s.capture_time_s for s in self.samples])
        if len(samples)<1 or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0) or not math.isfinite(max_gap_s) or max_gap_s<=0:
            raise ValueError('Pose times must be finite and strictly increasing')
        for s in self.samples:
            s.world_from_body=validate_transform(s.world_from_body)
            if not np.isfinite([s.position_sigma_m,s.rotation_sigma_rad]).all() or min(s.position_sigma_m,s.rotation_sigma_rad)<0:
                raise ValueError('Invalid pose uncertainty')

    def at(self,time_s):
        if not math.isfinite(time_s) or time_s<self.times[0]-1e-10 or time_s>self.times[-1]+1e-10:
            raise ValueError('No pose extrapolation')
        exact=np.flatnonzero(np.abs(self.times-time_s)<1e-10)
        if len(exact):
            return self.samples[int(exact[0])]
        hi=int(np.searchsorted(self.times,time_s));lo=hi-1
        a,b=self.samples[lo],self.samples[hi];gap=b.capture_time_s-a.capture_time_s
        if gap>self.max_gap_s+1e-10:
            raise ValueError('Pose interpolation gap exceeds limit')
        weight=(time_s-a.capture_time_s)/gap
        out=np.eye(4);out[:3,3]=(1-weight)*a.world_from_body[:3,3]+weight*b.world_from_body[:3,3]
        rotations=Rotation.from_matrix(np.stack((a.world_from_body[:3,:3],b.world_from_body[:3,:3])))
        out[:3,:3]=Slerp([a.capture_time_s,b.capture_time_s],rotations)([time_s]).as_matrix()[0]
        return PoseSample(time_s,out,max(a.position_sigma_m,b.position_sigma_m),max(a.rotation_sigma_rad,b.rotation_sigma_rad))


@dataclass
class DepthFrame:
    sensor_frame: str
    clock_id: str
    capture_time_s: float
    receive_time_s: float
    depth: np.ndarray
    valid: np.ndarray
    robot_mask: np.ndarray
    depth_sigma_m: float | np.ndarray=.003
    row_time_offsets_s: np.ndarray | None=None
    # Optical-axis depth to the closest robot surface; +inf means clear.
    # NaN/negative values are unresolved and reject the pixel.
    robot_depth_m: np.ndarray | None=None
    robot_mask_verified: bool=False


@dataclass
class WorldPoints:
    points_m: np.ndarray
    variance_z_m2: np.ndarray
    capture_time_s: np.ndarray
    world_frame: str
    source_sensor: str
    calibration_id: str
    diagnostics: dict


def transform_timed_points(points_sensor,variance_sensor,times,timeline,body_from_sensor,*,now_s,receive_time_s,
                           clock_id,source_sensor,calibration_id,max_observation_age_s=.25):
    """Shared integration seam for calibrated camera and LiDAR point clouds.

    A Mid-360 adapter can supply XYZ plus per-return acquisition times after
    clock calibration and robot-return masking. This does not generate its scan
    pattern, deskew raw packets, estimate pose or assume any ROS integration.
    """
    points=np.asarray(points_sensor,dtype=float);cov=np.asarray(variance_sensor,dtype=float);times=np.asarray(times,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or cov.shape!=(len(points),3,3) or times.shape!=(len(points),):
        raise ValueError('ExpectedNx3points,Nx3x3covariance,Ntimestamps')
    if clock_id!=timeline.clock_id:
        raise ValueError('Acquisition clock mismatch')
    if not np.isfinite([now_s,receive_time_s]).all() or receive_time_s>now_s or max_observation_age_s<=0:
        raise ValueError('Invalid receive/current time')
    extrinsic=validate_transform(body_from_sensor)
    valid=(np.isfinite(points).all(axis=1)&np.isfinite(cov).all(axis=(1,2))&np.isfinite(times)&
           (times<=receive_time_s)&(times<=now_s)&(now_s-times<=max_observation_age_s))
    # Reject negative covariance eigenvalues rather than claiming confidence.
    valid &= np.all(np.linalg.eigvalsh(np.where(np.isfinite(cov),cov,0.))>=-1e-12,axis=1)
    out=np.full_like(points,np.nan);variance=np.full(len(points),np.inf)
    no_pose=0
    for t in np.unique(times[valid]):
        ids=np.flatnonzero(valid&(times==t))
        try:
            pose=timeline.at(float(t))
        except ValueError:
            valid[ids]=False;no_pose+=len(ids);continue
        world_from_sensor=pose.world_from_body@extrinsic
        out[ids]=apply_rigid(points[ids],world_from_sensor)
        row=world_from_sensor[2,:3]
        variance[ids]=np.einsum('i,nij,j->n',row,cov[ids],row)
        lever=out[ids]-pose.world_from_body[:3,3]
        variance[ids]+=pose.position_sigma_m**2+pose.rotation_sigma_rad**2*(lever[:,0]**2+lever[:,1]**2)
    valid &= np.isfinite(out).all(axis=1)&np.isfinite(variance)&(variance>=0)
    return WorldPoints(out[valid],variance[valid],times[valid],timeline.world_frame,source_sensor,calibration_id,
                       dict(input_points=len(points),accepted_points=int(valid.sum()),no_pose_points=no_pose,
                            accepted_clock=clock_id,motion_compensation='acquisition-time pose per timestamp'))


def unproject_depth(frame,calibration,timeline,*,now_s,max_age_s=.25):
    if frame.sensor_frame!=calibration.sensor_frame:
        raise ValueError('Optical frame/calibration mismatch')
    shape=(calibration.height,calibration.width)
    depth=np.asarray(frame.depth,dtype=float)*calibration.depth_scale_m
    valid=np.asarray(frame.valid);robot=np.asarray(frame.robot_mask)
    if depth.shape!=shape or valid.shape!=shape or robot.shape!=shape or valid.dtype!=bool or robot.dtype!=bool:
        raise ValueError('Depth and explicit boolean masks must match calibration resolution')
    if not frame.robot_mask_verified:
        raise ValueError('Robot-return exclusion is an explicit prerequisite')
    if not np.isfinite([frame.capture_time_s,frame.receive_time_s]).all():
        raise ValueError('Invalid frame acquisition/receipt time')
    accepted=valid&~robot&np.isfinite(depth)&(depth>=calibration.min_z_m)&(depth<=calibration.max_z_m)
    if frame.robot_depth_m is not None:
        robot_depth=np.asarray(frame.robot_depth_m,dtype=float)
        if robot_depth.shape!=shape:
            raise ValueError('Robot depth buffer shape mismatch')
        known_clear=np.isposinf(robot_depth)
        known_robot=np.isfinite(robot_depth)&(robot_depth>0)
        accepted &= known_clear|(known_robot&(depth<robot_depth-.005))
    sigma=np.broadcast_to(np.asarray(frame.depth_sigma_m,dtype=float),shape)
    accepted &= np.isfinite(sigma)&(sigma>=0)
    offsets=np.zeros(calibration.height) if frame.row_time_offsets_s is None else np.asarray(frame.row_time_offsets_s,dtype=float)
    if offsets.shape!=(calibration.height,) or not np.isfinite(offsets).all():
        raise ValueError('Expected finite row acquisition offsets')
    v,u=np.nonzero(accepted)
    z=depth[v,u];x=(u-calibration.ppx)/calibration.fx;y=(v-calibration.ppy)/calibration.fy
    points=np.column_stack((x*z,y*z,z));rays=np.column_stack((x,y,np.ones(len(z))))
    cov=sigma[v,u,None,None]**2*rays[:,:,None]*rays[:,None,:]
    cov[:,0,0]+=(calibration.pixel_sigma*z/calibration.fx)**2
    cov[:,1,1]+=(calibration.pixel_sigma*z/calibration.fy)**2
    result=transform_timed_points(points,cov,frame.capture_time_s+offsets[v],timeline,calibration.body_from_optical,
        now_s=now_s,receive_time_s=frame.receive_time_s,clock_id=frame.clock_id,source_sensor=frame.sensor_frame,
        calibration_id=calibration.calibration_id,max_observation_age_s=max_age_s)
    result.diagnostics.update(frame_pixels=int(depth.size),optically_accepted_pixels=len(z),robot_mask_pixels=int(robot.sum()))
    return result


def calibrated_point_cloud(points_sensor,*,capture_times_s,receive_time_s,now_s,clock_id,timeline,
                           body_from_sensor,source_sensor,calibration_id,valid,robot_mask,robot_mask_verified,
                           isotropic_sigma_m=.01,min_range_m=.1,max_range_m=20.,max_age_s=.25):
    points=np.asarray(points_sensor,dtype=float);valid=np.asarray(valid);robot_mask=np.asarray(robot_mask)
    if points.ndim!=2 or points.shape[1]!=3 or valid.shape!=(len(points),) or robot_mask.shape!=valid.shape or valid.dtype!=bool or robot_mask.dtype!=bool:
        raise ValueError('ExpectedNx3points and explicit boolean masks')
    if not robot_mask_verified or not calibration_id:
        raise ValueError('Requires calibrated extrinsics and verified robot-return mask')
    sigma=np.broadcast_to(np.asarray(isotropic_sigma_m,dtype=float),(len(points),))
    ranges=np.linalg.norm(points,axis=1)
    accept=valid&~robot_mask&np.isfinite(sigma)&(sigma>=0)&(ranges>=min_range_m)&(ranges<=max_range_m)
    cov=sigma[:,None,None]**2*np.eye(3)[None]
    return transform_timed_points(points[accept],cov[accept],np.asarray(capture_times_s)[accept],timeline,body_from_sensor,
        now_s=now_s,receive_time_s=receive_time_s,clock_id=clock_id,source_sensor=source_sensor,
        calibration_id=calibration_id,max_observation_age_s=max_age_s)


class LocalHeightMap:
    """A fixed local odom raster with latest-acquisition cell updates.

    It never interpolates holes or fills missing rays with flat terrain. New
    visible pit samples replace old heights, rather than averaging a pit closed.
    Spread within one cell contributes uncertainty: an edge/vertical wall may
    become unusable. 3D overhang/obstacle occupancy is a separate future product.
    """
    def __init__(self,*,origin_xy=(-.6,-.6),size_xy=(1.2,1.2),resolution_m=.02,world_frame='odom'):
        self.origin=np.asarray(origin_xy,dtype=float);self.resolution=resolution_m;self.world_frame=world_frame
        self.shape=tuple(np.ceil(np.asarray(size_xy)/resolution_m).astype(int))
        if self.origin.shape!=(2,) or not np.isfinite(self.origin).all() or resolution_m<=0 or min(self.shape)<=0:
            raise ValueError('Invalid map bounds')
        self.height=np.full(self.shape,np.nan);self.variance=np.full(self.shape,np.inf)
        self.observed=np.zeros(self.shape,dtype=bool);self.capture=np.full(self.shape,np.nan)
        self.source=np.full(self.shape,'',dtype='<U80')

    def integrate(self,cloud):
        if cloud.world_frame!=self.world_frame:
            raise ValueError('World frame mismatch; no implicit map/odom correction')
        p=np.asarray(cloud.points_m);var=np.asarray(cloud.variance_z_m2);times=np.asarray(cloud.capture_time_s)
        if p.shape!=(len(var),3) or times.shape!=var.shape:
            raise ValueError('Malformed point batch')
        finite=np.isfinite(p).all(axis=1)&np.isfinite(var)&(var>=0)&np.isfinite(times)
        p,var,times=p[finite],var[finite],times[finite]
        ij=np.floor((p[:,:2]-self.origin)/self.resolution).astype(int)
        inside=np.all((ij>=0)&(ij<self.shape),axis=1);p,var,times,ij=p[inside],var[inside],times[inside],ij[inside]
        updated=0
        if len(p):
            keys=np.ravel_multi_index(ij.T,self.shape)
            for key in np.unique(keys):
                ids=np.flatnonzero(keys==key);cell=np.unravel_index(key,self.shape);latest=float(times[ids].max())
                if self.observed[cell] and latest<self.capture[cell]-1e-10:
                    continue
                # Avoid mixing different acquisition slices into one fresh cell.
                ids=ids[times[ids]>=latest-.002]
                values=p[ids,2];median=float(np.median(values));span=float(np.ptp(values))
                new_variance=float(var[ids].max())+span**2
                if self.observed[cell] and abs(latest-self.capture[cell])<=.002:
                    # Simultaneous disagreeing views cannot make an edge or pit
                    # look confident merely because the last camera won.
                    disagreement=median-self.height[cell]
                    new_variance=max(self.variance[cell],new_variance)+disagreement**2
                self.height[cell]=median
                # Correlated stereo pixels do not yield 1/N covariance collapse.
                self.variance[cell]=new_variance
                self.capture[cell]=latest;self.observed[cell]=True;self.source[cell]=cloud.source_sensor;updated+=1
        return dict(accepted_points=len(p),updated_cells=updated)

    def channels(self,now_s):
        return terrain_channels(self.height,self.variance,self.observed,self.capture,now_s=now_s)

    def local_patch(self,world_from_body,*,now_s,extent_m=.8,resolution_m=.02):
        pose=validate_transform(world_from_body)
        forward=-pose[:3,1];forward[2]=0
        norm=np.linalg.norm(forward)
        if norm<1e-6:
            raise ValueError('Body heading is undefined')
        forward/=norm;left=np.cross([0,0,1.],forward)
        coordinates=np.arange(-extent_m/2+resolution_m/2,extent_m/2,resolution_m)
        f,l=np.meshgrid(coordinates,coordinates,indexing='ij')
        xy=pose[:2,3]+f[...,None]*forward[:2]+l[...,None]*left[:2]
        ij=np.floor((xy-self.origin)/self.resolution).astype(int)
        inside=np.all((ij>=0)&(ij<self.shape),axis=2)
        h=np.full(f.shape,np.nan);var=np.full(f.shape,np.inf);seen=np.zeros(f.shape,dtype=bool);stamp=np.full(f.shape,np.nan)
        ids=ij[inside];h[inside]=self.height[ids[:,0],ids[:,1]]-pose[2,3]
        var[inside]=self.variance[ids[:,0],ids[:,1]];seen[inside]=self.observed[ids[:,0],ids[:,1]];stamp[inside]=self.capture[ids[:,0],ids[:,1]]
        return dict(channels=terrain_channels(h,var,seen,stamp,now_s=now_s),height_relative_plate_m=h,
            axes='rows=forward,columns=left; gravity-aligned; height relative to body plate Z',world_xy=xy,
            resolution_m=resolution_m,world_frame=self.world_frame)


STEP=(.20,.35,-.08,.08,.02)
PIT=(-.35,-.20,-.08,.08,-.08)


def scene_height(x,y):
    z=np.zeros(np.broadcast(x,y).shape)
    for x0,x1,y0,y1,height in (STEP,PIT):
        z=np.where((x>=x0)&(x<=x1)&(y>=y0)&(y<=y1),height,z)
    return z


def synthetic_depth(calibration,world_from_body):
    """Analytic opaque floor/20mmstep/80mmpit with vertical walls; optical Z."""
    v,u=np.indices((calibration.height,calibration.width))
    rays=np.stack(((u-calibration.ppx)/calibration.fx,(v-calibration.ppy)/calibration.fy,np.ones_like(u)),axis=-1)
    world_from_camera=world_from_body@calibration.body_from_optical
    direction=np.einsum('...i,ji->...j',rays,world_from_camera[:3,:3]);o=world_from_camera[:3,3]
    best=np.full(u.shape,np.inf)
    for height in (0.,STEP[4],PIT[4]):
        with np.errstate(divide='ignore',invalid='ignore'):
            t=(height-o[2])/direction[...,2]
            p=o+t[...,None]*direction
        allowed=(t>0)&np.isfinite(t)&np.isclose(scene_height(p[...,0],p[...,1]),height)
        best=np.where(allowed&(t<best),t,best)
    for x0,x1,y0,y1,height in (STEP,PIT):
        for axis,value,other_min,other_max in [(0,x0,y0,y1),(0,x1,y0,y1),(1,y0,x0,x1),(1,y1,x0,x1)]:
            with np.errstate(divide='ignore',invalid='ignore'):
                t=(value-o[axis])/direction[...,axis];p=o+t[...,None]*direction
            allowed=(t>0)&np.isfinite(t)&(p[...,1-axis]>=other_min)&(p[...,1-axis]<=other_max)&(p[...,2]>=min(0,height))&(p[...,2]<=max(0,height))
            best=np.where(allowed&(t<best),t,best)
    return np.where(np.isfinite(best),best,np.nan)


def write_demo(out):
    from scipy.spatial import cKDTree
    study=ROOT/'artifacts/sensor_mount_study_2026-09-09'
    report=json.loads((study/'report.json').read_text());expanded=json.loads((study/'expanded_exact_mounts.json').read_text())
    rig=expanded['finalists'][0]
    mask_data=np.load(study/(rig['id']+'_expanded_exact.npz'))
    nominal=next(i for i,s in enumerate(expanded['finalist_cases']) if s['roll_deg']==0 and s['pitch_deg']==0 and s['pose']=='stance')
    coverage=mask_data['clear'][nominal];probes=np.array(report['targets']['points_world_m']);tree=cKDTree(probes[:,:2])
    model=LocalHeightMap();body=transform((0,0,.124))
    timeline=PoseTimeline([PoseSample(10.,body,.001,.001),PoseSample(10.10,body,.001,.001)])
    records=[]
    for i,camera in enumerate(rig['cameras']):
        ext=np.eye(4);ext[:3,:3]=camera['optical_rotation_body'];ext[:3,3]=camera['body_position_m']
        calibration=CameraCalibration(camera['id'],'synthetic_known_pinhole_not_device_calibration',160,90,
            report['profiles']['D405_848x480']['fx']*160/848,report['profiles']['D405_848x480']['fy']*90/480,80,45,.07,.5,ext,calibrated=True)
        depth=synthetic_depth(calibration,body)
        v,u=np.indices(depth.shape);optical=np.stack(((u-calibration.ppx)*depth/calibration.fx,(v-calibration.ppy)*depth/calibration.fy,depth),axis=-1)
        world=apply_rigid(optical.reshape(-1,3),body@ext).reshape(*depth.shape,3)
        distance,index=tree.query(np.nan_to_num(world[...,:2],nan=100.))
        # Geometry-informed pattern fixture only: nearby flat-CAD samples select
        # pixels; this is not a new exact self-occlusion rendering of the step/pit.
        visible=(distance<.025)&coverage[i,index]
        robot_mask=~visible
        frame=DepthFrame(camera['id'],'replay_clock',10.,10.05,depth,np.isfinite(depth),robot_mask,
                         depth_sigma_m=.002,robot_mask_verified=True)
        cloud=unproject_depth(frame,calibration,timeline,now_s=10.10)
        result=model.integrate(cloud);records.append(dict(sensor=camera['id'],projection=cloud.diagnostics,map_integration=result))
    fresh=model.channels(10.10);stale=model.channels(10.30)
    x=model.origin[0]+(np.arange(model.shape[0])+.5)*model.resolution
    y=model.origin[1]+(np.arange(model.shape[1])+.5)*model.resolution
    xx,yy=np.meshgrid(x,y,indexing='ij');truth=scene_height(xx,yy);usable=fresh[...,1].astype(bool)
    # Interiors avoid mixed wall cells; report missing interiors, never exclude
    # them silently from a completeness claim.
    bands={}
    for name,(x0,x1,y0,y1,height) in [('step',STEP),('pit',PIT)]:
        region=(xx>x0+.025)&(xx<x1-.025)&(yy>y0+.025)&(yy<y1-.025)
        known=region&usable
        bands[name]=dict(required_cells=int(region.sum()),usable_cells=int(known.sum()),
            mean_height_m=float(model.height[known].mean()) if known.any() else None,
            target_height_m=height,all_region_observed=support_region_observed(fresh,region))
    result=dict(version=1,status='synthetic_cpu_perception_contract_fixture',source_rig=rig['id'],
        source_coverage_sha256=hashlib.sha256((study/'expanded_exact_mounts.json').read_bytes()).hexdigest(),
        source_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        depth_input='160x90synthetic rectified optical-Z planes+walls; known synthetic intrinsics; not hardware calibration',
        visibility='nearest25mmneighbour of actual flat-CAD sampled mask, independently per camera; coherent geometric pattern, not exact step/pit camera rendering',
        map_contract='fixed continuous odomXY raster; observed/height/variance/acquisition-time; gravity-aligned body patch available; no unknown fill',
        sensors=records,total_cells=int(usable.size),fresh_usable_cells=int(usable.sum()),stale_usable_cells=int(stale[...,1].sum()),
        unobserved_usable_cells=int(np.sum(usable&~model.observed)),step_pit=bands,
        no_real_sensor_or_ros_integration=True,no_actor_injection=True,
        outstanding=['real selected-mode intrinsics/extrinsics and clock alignment','moving-C-CAD pixel masks at acquisition time','real noise and timing calibration','actual Mid360 packets and acquisition pattern','LIO/visual estimator','3D obstacle/overhang layer','student learning and simulator/runtime profiling'])
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'local_map_fixture.npz',height_m=model.height,variance_m2=model.variance,
        observed=model.observed,capture_time_s=model.capture,fresh_channels=fresh,stale_channels=stale,
        origin_xy=model.origin,resolution_m=model.resolution,truth_height_m=truth)
    patch=model.local_patch(body,now_s=10.10)
    np.savez_compressed(out/'student_patch_fixture.npz',**{k:v for k,v in patch.items() if isinstance(v,np.ndarray)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,5.2))
    extent=[y[0],y[-1],x[0],x[-1]]
    for ax,data,title in [(axes[0],truth,'Synthetic collision geometry'),(axes[1],np.where(usable,model.height,np.nan),'Visible height; unknown stays blank'),(axes[2],fresh[...,1],'Usable observations at 100 ms'),(axes[3],stale[...,1],'At 300 ms: stale observations rejected')]:
        if ax in axes[:2]:
            im=ax.imshow(data,origin='lower',extent=extent,vmin=-.08,vmax=.02,cmap='coolwarm')
        else:
            im=ax.imshow(data,origin='lower',extent=extent,vmin=0,vmax=1,cmap='Greens')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Odom Y, m');ax.set_ylabel('Odom X, m')
        ax.set_xlim(-.45,.45);ax.set_ylim(-.45,.45)
    fig.suptitle('Depth → acquisition-time transforms → uncertain local map',fontsize=15,weight='bold')
    for ax in axes[:2]:
        label_box=dict(boxstyle='round,pad=.18',fc='#24344a',ec='none',alpha=.75)
        ax.text(0,.265,'+20 mm',ha='center',va='center',color='white',fontsize=8,weight='bold',bbox=label_box)
        ax.text(0,-.265,'−80 mm',ha='center',va='center',color='white',fontsize=8,weight='bold',bbox=label_box)
    fig.text(.5,.03,'Synthetic calibrated fixture + geometry-informed visibility. No real sensor, ROS or learned perception claim.\nWhite height cells have no usable support observation; coloured ground is a measured fixture height.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.15,1,.92));fig.savefig(out/'perception_replay.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/perception_readiness_2026-09-09')
    write_demo(parser.parse_args().out)
