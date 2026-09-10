"""CPU flat-scene acquisition replay with actual004 motion and C mesh occlusion."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import screen_sensor_mounts as screen
from sensor_mount_math import optical_visibility,axis_rotation
from perception_replay import LocalHeightMap,PoseSample,PoseTimeline,calibrated_point_cloud

RESOLUTION=.02
ORIGIN=np.array([-.6,-.6])
SHAPE=(60,60)
LEASE=.25
LATENCY=.04
HORIZONS=(0.,.1,.25,.5,3.,6.)


class StudyRobot(screen.RobotBounds):
    """Generic root/named FK override; mesh screening implementation unchanged."""
    def forward(self,q):
        parents={j.find('parent').get('link') for j in self.joints}
        children={j.find('child').get('link') for j in self.joints}
        roots=parents-children
        if len(roots)!=1:raise ValueError('One explicit URDF root required')
        out={roots.pop():np.eye(4)};remaining=list(self.joints)
        while remaining:
            advanced=False
            for j in remaining[:]:
                parent=j.find('parent').get('link')
                if parent not in out:continue
                out[j.find('child').get('link')]=out[parent]@screen.origin(j)@axis_rotation(screen.vector(j.find('axis').get('xyz')),q[j.get('name')])
                remaining.remove(j);advanced=True
            if not advanced:raise ValueError('Disconnected recorded joint layout')
        return out


def ground_grid():
    x=ORIGIN[0]+(np.arange(SHAPE[0])+.5)*RESOLUTION
    y=ORIGIN[1]+(np.arange(SHAPE[1])+.5)*RESOLUTION
    xx,yy=np.meshgrid(x,y,indexing='ij')
    return np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))


def footprint_mask(points,centres,radius=.015):
    """Conservative grid-cell intersection with a proposed 15 mm foot disk."""
    centres=np.asarray(centres)
    if centres.ndim!=2 or centres.shape[1]!=3 or not np.isfinite(centres).all():
        raise ValueError('Finite world foothold centres required')
    lower=ORIGIN;upper=ORIGIN+np.array(SHAPE)*RESOLUTION
    inside=((centres[:,:2]-radius>=lower)&(centres[:,:2]+radius<=upper)).all()
    distance=np.linalg.norm(points[:,None,:2]-centres[None,:,:2],axis=-1)
    masks=distance<=radius+RESOLUTION*np.sqrt(2)/2
    return masks.any(1).reshape(SHAPE),masks, bool(inside)


def pose_matrix(data,index):
    T=np.eye(4);T[:3,:3]=data['rotation_world_from_body'][index,0];T[:3,3]=data['position_world_m'][index,0]
    q=data['quaternion_world_xyzw'][index,0]
    if not np.isfinite(T).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4 or not np.allclose(Rotation.from_quat(q).as_matrix(),T[:3,:3],atol=2e-6,rtol=0):
        raise ValueError('Raw XYZW quaternion and explicit body rotation disagree')
    return T


def visibility_sequence(data,mount):
    screen.ROOT=ROOT/'assets'
    robot=StudyRobot(screen.ROOT/'robot/hexapod_mkii_length_study/urdf/f050_t060.urdf')
    exact=screen.ExactMeshScreen(robot);points=ground_grid();cameras=mount['selected_rig']['cameras'];profile=mount['profile']
    #10 Hz subset of actual50Hz recorded frames;2more recorded frames provide
    #current body/foot positions at the explicitly assumed40ms receipt latency.
    indices=np.arange(199,len(data['time_s'])-2,5)
    clear=[];optics=[];ambiguous=[];start=time.monotonic()
    for n,index in enumerate(indices):
        body=pose_matrix(data,index);ground=np.einsum('ni,ij->nj',points-body[:3,3],body[:3,:3])
        fk=robot.forward(dict(zip(data['joint_names'].tolist(),data['joint_position_rad'][index,0])))
        cm=[];om=[];am=[]
        for camera in cameras:
            p=np.asarray(camera['body_position_m']);R=np.asarray(camera['optical_rotation_body'])
            optical=np.einsum('ni,ij->nj',ground-p,R);inside=optical_visibility(optical,profile);ids=np.flatnonzero(inside)
            left,_,left_amb=exact.clearance(p,ground[ids],fk)
            right,_,right_amb=exact.clearance(p+profile['baseline_m']*R[:,0],ground[ids],fk)
            visible=np.zeros(len(points),bool);visible[ids]=left&right
            ambiguity=np.zeros(len(points),bool);ambiguity[ids]=left_amb|right_amb
            cm.append(visible);om.append(inside);am.append(ambiguity)
        clear.append(cm);optics.append(om);ambiguous.append(am)
        if n%20==0:print('Ray frame',n+1,'of',len(indices),'elapsed',round(time.monotonic()-start,1),flush=True)
    return dict(indices=indices,points=points,clear=np.asarray(clear),optics=np.asarray(optics),origin_ambiguous=np.asarray(ambiguous))


def run_maps(data,mount,visibility):
    points=visibility['points'];cameras=mount['selected_rig']['cameras'];sectors=[c['leg_sector'] for c in cameras]
    samples=[PoseSample(float(data['time_s'][i,0]),pose_matrix(data,i),position_sigma_m=.001,rotation_sigma_rad=.001) for i in range(len(data['time_s']))]
    timeline=PoseTimeline(samples,clock_id='actual004_replay_clock',world_frame='source004_world')
    rows=[];last_maps={};map_arrays={}
    for rig in mount['selected_rig']['rig_comparisons']:
        members=[sectors.index(name) for name in rig['sectors']]
        for dropout in (False,True):
            grid=LocalHeightMap(origin_xy=ORIGIN,size_xy=(1.2,1.2),resolution_m=RESOLUTION,world_frame=timeline.world_frame)
            key=str(rig['camera_count'])+('cam_dropout' if dropout else 'cam_continuous')
            case_rows=[];fresh=[];usable_masks=[];ever_masks=[];required_masks=[]
            for frame,index in enumerate(visibility['indices']):
                capture=float(data['time_s'][index,0]);now=capture+LATENCY
                lost=dropout and 8.0-1e-8<=capture<8.6-1e-8
                body=pose_matrix(data,index);ground=np.einsum('ni,ij->nj',points-body[:3,3],body[:3,:3])
                instant=visibility['clear'][frame,members].any(0)
                optical=visibility['optics'][frame,members].any(0)
                if not lost:
                    for camera_index in members:
                        camera=cameras[camera_index];p=np.asarray(camera['body_position_m']);R=np.asarray(camera['optical_rotation_body'])
                        optical_points=np.einsum('ni,ij->nj',ground-p,R);clear=visibility['clear'][frame,camera_index]
                        T=np.eye(4);T[:3,:3]=R;T[:3,3]=p
                        cloud=calibrated_point_cloud(optical_points,capture_times_s=np.full(len(points),capture),receive_time_s=now,
                            now_s=now,clock_id=timeline.clock_id,timeline=timeline,body_from_sensor=T,
                            source_sensor=camera['id'],calibration_id='synthetic_nominal_profile_not_device_calibration',
                            valid=clear,robot_mask=~clear,robot_mask_verified=True,isotropic_sigma_m=.003,
                            min_range_m=.001,max_range_m=1.,max_age_s=LEASE)
                        grid.integrate(cloud)
                # Footprint coordinates come from actual/recorded planner state
                #at receipt time, never from future terrain traversal.
                centres=data['planned_footprint_centres_world_m'][index+2]
                required,per_foot,inside=footprint_mask(points,centres)
                channels=grid.channels(now);usable=channels[...,1].astype(bool)
                current_required,_,_=footprint_mask(points,data['reference_point_world_m'][index+2,0])
                active=int(data['active_swing_leg_index'][index+2])
                active_mask=per_foot[:,active] if active>=0 else None
                fractions=[]
                for horizon in HORIZONS:
                    later=grid.channels(now+horizon)[...,1].astype(bool)
                    fractions.append(float(later[required].mean()) if inside and required.any() else 0.)
                row=dict(capture_time_s=capture,receive_time_s=now,dropped=lost,
                    instantaneous_optical_fraction=float(optical.mean()),instantaneous_mesh_clear_fraction=float(instant.mean()),
                    currently_acquired_fraction=0. if lost else float(instant.mean()),ever_observed_fraction=float(grid.observed.mean()),
                    retained_usable_fraction=float(usable.mean()),required_cells=int(required.sum()),required_inside_map=inside,
                    required_footprint_instantaneous_fraction=float(instant.reshape(SHAPE)[required].mean()),
                    required_footprint_optical_fraction=float(optical.reshape(SHAPE)[required].mean()),
                    current_planted_and_swinging_footprint_usable_fraction=float(usable[current_required].mean()),
                    active_swing_leg_index=active,
                    active_planned_footprint_usable_fraction=None if active_mask is None else float(usable.ravel()[active_mask].mean()),
                    required_footprint_usable_fraction=float(usable[required].mean()),
                    required_per_sector_usable_fraction=[float(usable.ravel()[per_foot[:,i]].mean()) for i in range(6)],
                    required_usable_at_future_use_fraction=fractions,
                    complete_required_support_observation=bool(inside and usable[required].all()))
                case_rows.append(row);fresh.append(instant);usable_masks.append(usable.ravel());ever_masks.append(grid.observed.ravel().copy());required_masks.append(required.ravel())
            values=lambda key:np.array([r[key] for r in case_rows])
            rows.append(dict(id=key,camera_count=rig['camera_count'],sectors=rig['sectors'],dropout=dropout,
                mean_instantaneous_visibility_fraction=float(values('instantaneous_mesh_clear_fraction').mean()),
                mean_retained_usable_fraction=float(values('retained_usable_fraction').mean()),
                final_ever_observed_fraction=float(values('ever_observed_fraction')[-1]),
                mean_required_footprint_usable_fraction=float(values('required_footprint_usable_fraction').mean()),
                minimum_required_footprint_usable_fraction=float(values('required_footprint_usable_fraction').min()),
                complete_required_footprint_frames=int(values('complete_required_support_observation').sum()),
                mean_required_footprint_optical_fraction=float(values('required_footprint_optical_fraction').mean()),
                mean_active_planned_footprint_usable_fraction=float(np.mean([r['active_planned_footprint_usable_fraction'] for r in case_rows if r['active_planned_footprint_usable_fraction'] is not None])),
                mean_required_future_use_fraction=np.mean([r['required_usable_at_future_use_fraction'] for r in case_rows],axis=0).tolist(),frames=case_rows))
            map_arrays[key+'_instantaneous']=np.array(fresh);map_arrays[key+'_usable']=np.array(usable_masks);map_arrays[key+'_ever']=np.array(ever_masks);map_arrays[key+'_required']=np.array(required_masks)
            last_maps[key]=dict(height=grid.height,capture=grid.capture,observed=grid.observed,variance=grid.variance)
    return rows,map_arrays


def main():
    data=dict(np.load(ROOT/'inputs/motion_source004.npz'));mount=json.loads((ROOT/'inputs/mounts.json').read_text())
    cache=ROOT/'visibility.npz'
    receipt=ROOT/'visibility_receipt.json'
    identity={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [ROOT/'inputs/motion_source004.npz',ROOT/'inputs/mounts.json',Path(__file__),*(ROOT/'runtime').glob('*.py'),*(ROOT/'assets').rglob('*')] if f.is_file()}
    if cache.exists() and receipt.exists() and json.loads(receipt.read_text())==identity:
        visibility=dict(np.load(cache))
    else:
        visibility=visibility_sequence(data,mount);np.savez_compressed(cache,**visibility);receipt.write_text(json.dumps(identity,indent=2)+'\n')
    rows,maps=run_maps(data,mount,visibility);np.savez_compressed(ROOT/'map_masks.npz',**maps,points=visibility['points'],indices=visibility['indices'])
    report=dict(scope='Synthetic sparse flat-ground acquisition with actual004 C mesh/body/joint motion; no real sensor or terrain traversal',
        actual_motion_trace_sha256=json.loads((ROOT/'SOURCE_SHA256.json').read_text())['actual_motion_trace_sha256'],
        frame_count=len(visibility['indices']),capture_hz=10.,latency_s=LATENCY,max_usable_age_s=LEASE,grid_resolution_m=RESOLUTION,
        optical_profile=mount['profile'],profile_primary_source='Frozen D400 datasheet rev020 tables3-49,4-11 and section4.4; https://www.realsenseai.com/download/21345/?tmstv=1780360410',
        synthetic_noise_and_pose_assumptions={'point_sigma_m':.003,'body_position_sigma_m':.001,'body_rotation_sigma_rad':.001,'hardware_validated':False},
        calibration='Nominal inferred profile and hypothetical mounts, not device calibration',scene='flat world Z=0 sampled at20mm cell centres; not a full depth-image renderer',
        motion='Raw actual004 recorded poses/joints from failed wave; no physics pose prescribed or terrain transplant',
        occlusion='Exact included C simulation visual triangles, both stereo segments; camera-origin ambiguity is unknown',
        mounts='Hashed production-anchor proposals reused as hypothetical C placements; bracket/cable/housing feasibility unvalidated',
        observed_mask='Ever acquired sampled cells; does not imply current freshness',usable_mask='Explicit250ms age and15mm uncertainty criteria from existing terrain contract',
        required_footprints='All six15mmradius footprint disks conservatively rasterized; current feet plus active original planned endpoint at receipt time',
        future_use_horizons_s=HORIZONS,future_use_assumption='No future reacquisition. Same required world cells; this is an age-lease stress test, not a computed stopping envelope.',
        pose_provider='Recorded simulator ground truth with explicitly assumed1mm/.001rad uncertainty; no real pose estimator qualified',
        support_eligibility_computed=False,physics_qualification=False,hardware_qualification=False,cases=rows)
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in rows],indent=2),flush=True)


if __name__=='__main__':main()
