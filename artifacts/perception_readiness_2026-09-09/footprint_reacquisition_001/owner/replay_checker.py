"""Causal new-footprint lease replay of frozen004 flat acquisition evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from footprint_checker import *
from perception_replay import PoseSample,PoseTimeline,calibrated_point_cloud
ROOT=Path(__file__).resolve().parent
FRAME='source004_world';CLOCK='actual004_replay_clock';CALIBRATION='synthetic_nominal_profile_not_device_calibration'
ORIGIN=(-.6,-.6);SIZE=(1.2,1.2);RES=.02;SHAPE=(60,60)
LEGS=('lf','lm','lr','rf','rm','rr')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    source=json.loads((ROOT/'INPUTS_SHA256.json').read_text())
    for section in ['files','runtime_files']:
        if not all(sha(ROOT/k)==v for k,v in source[section].items()):raise ValueError('Replay input/runtime identity mismatch')
    return source

def pose_matrix(data,index):
    T=np.eye(4);T[:3,:3]=data['rotation_world_from_body'][index,0];T[:3,3]=data['position_world_m'][index,0]
    if not np.allclose(Rotation.from_quat(data['quaternion_world_xyzw'][index,0]).as_matrix(),T[:3,:3],atol=2e-6,rtol=0):raise ValueError('Explicit XYZW pose mismatch')
    return T

def prepare_frames(data,mount,visibility):
    timeline=PoseTimeline([PoseSample(float(data['time_s'][i,0]),pose_matrix(data,i),position_sigma_m=.001,rotation_sigma_rad=.001) for i in range(len(data['time_s']))],clock_id=CLOCK,world_frame=FRAME)
    frames={};points=visibility['points'];cameras=mount['selected_rig']['cameras']
    for sequence,index in enumerate(visibility['indices']):
        capture=float(data['time_s'][index,0]);received=float(data['time_s'][index+2,0])
        if abs(received-capture-.04)>1e-9:raise ValueError('Actual receipt is not capture plus 40 ms')
        T=pose_matrix(data,index);body_points=np.einsum('ni,ij->nj',points-T[:3,3],T[:3,:3])
        group=[]
        for k,camera in enumerate(cameras):
            p=np.asarray(camera['body_position_m']);R=np.asarray(camera['optical_rotation_body']);optical=np.einsum('ni,ij->nj',body_points-p,R)
            optics=visibility['optics'][sequence,k];clear=visibility['clear'][sequence,k];amb=visibility['origin_ambiguous'][sequence,k]
            valid=optics&clear&~amb;body_from_sensor=np.eye(4);body_from_sensor[:3,:3]=R;body_from_sensor[:3,3]=p
            # Optical min/max Z and FOV are already enforced by the hashed ray
            # cache. The generic point-cloud Euclidean guard is additional only.
            cloud=calibrated_point_cloud(optical,capture_times_s=np.full(len(points),capture),receive_time_s=received,now_s=received,clock_id=CLOCK,timeline=timeline,body_from_sensor=body_from_sensor,source_sensor=camera['id'],calibration_id=CALIBRATION,valid=valid,robot_mask=~valid,robot_mask_verified=True,isotropic_sigma_m=.003,min_range_m=.001,max_range_m=1.,max_age_s=LEASE_S)
            group.append(FrameReceipt(camera['id'],sequence,capture,received,CLOCK,cloud,optics.reshape(SHAPE),clear.reshape(SHAPE),amb.reshape(SHAPE),'frozen_exact_C_visual_stereo_cache_frame_'+str(sequence)))
        frames[int(index+2)]=group
    return frames

def run_case(case,data,mount,frames,timing,contacts):
    cameras=mount['selected_rig']['cameras'];start=float(data['time_s'][199,0])
    ledger=FootprintEvidenceLedger(origin_xy=ORIGIN,size_xy=SIZE,resolution_m=RES,world_frame=FRAME,clock_id=CLOCK,sensors=[SensorSchedule(c['id'],CALIBRATION,start,.1,.04) for c in cameras])
    uncertainty=DriftUncertainty(**case['uncertainty']);one=np.ones(SHAPE,bool);zero=np.zeros(SHAPE,bool);rows=[];events={};map_rows=[]
    for index in range(199,len(data['time_s'])):
        now=float(data['time_s'][index,0]);delivered=False;capture_visibility=None
        if index in frames:
            group=frames[index];capture=group[0].capture_time_s
            lost=any(low-1e-8<=capture<high-1e-8 for low,high in case['dropout_capture_intervals_s'])
            capture_visibility=float(np.stack([f.optical_mask&f.mesh_clear_mask&~f.origin_ambiguous_mask for f in group]).any(0).mean())
            if not lost:
                for frame in group:ledger.ingest(frame,now_s=now)
                delivered=True
        age=now-ledger.map.capture
        variance=ledger.map.variance+uncertainty.additional_z_sigma_m**2+(uncertainty.z_sigma_growth_m_per_s*np.maximum(age,0))**2
        channels=terrain_channels(ledger.map.height,variance,ledger.map.observed,ledger.map.capture,now_s=now,max_age_s=LEASE_S,max_std_m=MAX_STD_M)
        map_rows.append(dict(time_s=now,ever_observed_fraction=float(ledger.map.observed.mean()),retained_usable_fraction=float(channels[...,1].mean()),acquisition_received_this_control=delivered,sampled_visibility_at_capture_fraction=capture_visibility))
        reference=timing[index];swing=reference['swing'];active=int(data['active_swing_leg_index'][index])
        if active<0:continue
        if swing is None or reference['current_leg']!=LEGS[active]:raise ValueError('Matched reference and active leg disagree')
        key=reference['current_leg']+'@'+format(swing['start_s'],'.6f')
        # This is the original planned endpoint, including while the reference
        # performs a contact landing blend. No future actual touchdown is read.
        endpoint=data['planned_footprint_centres_world_m'][index,active]
        nominal_end=swing['start_s']+swing['duration_s'];use=max(now,nominal_end)
        proposal=FootstepProposal(key,LEGS[active],tuple(endpoint),.015,swing['start_s'],use,FRAME,CLOCK)
        geometry=SupportGeometry(one,one,zero,one,FRAME,CLOCK,now,'Known synthetic flat Z=0 fixture inside bounded raster; not sensor-estimated support','teacher_fixture_only')
        planted=[]
        for leg in range(6):
            point=contacts['contact_point_world_m'][index,0,leg];valid=bool(contacts['contact_point_valid'][index,0,leg]);loaded=bool(contacts['distal_contact'][index,0,leg])
            planted.append(PlantedContact(LEGS[leg],now,tuple(point),loaded,valid,FRAME,CLOCK))
        result=ledger.query(proposal,geometry,uncertainty,now_s=now,planted_contacts=planted)
        masks=result.pop('masks');result.update(index=index,reference_mode=reference['mode'],original_planned_end_s=nominal_end,original_swing_start_s=swing['start_s'],remaining_planned_horizon_s=max(0.,nominal_end-now),original_endpoint_world_m=endpoint.tolist(),ever_observed_fraction=float(ledger.map.observed.mean()),retained_usable_fraction=float(masks['usable_now'].mean()),acquisition_received_this_control=delivered,sampled_visibility_at_capture_fraction=capture_visibility,sampled_visibility_is_current_pose=False)
        rows.append(result);events.setdefault(key,[]).append(result)
    stats=[]
    for key,series in events.items():
        complete=[r for r in series if r['complete_usable_eligible_now']];covered=[r for r in series if r['lease_covers_requested_use_time']]
        stats.append(dict(proposal_id=key,foot_id=series[0]['foot_id'],first_query_time_s=series[0]['query_time_s'],last_query_time_s=series[-1]['query_time_s'],nominal_end_s=series[0]['original_planned_end_s'],queries=len(series),first_decision=series[0]['decision'],fresh_now_queries=len(complete),current_coverage_fraction=len(complete)/len(series),lease_covers_use_queries=len(covered),first_lease_covers_use_time_s=None if not covered else covered[0]['query_time_s'],ever_qualified_physical_abort=False))
    summary=dict(id=case['id'],queries=len(rows),fresh_now_queries=sum(r['complete_usable_eligible_now'] for r in rows),lease_covers_use_queries=sum(r['lease_covers_requested_use_time'] for r in rows),blocked_queries=sum(r['decision']=='blocked' for r in rows),missing_latest_frame_queries=sum(any(x['missing_latest_scheduled_frame'] for x in r['sensor_health'].values()) for r in rows),mean_retained_usable_map_fraction=float(np.mean([r['retained_usable_fraction'] for r in rows])),mean_ever_observed_map_fraction=float(np.mean([r['ever_observed_fraction'] for r in rows])),events=stats)
    return {'assumptions':case,'summary':summary,'rows':rows,'all_control_map_rows':map_rows}

def main():
    origin=verify();data=dict(np.load(ROOT/'inputs/motion_source004.npz'));mount=json.loads((ROOT/'inputs/mounts.json').read_text());visibility=dict(np.load(ROOT/'inputs/visibility.npz'));contacts=dict(np.load(ROOT/'inputs/measured_contacts_source004.npz'));timing={r['index']:r for r in json.loads((ROOT/'inputs/reference_timing_source004.json').read_text())}
    if not np.array_equal(data['time_s'],contacts['time_s']) or not np.array_equal(data['distal_contact'],contacts['distal_contact']):raise ValueError('Actual contact time/layout mismatch')
    frames=prepare_frames(data,mount,visibility)
    zero_u=dict(additional_z_sigma_m=0.,z_sigma_growth_m_per_s=0.,xy_sigma_m=.001,xy_sigma_growth_m_per_s=0.)
    cases=[dict(id='continuous_six_camera',dropout_capture_intervals_s=[],uncertainty=zero_u),dict(id='one_missing_capture_5p5',dropout_capture_intervals_s=[[5.5,5.6]],uncertainty=zero_u),dict(id='outage_8p0_to_8p6',dropout_capture_intervals_s=[[8.,8.6]],uncertainty=zero_u),dict(id='registration_drift',dropout_capture_intervals_s=[],uncertainty=dict(additional_z_sigma_m=.004,z_sigma_growth_m_per_s=.08,xy_sigma_m=.001,xy_sigma_growth_m_per_s=.004))]
    results=[run_case(c,data,mount,frames,timing,contacts) for c in cases]
    report=dict(scope='Prototype decision-only new-footprint lease replay; synthetic flat acquisition with recorded actual004 motion and hypothetical six-camera rig',input_identity=origin,grid_resolution_m=RES,proxy_foot_radius_m=.015,max_capture_age_s=LEASE_S,max_std_m=MAX_STD_M,capture_hz=10.,assumed_receipt_latency_s=.04,query_hz=50.,optical_profile=mount['profile'],optical_profile_provenance='Frozen D400 datasheet rev020 tables 3-49/4-11 and section 4.4; nominal inferred calibration, not device calibration',optical_envelope='Hashed cache enforces D405 selected FOV, optical min Z .07 m and max Z .5 m; both stereo lines of sight required, ambiguous origins unknown',self_occlusion='Actual004 moving C visual triangles in frozen exact-ray cache; unmodeled camera housings/cables remain a limitation',pose_uncertainty='Recorded simulator poses with synthetic 1 mm position / .001 rad rotation; 3 mm point noise; additional drift assumptions listed per case',current_support='Actual time-matched distal-contact point/validity remains separate, never fills terrain map or new foot disk',eligibility='Explicit ideal flat teacher geometry AND sensor usability; no real traversability estimator or teacher-to-sensor equivalence',future_observations_assumed=False,original009_stop_reference_latency_s=5.54,original009_excluded_settling_s=2.,stop_comparison='Recorded009 stop latency is context only; no004 motion extrapolation and no safe stop/abort path generated',stage2_complete=False,actor_integrated=False,hardware_qualified=False,terrain_traversal_qualified=False,cases=results)
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');summary={k:v for k,v in report.items() if k!='cases'};summary['cases']=[c['summary'] for c in results];(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['cases'],indent=2));verify()
if __name__=='__main__':main()
