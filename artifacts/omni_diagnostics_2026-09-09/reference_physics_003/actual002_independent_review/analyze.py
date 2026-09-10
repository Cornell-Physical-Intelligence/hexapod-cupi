"""Independent read-only review of reference002's first measured swing."""
import hashlib
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp/reference_physics_results_002/run/wave'
trace = BASE / 'trace.npz'
d = np.load(trace)
refs = json.loads((BASE / 'reference_states.json').read_text())
valid = [r for r in refs if 'result' in r and r['result']['valid'][0]]
last = refs[-1]['result']
start = 199
end = len(d['time_s']) - 1
t = d['time_s'][:, 0]
R0 = d['rotation_world_from_body'][start, 0]
forward = R0 @ np.array([0., -1., 0.])
left = R0 @ np.array([1., 0., 0.])
displacement = d['position_world_m'][end, 0] - d['position_world_m'][start, 0]
vel_integral = np.trapezoid(d['velocity_world_mps'][start:, 0], t[start:], axis=0)
lf_contact = d['distal_contact'][:, 0, 0]
flight = np.flatnonzero(~lf_contact[start+1:]) + start + 1
toe = d['reference_point_world_m'][:, 0]
joint_names = list(d['joint_names'])
runtime_lf = [joint_names.index(name) for name in
              ('revolute_1_1', 'revolute_1', 'revolute_2')]
state = last['state']
endpoint = np.asarray(state['swing']['endpoint_world_m'])
rows = []
for k in (199, 200, 222, 223, int(np.argmax(toe[223:273, 0, 2]))+223, 272, 273):
    rows.append(dict(index=k, time_s=float(t[k]), lf_contact=bool(lf_contact[k]),
        lf_force_z_n=float(d['normal_force_world_n'][k, 0, 0, 2]),
        lf_toe_world_m=toe[k, 0].tolist(),
        lf_velocity_world_mps=d['reference_point_velocity_world_mps'][k, 0, 0].tolist(),
        lf_target_rad=d['joint_target_rad'][k, 0, runtime_lf].tolist(),
        lf_actual_rad=d['joint_position_rad'][k, 0, runtime_lf].tolist(),
        lf_requested_torque_nm=d['computed_torque_nm'][k, 0, runtime_lf].tolist()))
report = dict(
    trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    reference_states_sha256=hashlib.sha256((BASE/'reference_states.json').read_bytes()).hexdigest(),
    duration_after_settle_s=float(t[end]-t[start]),
    measured_forward_displacement_m=float(displacement@forward),
    measured_left_displacement_m=float(displacement@left),
    integrated_forward_displacement_m=float(vel_integral@forward),
    displacement_integral_difference_m=float(np.linalg.norm(displacement-vel_integral)),
    nominal_requested_distance_m=float(.005*(t[end]-t[start])),
    mean_filtered_admitted_forward_mps=float(np.mean([r['result']['admitted_command'][0] for r in valid])),
    last_filtered_admitted_forward_mps=float(last['admitted_command'][0]),
    last_body_tracking_error_m=float(last['diagnostics']['body_tracking_error_m']),
    lf_flight_first_index=int(flight[0]), lf_flight_last_index=int(flight[-1]),
    lf_flight_samples=int(len(flight)),
    lf_measured_lift_from_last_contact_m=float(toe[flight,0,2].max()-toe[flight[0]-1,0,2]),
    lf_return_contact_time_s=float(t[end]),
    lf_return_contact_swing_fraction=float((t[end]-state['swing']['start_s'])/state['swing']['duration_s']),
    lf_distance_to_planned_endpoint_at_contact_m=float(np.linalg.norm(toe[end,0]-endpoint)),
    lf_vertical_difference_to_virtual_endpoint_m=float(toe[end,0,2]-endpoint[2]),
    other_support_foot_max_displacement_m=float(np.linalg.norm(toe[start+1:,1:]-toe[start,1:][None],axis=-1).max()),
    minimum_support_count=int(d['distal_contact'][200:,0].sum(-1).min()),
    max_post_settle_requested_torque_nm=float(abs(d['computed_torque_nm'][200:]).max()),
    max_post_settle_target_velocity_rad_s=float(abs(d['target_velocity_rad_s'][200:]).max()),
    max_post_settle_target_acceleration_rad_s2=float(abs(d['target_acceleration_rad_s2'][200:]).max()),
    termination_count=int(d['terminated'].sum()),truncation_count=int(d['truncated'].sum()),
    return_contact_samples_available=1,
    rows=rows)
(Path(__file__).parent/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
