"""Independent, read-only same-endpoint decomposition of actual paired001 data.

Every reported integration alternative is diagnostic. The original 50 Hz gate
and its recorded rejected verdict remain unchanged. No simulator is imported.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy.integrate import simpson


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def interval(t, p, v, begin, end):
    """Use exact recorded boundary times; reject missing or shifted boundaries."""
    t = np.asarray(t, dtype=np.float64)
    a, b = [int(np.argmin(np.abs(t - x))) for x in (begin, end)]
    if not (a < b and abs(t[a]-begin) < 1e-8 and abs(t[b]-end) < 1e-8):
        raise ValueError('Exact, ordered interval endpoints required')
    dt = np.diff(t[a:b+1])
    if not np.all(dt > 0):
        raise ValueError('Monotonic physics timestamps required')
    p = np.asarray(p, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    disp = p[b]-p[a]
    left = (v[a:b]*dt[:, None]).sum(0)
    right = (v[a+1:b+1]*dt[:, None]).sum(0)
    integrals = {'trapezoid':(left+right)/2, 'left':left, 'right':right,
                 'simpson':simpson(v[a:b+1], x=t[a:b+1], axis=0)}
    return {
        'start_s':float(t[a]), 'end_s':float(t[b]), 'intervals':b-a,
        'start_index':a, 'end_index':b, 'duration_s':float(t[b]-t[a]),
        'position_displacement_m':disp.tolist(),
        'integrations':{k:{'integral_m':x.tolist(), 'residual_m':(disp-x).tolist(),
                           'residual_norm_m':float(np.linalg.norm(disp-x))}
                        for k,x in integrals.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.raw_root/'remote_audit.json').read_text())
    for name, digest in audit['raw_payloads'].items():
        if sha(args.raw_root/name) != digest:
            raise ValueError('Raw mismatch: '+name)
    run = args.raw_root/'run/paired_forward'
    d = dict(np.load(run/'trace.npz', allow_pickle=False))
    s = dict(np.load(run/'physics_substeps.npz', allow_pickle=False))
    state = json.loads((run/'state.json').read_text())
    refs = json.loads((run/'reference_states.json').read_text())
    tools = args.source/'tools'; sys.path.insert(0, str(tools))
    from screen_metrics import measured_progress
    from physics_substeps import displacement_check
    old = measured_progress(d, start_step=299, end_step=1499)
    old400 = displacement_check(s, 300, 1500)
    assert not state['gate']['passed']
    assert abs(old['displacement_integral_difference_m'] -
               state['gate']['independent_forward_motion']['displacement_integral_difference_m']) < 1e-8
    pairs = [('position_world_m','root_link_position_world_m'),
             ('velocity_world_mps','root_link_velocity_world_mps'),
             ('quaternion_world_xyzw','root_link_quaternion_world_xyzw'),
             ('joint_position_rad','joint_position_rad'),
             ('joint_velocity_rad_s','joint_velocity_rad_s'),
             ('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]
    for field, high in pairs:
        assert np.array_equal(d[field], s[high][8::8]), field
    assert np.array_equal(s['relative_physics_index'], np.arange(19201))
    assert np.array_equal(s['sim_step_counter']-s['sim_step_counter'][0], np.arange(19201))
    assert np.allclose(np.diff(s['sdk_sim_timestamp_s']), .0025, atol=1e-7, rtol=0)
    assert len(d['time_s']) == 2400
    t = s['time_s'].astype(float)
    p = s['root_link_position_world_m'][:,0].astype(float)
    v = s['root_link_velocity_world_mps'][:,0].astype(float)
    pc = s['root_com_position_world_m'][:,0].astype(float)
    vc = s['root_com_velocity_world_mps'][:,0].astype(float)
    w = s['root_angular_velocity_world_rad_s'][:,0].astype(float)
    t50 = d['time_s'][:,0].astype(float)
    p50 = d['position_world_m'][:,0].astype(float)
    v50 = d['velocity_world_mps'][:,0].astype(float)
    last = refs[-1]['result']['state']
    quiet = last.get('reference_quiet_time_s', last.get('controller',{}).get('reference_quiet_time_s'))
    if quiet is None:
        # Pair schema keeps shared controller state under scalar_state.
        quiet = last['scalar_state']['reference_quiet_time_s']
    # The source ceil on the exact serialized time excludes 100 further controls.
    quiet_start_control = int(np.ceil(quiet/.02))+100
    quiet_start = float(t50[quiet_start_control])
    segments = [('pre_motion_hold',4,6),('motion',6,30),
                ('motion_6_12',6,12),('motion_12_18',12,18),
                ('motion_18_24',18,24),('motion_24_30',24,30),
                ('stop_until_reference_quiet',30,round(quiet,2)),
                ('excluded_quiet_settle',round(quiet,2),quiet_start),
                ('scored_quiet_endpoints',quiet_start,48)]
    blocks = {name:{'link50_float64':interval(t50,p50,v50,a,b),
                  'link400_float64':interval(t,p,v,a,b),
                  'com400_float64':interval(t,pc,vc,a,b)} for name,a,b in segments}
    a,b = 2400,12000; h=.0025
    velmid = (v[a:b]+v[a+1:b+1])/2
    actual_interval_v = np.diff(p[a:b+1], axis=0)/h
    rate_residual = actual_interval_v-velmid
    delta = p[b]-p[a]
    trap400 = np.asarray(blocks['motion']['link400_float64']['integrations']['trapezoid']['integral_m'])
    trap50 = np.asarray(blocks['motion']['link50_float64']['integrations']['trapezoid']['integral_m'])
    # Group additive control-interval residuals by declared reference mode.
    by_mode = {}
    for row in refs:
        control = row['physical_step']
        if not 300 <= control < 1500: continue
        result = row['result']; assert abs(result['target_time_s']-t50[control]) < 1e-8
        key = result['state']['mode']
        item = by_mode.setdefault(key, {'controls':0,'residual_m':np.zeros(3)})
        lo,hi=control*8,(control+1)*8
        integ=((v[lo:hi]+v[lo+1:hi+1])/2).sum(0)*h
        item['controls']+=1; item['residual_m']+=p[hi]-p[lo]-integ
    assert sum(x['controls'] for x in by_mode.values()) == 1200
    for x in by_mode.values(): x['residual_m']=x['residual_m'].tolist()
    rotated = np.einsum('tij,tj->ti',d['rotation_world_from_body'][:,0],d['velocity_body_mps'][:,0])
    lever_residual = vc-v-np.cross(w,pc-p)
    lags = []
    for shift in range(-8,9):
        vv=v[a+shift:b+shift+1]; ii=(vv[:-1]+vv[1:]).sum(0)*h/2
        lags.append({'shift_substeps':shift,'shift_s':shift*h,'residual_m':(delta-ii).tolist(),
                     'residual_norm_m':float(np.linalg.norm(delta-ii))})
    # Per-control cumulative arrays make the 50 Hz sampling contribution explicit.
    motion_t=t[a:b+1:8]
    hc=np.r_[np.zeros((1,3)),np.cumsum(((v[a:b]+v[a+1:b+1])/2)*h,axis=0)][::8]
    lv=v50[299:1500]
    lc=np.r_[np.zeros((1,3)),np.cumsum((lv[:-1]+lv[1:])*.01,axis=0)]
    pd=p[a:b+1:8]-p[a]
    np.savez_compressed(args.output/'decomposition_arrays.npz',time_s=motion_t,
        position_displacement_m=pd,velocity_integral_50_m=lc,velocity_integral_400_m=hc,
        residual_50_m=pd-lc,residual_400_m=pd-hc,
        substep_phase_mean_residual_mps=rate_residual.reshape(1200,8,3).mean(0))
    report = {
        'scope':'Independent actual paired001 root-rate diagnosis; original rejection retained. No physics correction or velocity-fidelity admission.',
        'input_raw_files_verified':len(audit['raw_payloads']), 'raw_payloads':audit['raw_payloads'],
        'remote_audit_sha256':sha(args.raw_root/'remote_audit.json'),
        'source_manifest_sha256':audit['source_manifest_sha256'],
        'source_functions_sha256':{str(f.relative_to(args.source)):sha(f) for f in (tools/'screen_metrics.py',tools/'physics_substeps.py',tools/'physics_telemetry.py')},
        'recorded_original_gate':state['gate']['independent_forward_motion'],
        'local_original_float32_replay':old,
        'original_source_float32_400Hz_diagnostic':old400,
        'same_endpoint_and_clock_checks':{'control_boundaries':[300,1500], 'control_trace_indices':[299,1499],
           'physics_indices':[2400,12000], 'time_s':[6,30], 'all_2400_control_endpoints_equal':True,
           'all_19201_physics_indices_and_counters_equal':True, 'all_sdk_substep_timestamps_advance':True,
           'quiet_requested_s':last.get('stop_requested_time_s'), 'quiet_reference_s':quiet,
           'scored_quiet_first_sample_s':quiet_start, 'quiet_source_samples':state['gate']['final_quiet_stop_window']['window_samples'],
           'quiet_source_duration_s':state['gate']['final_quiet_stop_window']['window_duration_s'],
           'note':'504 samples are scored as 10.08 s by source; first-to-last position endpoints span 10.06 s. Both are retained.'},
        'blocks':blocks,
        'decomposition_identity':{'equation':'e50 = e400 + (integral400 - integral50)',
            'e50_m':(delta-trap50).tolist(),'e400_m':(delta-trap400).tolist(),
            'sampling_contribution_m':(trap400-trap50).tolist(),
            'additive_identity_max_error_m':float(np.max(abs((delta-trap400)+(trap400-trap50)-(delta-trap50))))},
        'frame_checks':{'rotation_body_to_world_reconstructs_com_velocity_max_abs_mps':float(np.max(abs(rotated-d['com_velocity_world_mps'][:,0]))),
            'com_minus_link_velocity_equals_omega_cross_offset_max_abs_mps':float(np.max(abs(lever_residual))),
            'link_com_400_residual_difference_m':(np.asarray(blocks['motion']['link400_float64']['integrations']['trapezoid']['residual_m'])-np.asarray(blocks['motion']['com400_float64']['integrations']['trapezoid']['residual_m'])).tolist(),
            'reported_navigation_mapping':'[-body_COM_velocity_y, body_COM_velocity_x, body_angular_velocity_z]',
            'point_warning':'Navigation uses body-expressed root-COM velocity; original gap uses matched root-link position and root-link world velocity.'},
        'within_substep_motion':{'mean_position_interval_rate_mps':actual_interval_v.mean(0).tolist(),
            'mean_sdk_midpoint_rate_mps':velmid.mean(0).tolist(),
            'mean_rate_residual_mps':rate_residual.mean(0).tolist(),
            'rms_rate_residual_mps':np.sqrt((rate_residual**2).mean(0)).tolist(),
            'substep_phases_1_to_8_mean_residual_mps':rate_residual.reshape(1200,8,3).mean(0).tolist(),
            'reference_mode_additive_residuals':by_mode},
        'lag_sensitivity_not_a_clock_correction':lags,
        'limits':['No direct native solver internal velocities or integration corrections were exported.',
                  'No joint-rate-bias causal attribution is made.',
                  'Finite-difference velocity is an interval measurement, not a proven instantaneous replacement.',
                  'No alternative quadrature, frame or window changes the original 5 mm verdict.'],
    }
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_raw_verified':len(audit['raw_payloads']),
        'recorded_gap_mm':1000*state['gate']['independent_forward_motion']['displacement_integral_difference_m'],
        '400_float64_gap_mm':1000*blocks['motion']['link400_float64']['integrations']['trapezoid']['residual_norm_m'],
        'quiet_first_sample_s':quiet_start,'reference_mode_residuals':by_mode},indent=2))


if __name__ == '__main__': main()
