"""Separate proposed diagnostic scorer; never a walking/Stage2 gate override."""
import numpy as np
from quiet_contract import quiet_metrics
from load_transfer import PAIR, CORNERS


def longest_true_duration(mask, dt):
    longest=current=0
    for value in mask:
        current=current+1 if value else 0
        longest=max(longest,current)
    return longest*dt


def score_transfer(data, references, *, substeps=None, dt=.02):
    """Complete source004-shaped trace and P/V/A metadata; no force inference.

    The caller must separately verify immutable runtime/source/asset provenance.
    Synthetic fixtures can exercise these proposed criteria, not qualify physics.
    """
    if dt != .02:raise ValueError('Exact 50 Hz contract required')
    n=len(data['time_s'])
    if n<2 or len(references)!=n:raise ValueError('Complete aligned measured/reference trace required')
    times=np.asarray(data['time_s'])[:,0]
    if not np.allclose(np.diff(times),dt,atol=1e-7,rtol=0):raise ValueError('Contiguous physical time required')
    for key,value in data.items():
        if key not in ('contact_point_world_m','joint_names') and not np.isfinite(value).all():
            raise ValueError('Nonfinite measured data: '+key)
    contacts=data['distal_contact'][:,0]&data['contact_point_valid'][:,0]
    if not np.isfinite(data['contact_point_world_m'][:,0][contacts]).all():
        raise ValueError('Contact label has no finite measured point')
    if any(not bool(r['valid'][0]) for r in references):raise ValueError('Reference rejected; retain failure, no success score')
    for k, reference in enumerate(references):
        reference_time=float(reference['target_time_s'])
        validity=np.asarray(reference['valid'])
        if validity.dtype!=bool or validity.shape!=(1,):
            raise ValueError('Exact boolean single-replica reference validity required')
        if not np.isfinite(reference_time) or abs(reference_time-times[k]) > 1e-7:
            raise ValueError('Reference target time does not match paired pre-reset measured row')
        target=np.asarray(reference['q_ref'])
        if target.shape!=(1,18) or not np.isfinite(target).all() or np.max(np.abs(target-data['joint_target_rad'][k]))>2e-7:
            raise ValueError('Reference target does not match actual emitted target in paired row')
    quiet_time=references[-1]['state']['reference_quiet_time_s']
    failures=[]
    if quiet_time is None:
        failures.append('No finite reference return');quiet=None
    else:
        start=int(np.searchsorted(times,quiet_time+2.-1e-7))
        quiet=quiet_metrics(data,0,start,list(data['joint_names']),dt) if start<n-1 else None
        if quiet is None or quiet['window_duration_s']<10.-1e-7 or not quiet['pass']:
            failures.append('Missing or failed existing measured 10 s quiet criteria')
    normal=data['normal_force_world_n'][:,0]
    reaction=data['reaction_force_world_n'][:,0]
    z=data['reference_point_world_m'][:,0,:,2]
    airborne=np.zeros(n,bool)
    flight_counts=np.zeros(2,int);baselines=z[0,list(PAIR)].copy();last_contact=baselines.copy()
    lifts=np.zeros((n,2))
    for k in range(n):
        pair_clear=[]
        for j,leg in enumerate(PAIR):
            if contacts[k,leg]:
                flight_counts[j]=0;last_contact[j]=z[k,leg]
            else:
                if flight_counts[j]==0:baselines[j]=last_contact[j]
                flight_counts[j]+=1
            lifts[k,j]=z[k,leg]-baselines[j]
            pair_clear.append(flight_counts[j]>=2 and lifts[k,j]>=.002)
        airborne[k]=all(pair_clear) and (np.linalg.norm(normal[k,list(PAIR)],axis=-1)<=1.).all()
    hold=np.array([r['state']['mode']=='unloaded_hold' for r in references])
    stable=hold&airborne
    longest=longest_true_duration(stable,dt)
    if longest<1.-1e-7:failures.append('No contiguous 1 s measured two-foot unload with >=2 mm current clearance')
    if not contacts[:,list(CORNERS)].all():failures.append('Required corner support lost')
    if np.abs(data['computed_torque_nm']).max()>1.6:failures.append('Control-rate requested torque >1.6 N m')
    if np.abs(data['applied_torque_nm']).max()>1.60001:failures.append('Control-rate applied torque exceeds contract')
    nonfoot=data['shaft_contact'].any() or data['coxa_contact'].any() or data['femur_contact'].any() or data['base_contact'].any()
    if nonfoot or data['terminated'].any() or data['truncated'].any():failures.append('Non-foot contact or terminal/reset')
    if not contacts[-3:].all():failures.append('Final measured six-foot support not stable')
    expected_weight=8.26081134*9.81
    force_review=None
    if stable.any():
        forces=reaction[stable]
        mean_total=forces.sum(1).mean(0)
        relative_error=abs(mean_total[2]-expected_weight)/expected_weight
        force_review=dict(mean_reaction_each_foot_world_n=forces.mean(0).tolist(),
            mean_abs_requested_torque_each_joint_nm=np.abs(data['computed_torque_nm'][stable,0]).mean(0).tolist(),
            max_requested_torque_nm=float(np.abs(data['computed_torque_nm'][stable,0]).max()),
            remaining_requested_torque_margin_nm=float(1.6-np.abs(data['computed_torque_nm'][stable,0]).max()),
            nominal_static_four_corner_reference_nm={'vertical_only':1.12866,'ideal_mu_0p6':.621831},
            static_comparison_caveat='Nominal-pose ideal LP, not a force-control target or prediction at the measured deflected pose',
            mean_total_reaction_world_n=mean_total.tolist(),expected_static_weight_n=expected_weight,
            relative_vertical_balance_error=float(relative_error),
            maximum_tangential_to_normal_ratio=float((np.linalg.norm(forces[:,:,:2],axis=-1)/np.maximum(forces[:,:,2],1e-8)).max()))
        if relative_error>.05:failures.append('Proposed steady-load force balance error exceeds 5%')
    substep_review=None
    if substeps is None:
        failures.append('Missing complete 400 Hz requested/applied torque evidence')
    else:
        expected=n*8+1
        validate_substep_trace(substeps,data,dt=dt)
        maximum=float(np.abs(substeps['computed_torque_nm']).max())
        applied_maximum=float(np.abs(substeps['applied_torque_nm']).max())
        if maximum>1.6 or applied_maximum>1.60001:failures.append('Substep requested/applied torque exceeds contract')
        substep_review=dict(samples=len(substeps['computed_torque_nm']),expected_samples=expected,
            max_requested_torque_nm=maximum,max_applied_torque_nm=applied_maximum)
    return dict(scope='Proposed static load-transfer diagnostic criteria, not adopted walking gates',
        proposal_only=True,proposed_criteria_met=not failures,failed_proposed_criteria=failures,
        longest_measured_unloaded_hold_s=longest,measured_pair_peak_lift_m=lifts.max(0).tolist(),
        force_review=force_review,quiet_review=quiet,substep_review=substep_review,
        external_source_asset_provenance_verified=False,physics_qualification=False,stage2_complete=False)


def check_substep_batch(rows, control_index, *, initial_counter, control_row):
    """Proposed host hook: call after each step, before emitting another target."""
    if len(rows)!=8:raise ValueError('Eight captured physics substeps required')
    for index,row in enumerate(rows,1):
        if int(row['control_index'])!=control_index or int(row['substep_index'])!=index:
            raise ValueError('Missing, repeated or reordered physics substep')
        for key,bound in (('computed_torque_nm',1.6),('applied_torque_nm',1.60001)):
            value=np.asarray(row[key])
            if value.shape!=(1,18) or not np.isfinite(value).all() or np.abs(value).max()>bound:
                raise ValueError('Stop before next target: invalid or overlimit physics-substep '+key)
    counters=np.array([int(row['sim_step_counter']) for row in rows])
    if not np.array_equal(counters,initial_counter+8*control_index+np.arange(1,9)):
        raise ValueError('Batch physics counters do not match the declared control')
    for key in ('time_s','sdk_sim_timestamp_s'):
        times=np.array([float(row[key]) for row in rows])
        if not np.isfinite(times).all() or not np.allclose(np.diff(times),.0025,atol=1e-7,rtol=0):
            raise ValueError('Batch has missing/reordered 400 Hz timestamps')
    for key in ('computed_torque_nm','applied_torque_nm'):
        if not np.array_equal(rows[-1][key],control_row[key]):
            raise ValueError('Batch endpoint differs from measured control row')
    return {'complete':True,'control_index':control_index,'substeps':8}


def validate_substep_trace(substeps, control, *, dt=.02):
    """Mirror source004 capture cadence, counters and exact endpoint equality."""
    n=len(control['time_s']);count=n*8+1
    for key in ('computed_torque_nm','applied_torque_nm'):
        value=np.asarray(substeps[key])
        if value.shape!=(count,1,18) or not np.isfinite(value).all():
            raise ValueError('Complete finite single-replica substep torque shape required: '+key)
        if not np.array_equal(value[8::8],control[key]):
            raise ValueError('Substep torque differs from paired control endpoint: '+key)
    expected_control=np.r_[-1,np.repeat(np.arange(n),8)]
    expected_substep=np.r_[0,np.tile(np.arange(1,9),n)]
    for key,expected in [('control_index',expected_control),('substep_index',expected_substep)]:
        if not np.array_equal(substeps[key],expected):
            raise ValueError('Missing, reordered or repeated substep index: '+key)
    counter=np.asarray(substeps['sim_step_counter'])
    if counter.shape!=(count,) or not np.array_equal(counter-counter[0],np.arange(count)):
        raise ValueError('Physics counter did not advance exactly once per substep')
    relative=np.asarray(substeps['time_s']);sdk=np.asarray(substeps['sdk_sim_timestamp_s'])
    for key,value in [('time_s',relative),('sdk_sim_timestamp_s',sdk)]:
        if value.shape!=(count,) or not np.isfinite(value).all() or not np.allclose(np.diff(value),dt/8,atol=1e-7,rtol=0):
            raise ValueError('Exact finite 400 Hz cadence required: '+key)
    expected_times=np.asarray(control['time_s'])[:,0]
    origin=expected_times[0]-dt-relative[0]
    if not np.allclose(relative[8::8]+origin,expected_times,atol=1e-7,rtol=0):
        raise ValueError('Substep and pre-reset control timestamps disagree')
