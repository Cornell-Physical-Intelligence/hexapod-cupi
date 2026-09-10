"""Independent matched-origin arithmetic; retain original quiet/physics verdicts."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent/'run'
CASES=('origin_a','near','far','near_opposite','origin_repeat')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=[]
for case in CASES:
    state=json.loads((RUN/case/'state.json').read_text())
    initial=json.loads((RUN/case/'initial_readback.json').read_text())
    ground=json.loads((RUN/case/'ground_readback.json').read_text())
    assert state['status']=='completed' and state['control_steps']==1000
    assert initial['passed'] and ground['passed']
    with np.load(RUN/case/'physics_substeps.npz') as raw:
        assert len(raw['time_s'])==8001
        np.testing.assert_array_equal(raw['relative_physics_index'],np.arange(8001))
        t=raw['time_s'][1600:].astype(float);dt=np.diff(t)
        np.testing.assert_allclose(dt,.0025,rtol=0,atol=1e-12)
        q=raw['joint_position_rad'][1600:].astype(float)
        v=raw['joint_velocity_rad_s'][1600:].astype(float)
        assert np.isfinite(q).all() and np.isfinite(v).all()
        integral=np.sum(.5*(v[1:]+v[:-1])*dt[:,None,None],axis=0)[0]
        delta=(q[-1]-q[0])[0];gap=integral-delta;j=int(np.argmax(np.abs(gap)))
        p=raw['root_link_position_world_m'][1600:].astype(float)
        linear=raw['root_link_velocity_world_mps'][1600:].astype(float)
        root_gap=np.linalg.norm((p[-1]-p[0])-np.sum(.5*(linear[1:]+linear[:-1])*dt[:,None,None],axis=0),axis=-1)[0]
        quiet=state['unchanged_quiet_gate'];physical=state['unchanged_physical_gate']
        rows.append(dict(case=case,translation_xy_m=state['identity']['translation_xy_m'],
            actual_origin_sync_verified=initial['origin_storage_sync']['passed'],
            existing_physical_passed=physical['passed'],existing_quiet_passed=quiet['pass'],
            reported_quiet_joint_rms_max_rad_s=quiet['max_joint_velocity_rms_rad_s'],
            worst_joint=raw['joint_names'][j].item(),actual_angle_delta_rad=float(delta[j]),
            actual_angle_range_rad=float(np.ptp(q[:,0,j])),reported_rate_integral_rad=float(integral[j]),
            reported_integral_minus_angle_rad=float(gap[j]),
            angle_increment_interval_rms_rad_s=float(np.sqrt(np.mean((np.diff(q[:,0,j])/dt)**2))),
            root_link_displacement_integral_gap_m=float(root_gap),raw_substeps_sha256=sha(RUN/case/'physics_substeps.npz')))

repeat={name:sha(RUN/'origin_a'/name)==sha(RUN/'origin_repeat'/name)
        for name in ('trace.npz','physics_substeps.npz','physics_control_integrals.npz')}
report=dict(rows=rows,origin_repeat_raw_byte_equality=repeat,
    all_existing_physical_passed=all(r['existing_physical_passed'] for r in rows),
    all_existing_quiet_passed=all(r['existing_quiet_passed'] for r in rows),
    worst_joint_same_in_all_cases=len({r['worst_joint'] for r in rows})==1,
    max_abs_joint_integral_gap_rad=max(abs(r['reported_integral_minus_angle_rad']) for r in rows),
    min_abs_joint_integral_gap_rad=min(abs(r['reported_integral_minus_angle_rad']) for r in rows),
    scope='Five common-state cold runs. Translation does not eliminate the large reported-rate/angle disagreement in any tested case. Native cause remains unproven; no metric substitution, PPO or velocity-fidelity admission.',
    stage2_complete=False,velocity_fidelity_qualified=False)
path=ROOT/'report.json'
if path.exists():assert json.loads(path.read_text())==report
else:path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'physical_passes':sum(r['existing_physical_passed'] for r in rows),
    'quiet_passes':sum(r['existing_quiet_passed'] for r in rows),'repeat_exact':repeat,
    'joint_integral_gap_range_rad':[report['min_abs_joint_integral_gap_rad'],report['max_abs_joint_integral_gap_rad']]}))
