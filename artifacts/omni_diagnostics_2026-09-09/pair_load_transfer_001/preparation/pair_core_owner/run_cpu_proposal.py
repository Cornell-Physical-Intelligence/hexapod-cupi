"""Produce clearly labeled synthetic evidence for the proposed diagnostic."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synthetic_fixture import Fixture
from load_transfer import PairLoadTransfer,PAIR,CORNERS
from score_transfer import score_transfer
from test_score_transfer import synthetic_substeps

p=Path(__file__).resolve().parent
f=Fixture();c=PairLoadTransfer(f.names);c.reset(f.snapshot());rows=[];refs=[]
for k in range(1100):
    out=c.step(f.snapshot())
    if not out['valid'][0]:raise RuntimeError(out['failure_reason'])
    f.advance(out);rows.append(f.snapshot());refs.append(out)
d={key:np.stack([r[key] for r in rows]) for key in rows[0]};d['joint_names']=np.array(f.names)
# Deliberately prescribed torque/contact/body fixture, NOT generated physics.
sub=synthetic_substeps(d)
scored=score_transfer(d,refs,substeps=sub)
q=np.concatenate([r['q_ref'] for r in refs]);v=np.concatenate([r['v_ref'] for r in refs]);a=np.concatenate([r['a_ref'] for r in refs])
limits=d['soft_joint_pos_limits_rad'][0,0]
feet=d['reference_point_world_m'][:,0];baseline=c.measured_toes0
result=dict(scope='Synthetic kinematic/contact/unit-force fixture only; no Isaac or real sensor evidence',
    physical_run=False,proposed_only=True,config=refs[0]['state']['config'],runtime_joint_names=f.names,
    source004_interface=True,total_reference_duration_s=float(d['time_s'][-1,0]),
    selected_pair=['lm','rm'],retained_supports=['lf','lr','rf','rr'],
    max_reference_velocity_rad_s=float(abs(v).max()),max_reference_acceleration_rad_s2=float(abs(a).max()),
    minimum_soft_joint_margin_rad=float(np.minimum(q-limits[:,0],limits[:,1]-q).min()),
    maximum_pair_planar_excursion_m=float(np.linalg.norm(feet[:,list(PAIR),:2]-baseline[list(PAIR),:2],axis=-1).max()),
    max_pair_measured_fixture_lift_m=(feet[:,list(PAIR),2]-baseline[list(PAIR),2]).max(0).tolist(),
    final_target_equals_initial=bool(np.array_equal(refs[-1]['q_ref'][0],c.q0)),
    reference_quiet_time_s=c.reference_quiet_time,
    maximum_corner_target_change_rad=float(abs(q[:,c.base.to_leg.reshape(6,3)[list(CORNERS)].reshape(-1)]-c.q0[c.base.to_leg.reshape(6,3)[list(CORNERS)].reshape(-1)]).max()),
    synthetic_criteria_review=scored,
    caveats=['Synthetic body pose follows a fixed fixture; actual body is never prescribed by the proposed generator.',
             'Synthetic torque is a fixed 0.7 N m and synthetic force distribution is equal over declared contacts.',
             'Those fixture values test rejection/scoring plumbing and predict no actual torque or load allocation.',
             'No GPU entrypoint, new source manifest, host dispatch or physical gate adoption is provided.'])
(p/'report.json').write_text(json.dumps(result,indent=2)+'\n')
time=d['time_s'][:,0]
fig,axs=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
for j,label in zip(PAIR,['LM','RM']):axs[0].plot(time,(feet[:,j,2]-baseline[j,2])*1000,label=label)
axs[0].set(ylabel='Synthetic toe rise (mm)',title='Proposed middle-pair load transfer — CPU trajectory only');axs[0].legend()
axs[1].plot(time,abs(v).max(1),label='Maximum reference speed');axs[1].set(ylabel='Joint rate (rad/s)',xlabel='Seconds from settled reset')
for ax in axs:
 for t in (2,5,7,10):ax.axvline(t,color='gray',linewidth=.7,alpha=.4)
 ax.grid(alpha=.2)
fig.savefig(p/'reference_schedule.png',dpi=150)
print(json.dumps({k:v for k,v in result.items() if k!='synthetic_criteria_review'},indent=2))
