"""Reproducible CPU-only feasibility/continuity report. Writes only beside itself."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reference import *

OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
g=SerialGeometry();r=TwistReference(g)
start=time.monotonic()
commands=[('stand',[0,0,0])]
for yaw in (-.4,-.2,.2,.4):commands.append((f'turn_{yaw:+.2f}',[0,0,yaw]))
for speed in (.05,.10,.20):
    for bearing in range(16):
        a=2*math.pi*bearing/16
        for yaw in (-.4,-.2,0,.2,.4):
            commands.append((f'v{speed:.2f}_bearing{bearing*22.5:g}_yaw{yaw:+.2f}',[speed*math.cos(a),speed*math.sin(a),yaw]))
phase=torch.arange(256,dtype=DTYPE)/256
rows=[]
for label,command in commands:
    c=tensor([command]).expand(len(phase),3)
    feet=r.feet(phase,c);q=g.ik(feet)
    f=r.frequency(c)
    # Independent central time difference; no learned/actuator velocity claims.
    h=1e-5
    vel=(r.feet(phase+h*f,c)-r.feet(phase-h*f,c))/(2*h)
    _,J,_=g.fk(q['q_checked']);qvel=torch.linalg.solve(J,vel[...,None]).squeeze(-1)
    finite_velocity=torch.isfinite(qvel).all(-1)
    valid=q['valid'] & finite_velocity & (qvel.abs()<=50.26548246).all(-1)
    rows.append(dict(name=label,command=command,frequency_hz=float(f[0]),
                     all_sampled_kinematics_valid=bool(valid.all()),
                     invalid_leg_phase_samples=int((~valid).sum()),
                     max_requested_ik_error_m=float(q['error_m'].max()),
                     joint_limit_fail_samples=int((~q['in_limits']).sum()),
                     minimum_soft_limit_margin_rad=float(q['minimum_joint_margin_rad'].min()),
                     minimum_jacobian_singular_value_m=float(q['min_jacobian_singular_value_m'].min()),
                     max_joint_speed_rad_s=float(qvel.abs().max()),
                     max_horizontal_excursion_from_nominal_m=float((feet[...,:2]-g.feet0[None,:,:2]).norm(dim=-1).max())))

# One continuous episode, no phase/reset between directions and stop.
segments=[('start',2,[0,0,0]),('forward',5,[.1,0,0]),('reverse',5,[-.1,0,0]),
          ('left',5,[0,.1,0]),('left_arc',5,[0,.1,.2]),('right_arc',5,[.1,0,-.2]),
          ('yaw',5,[0,0,.3]),('stop',12,[0,0,0])]
filt=FilterState(r);trace=[];q_last=None;lastfoot=None;max_delta=0;invalid=0
for label,duration,target in segments:
    for j in range(int(duration/.02)):
        state=filt.step([target],.02)
        q=state['q_checked'][0];foot=state['feet'][0]
        delta=torch.zeros_like(q) if q_last is None else q-q_last
        max_delta=max(max_delta,float(delta.abs().max()));invalid+=int((~state['valid']).sum())
        trace.append(dict(t=.02*len(trace),label=label,command=filt.command[0].tolist(),
                          rate=filt.rate[0].tolist(),phase=float(filt.phase[0]),
                          feet=foot.tolist(),q=q.tolist(),q_velocity=state['q_velocity'][0].tolist(),
                          valid=bool(state['valid'].all()),delta_max=float(delta.abs().max())))
        q_last=q;lastfoot=foot
# Motion of nominal stance points in world if the *filtered requested* twist were
# perfectly achieved. This is a kinematic slip diagnostic, not actual slip.
transition_slip=[]
for sample in trace:
    phase0=tensor([sample['phase']]);c=tensor([sample['command']]);rate=tensor([sample['rate']])
    state=r.state(phase0,c,rate)
    p=torch.remainder(phase0[:,None]+r.offsets,1)
    mask=p<r.config.duty
    xy=state['feet'][...,:2]
    body=nav_to_body(c)[:,None,:]+c[:,None,2,None]*torch.stack((-xy[...,1],xy[...,0]),-1)
    residual=(state['foot_velocity'][...,:2]+body).norm(dim=-1)[mask]
    transition_slip.append(float(residual.max()))
arr={key:np.asarray([t[key] for t in trace]) for key in ('t','command','rate','phase','feet','q','q_velocity','valid','delta_max')}
arr['stance_reference_world_slip_mps']=np.asarray(transition_slip)
np.savez_compressed(OUT/'transition_trace.npz',**arr)

original_q=tensor(g.reference['positions_rad']).reshape(-1,6,3)
original_feet,_,_=g.fk(original_q)
p=(torch.arange(len(original_q),dtype=DTYPE)+.5)/len(original_q)
match=r.feet(p,tensor([[.2,0,0]]).expand(len(p),3),frequency_override=1.3)
benchmark_error=float((match-original_feet).norm(dim=-1).max())
report=dict(status='CPU_only_contingency_not_selected_or_dynamically_admitted',
    source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   [Path(__file__),OUT/'reference.py',OUT/'test_reference.py',BENCH/'f050_t060.urdf',BENCH/'candidate_c_reference.json',ROOT/'tools/omni_flat_math.py']},
    benchmark=dict(urdf_sha256=g.urdf_sha,reference_cycle_shape_max_fk_error_m=benchmark_error,
                   match_condition='forward .20m/s, fixed 1.3Hz, duty .65, lift .020m; not a learned-policy comparison',
                   selected_checkpoint_sha256='5ecbe978d659413dc050fec93b0e211a3eadf6ce1c807790d87c14044ff245a2'),
    configuration=r.config.__dict__,
    command_screen=dict(cases=len(rows),phases_per_case=256,leg_phase_samples=len(rows)*256*6,
                        passed_cases=sum(x['all_sampled_kinematics_valid'] for x in rows),
                        failed_cases=sum(not x['all_sampled_kinematics_valid'] for x in rows),
                        notes='Point-foot workspace + soft limits + local conditioning + URDF speed only; no torque/contact/collision/support admission',rows=rows),
    transition=dict(seconds=len(trace)*.02,segments=segments,invalid_leg_steps=invalid,
                    max_joint_reference_step_rad=max_delta,
                    final_max_q_velocity_rad_s=float(np.abs(arr['q_velocity'][-1]).max()),
                    final_max_foot_offset_m=float(np.linalg.norm(arr['feet'][-1]-g.feet0.numpy(),axis=-1).max()),
                    max_stance_world_reference_slip_mps=max(transition_slip),
                    note='Slip diagnostic assumes commanded twist exactly achieved; actual robot tracking is unknown. Target slew/actuator dynamics are not simulated.'),
    cpu_seconds=time.monotonic()-start,
    limitations=['No simulator, trained actor, motor torque/thermal load, support forces or collisions evaluated.',
                 'Point feet from frozen serial C benchmark; not full pad contact or physical four-bar motion.',
                 'C2 phase path and filtered commands do not guarantee no-slip during command transitions.',
                 'No sensor, terrain, command governor, residual-policy integration or architecture selection.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,(title,cmd) in zip(axes[0], [('Forward',[.1,0,0]),('Left strafe',[0,.1,0]),('Forward + left turn',[.1,0,.3])]):
    feet=r.feet(phase,tensor([cmd]).expand(len(phase),3)).numpy()
    for i,name in enumerate(LEGS):
        ax.plot(-feet[:,i,1],feet[:,i,0],label=name,lw=1.8)
        ax.scatter(-g.feet0[i,1],g.feet0[i,0],s=16)
    ax.set(xlabel='Forward (-body Y), m',ylabel='Left (+body X), m',title=title,aspect='equal');ax.grid(alpha=.2)
axes[0,0].legend(ncol=3,fontsize=8)
for i,name in enumerate(('Forward m/s','Left m/s','Yaw rad/s')):axes[1,0].plot(arr['t'],arr['command'][:,i],label=name)
axes[1,0].legend(fontsize=8);axes[1,0].set(title='Continuous reversals, arcs and settling',xlabel='Time, s',ylabel='Filtered body command');axes[1,0].grid(alpha=.2)
axes[1,1].plot(arr['t'],np.max(np.abs(arr['q_velocity']),axis=(1,2)),color='#166885')
axes[1,1].set(title='Reference motion settles; feedback stays available',xlabel='Time, s',ylabel='Maximum |joint reference velocity|, rad/s');axes[1,1].grid(alpha=.2)
axes[1,2].plot(arr['t'],arr['stance_reference_world_slip_mps'],color='#b26717')
axes[1,2].set(title='Transition limitation: predicted stance slip',xlabel='Time, s',ylabel='Reference world slip, m/s');axes[1,2].grid(alpha=.2)
fig.suptitle('Kinematic contingency only — no dynamic, torque, contact or training admission',fontsize=15)
fig.savefig(OUT/'prototype.png',dpi=150);plt.close(fig)
print(json.dumps({k:report[k] for k in ('status','benchmark','transition','cpu_seconds')},indent=2))
print('CASE_SUMMARY',report['command_screen']['passed_cases'],report['command_screen']['failed_cases'])
print('MAX_JOINT_SPEED',max(row['max_joint_speed_rad_s'] for row in rows))
print('FAILURES',[row['name'] for row in rows if not row['all_sampled_kinematics_valid']][:20])
