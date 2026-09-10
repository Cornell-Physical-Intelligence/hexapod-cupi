"""Read completed pair artifacts and render a compact, per-direction evidence report."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p/'tools/launch_omni_repair_spark.py').is_file())
sys.path.insert(0,str(REPO/'tools'))
from launch_omni_repair_spark import metrics,continuation_screen
ROOT=Path(__file__).parent

def trace_summary(path):
    x=np.load(path);rows=[]
    for i in range(x['age_s'].shape[1]):
        mask=(x['age_s'][:,i]>=2)&~x['terminated'][:,i]&~x['truncated'][:,i]
        ids=np.flatnonzero(mask)
        if len(ids)<2 or np.any(np.diff(ids)!=1):raise ValueError('Trace requires an uninterrupted settled window')
        q=x['joint_position_rad'][mask,i];dq=x['joint_velocity_rad_s'][mask,i]
        target=x['joint_target_rad'][mask,i];delta=np.diff(target,axis=0)
        pos=x['position_world_m'][mask,i];quat=x['quaternion_world_wxyz'][mask,i]
        # Robot's body-forward -Y; rotate it by quaternion then unwrap world yaw.
        w,a,b,c=quat.T
        forward_x=2*(w*c-a*b);forward_y=-(1-2*(a*a+c*c))
        heading=np.unwrap(np.arctan2(forward_y,forward_x))
        tau=x['computed_torque_nm'][mask,i]
        raw=x['raw_policy_action'][mask,i]
        fd=x['finite_difference_heading_rate_rad_s'][mask,i]
        spectrum=np.abs(np.fft.rfft(fd-fd.mean()))
        freq=np.fft.rfftfreq(len(fd),.02)
        rows.append({'samples':len(ids),'joint_velocity_rms_rad_s':float(np.sqrt(np.mean(dq*dq))),
            'joint_velocity_abs_p95_rad_s':float(np.quantile(np.abs(dq),.95)),
            'joint_fd_velocity_rms_rad_s':float(np.sqrt(np.mean((np.diff(q,axis=0)/.02)**2))),
            'max_joint_range_rad':float(np.ptp(q,axis=0).max()),
            'target_step_rms_rad':float(np.sqrt(np.mean(delta*delta))),
            'target_step_abs_p95_rad':float(np.quantile(np.abs(delta),.95)),
            'target_step_at_cap_fraction':float(np.mean(np.abs(delta)>=.029999)),
            'raw_action_rms':float(np.sqrt(np.mean(raw*raw))),
            'raw_action_temporal_std':float(np.mean(np.std(raw,axis=0))),
            'requested_torque_rate_rms_nm_s':float(np.sqrt(np.mean((np.diff(tau,axis=0)/.02)**2))),
            'planar_end_drift_m':float(np.linalg.norm(pos[-1,:2]-pos[0,:2])),
            'max_planar_excursion_m':float(np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1).max()),
            'heading_range_deg':float(np.rad2deg(np.ptp(heading))),
            'fd_heading_rate_std_rad_s':float(np.std(fd)),
            'fd_heading_rate_peak_hz':float(freq[1:][np.argmax(spectrum[1:])])})
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; published results are never overwritten')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): parser.error('Output must be a fresh directory')
    output.mkdir(parents=True)
    paths={'baseline':ROOT/'results/branch_a/baseline','A':ROOT/'results/branch_a/post','B':ROOT/'results/branch_b/post'}
    reports={k:json.loads((p/'diagnostics.json').read_text()) for k,p in paths.items()}
    m={k:metrics(v) for k,v in reports.items()}
    traces={k:trace_summary(p/'diagnostic_trace.npz') for k,p in paths.items()}
    screens={k:continuation_screen(m[k],m['baseline']) for k in ('A','B')}
    names=[r['name'] for r in reports['baseline']['scenarios']]
    data={'stage2_complete':False,'scope':'Two 50-update screens; no full qualification or visual pass',
          'metrics':m,'screens':screens,
          'standing_raw_reward_terms':{k:reports[k]['scenarios'][0]['windows']['post_settle_nonterminal']['mean_raw_reward_terms'] for k in paths},'trace_one_replica_per_scenario':{k:dict(zip(names,v)) for k,v in traces.items()},
          'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in paths.values() for p in d.iterdir() if p.is_file()}}
    (output/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# Two 50-update PPO comparisons','',
        'All results remain unqualified. Columns compare the same 12 scenarios, seed 7057, four replicas per scenario, and the settled nonterminal window from 2–12 seconds. Trace-only metrics below use one recorded replica per scenario. No further GPU work is automatically allocated.','',
        '| Metric | Original baseline | A: lower exploration | B: plus raw standing-action cost |','|---|---:|---:|---:|']
    for k in ('terminations','truncations','saturation','planar_error','yaw_error','power','stand_joint_velocity_rms'):
        lines.append('| '+k+' | '+' | '.join(f'{m[b][k]:.6g}' for b in paths)+' |')
    lines+=['','Per-direction values are planar error in m/s, yaw error in rad/s, and saturation percentage.','',
        '| Direction | Baseline planar / yaw / saturation | A | B |','|---|---:|---:|---:|']
    for name in names:
        vals=[]
        for b in paths:
            r=m[b]['scenarios'][name];vals.append(f"{r['planar_error_mps']:.4f} / {r['yaw_error_rad_s']:.4f} / {100*r['torque_saturation_fraction']:.2f}%")
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','Standing physical trace, one replica:','', '| Metric | Baseline | A | B |','|---|---:|---:|---:|']
    for k in traces['baseline'][0]:
        lines.append('| '+k+' | '+' | '.join(f'{traces[b][0][k]:.6g}' for b in paths)+' |')
    lines+=['','Standing raw-action mean square (four replicas): ' + ', '.join(f"{b}={reports[b]['scenarios'][0]['windows']['post_settle_nonterminal']['mean_raw_reward_terms']['stand_raw_action']:.6f}" for b in paths) + '.']
    for b in ('A','B'):
        lines+=['',f"Branch {b} allocation screen: `{json.dumps(screens[b])}`."]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,layout='constrained')
    xs=np.arange(len(names));colors=['#536878','#168a96','#dd8d34']
    for ax,key,label,scale in zip(axes,['planar_error_mps','yaw_error_rad_s','torque_saturation_fraction'],['Planar error (m/s)','Yaw error (rad/s)','Requested torque above cap (%)'],[1,1,100]):
        for offset,(b,color) in enumerate(zip(paths,colors)):
            ax.bar(xs+(offset-1)*.25,[m[b]['scenarios'][n][key]*scale for n in names],width=.24,color=color,label=b)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=3);axes[-1].set_xticks(xs,names,rotation=35,ha='right')
    fig.suptitle('PPO repair 003 — every direction remains subject to full qualification')
    fig.savefig(output/'direction_comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
