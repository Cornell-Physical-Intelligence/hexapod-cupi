"""Actual paired-motion raw data only; reference requests and measured data labeled."""
from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;run=H.parent/'run/paired_forward'
p=run/'trace.npz'
if not p.exists():p=run/'partial_trace.npz'
with np.load(p) as z:d={k:z[k] for k in z.files}
with np.load(run/'physics_substeps.npz') as z:s={k:z[k] for k in z.files}
r=json.loads((H/'report.json').read_text());t=d['time_s'][:,0];axis=-d['rotation_world_from_body'][299,0,:,1];axis[2]=0;axis/=np.linalg.norm(axis)
progress=np.sum((d['position_world_m'][:,0]-d['position_world_m'][299,0])*axis[None],axis=1)
assert np.isfinite(progress).all()
fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained');fig.set_facecolor('#f7f8fa')
for a in axes:
 a.set_facecolor('white');a.grid(alpha=.2);a.axvspan(0,4,color='#94a3b8',alpha=.18);a.axvspan(6,30,color='#60a5fa',alpha=.10)
quiet=(r.get('independent_gate') or {}).get('final_quiet_stop_window')
if quiet:
 final=(r.get('last_reference') or {}).get('state') or {};start=final.get('reference_quiet_time_s',0)+2
 for a in axes:a.axvspan(start,t[-1],color='#4ade80',alpha=.13)
axes[0].plot(t,progress*1000,label='Actual root-link forward displacement',color='#1d4ed8',lw=2)
axes[0].plot(t,np.clip(t-6,0,24)*10,label='Requested 10 mm/s × time',color='#64748b',ls='--');axes[0].set_ylabel('Progress (mm)');axes[0].legend(loc='upper left',ncol=2)
for i,name in enumerate(['LF','LM','LR','RF','RM','RR']):axes[1].plot(t,d['normal_force_world_n'][:,0,i,2],label=name,lw=1)
axes[1].axhline(1,color='black',ls=':',lw=1,label='1 N classification reference');axes[1].set_ylabel('Measured normal force Z (N)');axes[1].legend(ncol=7,fontsize=8,loc='upper left')
sel=s['time_s']>=4.0025;peak=abs(s['computed_torque_nm']).max(axis=(1,2));applied=abs(s['applied_torque_nm']).max(axis=(1,2))
axes[2].plot(s['time_s'][sel],peak[sel],label='All 400 Hz requested torque maxima',color='#7c3aed',lw=1)
axes[2].plot(s['time_s'][sel],applied[sel],label='Applied torque maxima',color='#f59e0b',lw=.7,alpha=.7);axes[2].axhline(1.6,color='#dc2626',ls='--',label='1.6 N·m bound');axes[2].set_ylabel('Torque (N·m)');axes[2].set_xlabel('Actual physical time (s)');axes[2].legend(ncol=3,fontsize=9)
axes[2].text(.01,.07,f"Initial-reset requested peak {r['full_initial_torque_peak_nm']:.2f} N·m retained in raw data; settling excluded from this torque view.",transform=axes[2].transAxes,fontsize=8,color='#475569')
axes[0].set_xlim(0,t[-1]);fig.suptitle(f"Actual paired reference 001 — {r['remote_status']}\nFresh standing → 0.01 m/s forward → stop; no PPO or terrain admission",fontsize=14,fontweight='bold')
fig.savefig(H/'actual_paired_motion.png',dpi=150)
