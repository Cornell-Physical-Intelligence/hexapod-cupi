"""Actual diagonal static diagnostic forces/torques; no generated or imposed motion."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parent;r=json.loads((H/'report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,7.2),gridspec_kw={'height_ratios':[1,1.15]},layout='constrained')
colors=['#23748c','#926346'];legs=['LF','LM','LR','RF','RM','RR']
for col,case in enumerate(('lf_rr','lr_rf')):
 v=r['cases'][case];g=v['independent_score'];force=np.asarray(g['force_review']['mean_reaction_each_foot_world_n'])[:,2];ax=axs[0,col]
 bars=ax.bar(legs,force,color=colors[col],width=.65);ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_ylim(0,32);ax.set_ylabel('Mean vertical reaction (N)');ax.set_title(case.upper().replace('_',' + ')+' unloaded · measured hold 2.02 s',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 margin=v['independent_post_measurement_replay']['extrema']['measured_support_margin_m']*1000
 ax.text(.97,.95,f'Min support margin {margin:.1f} mm\nRequired ≥50 mm',transform=ax.transAxes,ha='right',va='top',fontsize=9)
 with np.load(R/'run'/case/'physics_substeps.npz',allow_pickle=False) as z:t=z['time_s'];tau=abs(z['computed_torque_nm']).max(axis=(1,2))
 ax=axs[1,col];ax.axvspan(5,7,color='#bfc6cc',alpha=.35,label='Unload hold');ax.axvspan(12,22,color='#cfe8d5',alpha=.65,label='Scored quiet');ax.plot(t,tau,color=colors[col],lw=1.1);ax.axhline(1.6,color='#b84639',ls='--',lw=1.2,label='1.6 Nm limit');ax.set(xlim=(0,22),ylim=(0,1.72),xlabel='Diagnostic time (s; begins after 4 s startup)',ylabel='Max requested joint torque (Nm)');ax.grid(alpha=.15);ax.set_title(f'All 8,801 physics samples · peak {tau.max():.3f} Nm',loc='left');ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.suptitle('Both diagonal pairs passed the static load-transfer checks',fontsize=16,fontweight='bold')
fig.supxlabel('Actual full-C physics • fixed body-motion command • no walking/PPO admission • startup transients retained separately',fontsize=10)
out=H/'actual_diagonal_loads.png'
if out.exists():raise FileExistsError(out)
fig.savefig(out,dpi=180);plt.close(fig)
