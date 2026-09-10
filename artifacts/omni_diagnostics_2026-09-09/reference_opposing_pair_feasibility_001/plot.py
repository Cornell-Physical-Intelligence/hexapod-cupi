"""Static scientific summary of planned CPU targets, never actual gait telemetry."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;r=json.loads((P/'report.json').read_text())
with np.load(P/'planned_targets.npz') as z:d={k:z[k].copy() for k in z.files}
fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
fig.suptitle('Opposing-pair target feasibility at 0.01 m/s — CPU plan, no contact physics',fontsize=14)
t=d['time_s'];colors=['#007F80','#BC5B00','#6741A5'];labels=['LM + RM','LF + RR','LR + RF']
axs[0].plot(t,np.max(abs(d['v']),axis=1),label='Maximum executable |velocity|',color='#007F80');axs[0].axhline(1.75,color='#AD2626',linestyle='--',label='Reference limit: 1.75 rad/s');axs[0].set_ylabel('rad/s');axs[0].legend(loc='upper right',fontsize=8)
axs[1].plot(t,np.max(abs(d['a']),axis=1),label='Maximum executable |acceleration|',color='#007F80');axs[1].axhline(6,color='#AD2626',linestyle='--',label='Reference limit: 6 rad/s²');axs[1].set_ylabel('rad/s²');axs[1].legend(loc='upper right',fontsize=8)
for j in range(3):
 mask=d['active_pair_index']==j;y=np.where(mask,d['projected_margin_m']*1000,np.nan);axs[2].plot(t,y,label=labels[j],color=colors[j])
axs[2].axhline(50,color='#AD2626',linestyle='--',label='Proposed margin: 50 mm');axs[2].set_ylabel('Planned support\nmargin (mm)');axs[2].set_xlabel('Planned time (s)');axs[2].legend(loc='upper right',fontsize=8,ncol=2)
for ax in axs:
 ax.axvspan(0,2,color='gray',alpha=.1);ax.axvspan(26,44,color='gray',alpha=.1);ax.grid(alpha=.2);ax.set_xlim(0,44)
fig.savefig(P/'candidate_targets.png',dpi=160)
