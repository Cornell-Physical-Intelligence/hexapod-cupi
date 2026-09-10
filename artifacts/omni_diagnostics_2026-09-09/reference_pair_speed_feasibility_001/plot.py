from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
H=Path(__file__).resolve().parent;summary=json.loads((H/'SUMMARY.json').read_text());cases=summary['cases'];speeds=[.02,.025,.03,.05];swings=[2.,1.5,1.,.75,.5]
code={'joint_residual_margin':1,'reference_velocity':2,'reference_acceleration':3};labels=['Pass','Joint reserve','Velocity','Acceleration'];colors=['#b5d6c4','#edbb8f','#d7badf','#e4a2a2'];mat=np.zeros((4,5),int)
fig,(ax,rate)=plt.subplots(1,2,figsize=(13.5,5.7),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
for i,s in enumerate(speeds):
 for j,t in enumerate(swings):
  c=next(c for c in cases if c['parameters']['requested_forward_left_yaw'][0]==s and c['parameters']['swing_s']==t);cross=c['diagnostic']['first_crossing'];mat[i,j]=0 if cross is None else code[cross['categories'][0]]
  label='PASS\n44 s' if cross is None else labels[mat[i,j]]+'\n'+f"{cross['time_s']:.2f} s"
  ax.text(j,i,label,ha='center',va='center',fontsize=10)
ax.imshow(mat,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4.5),4),aspect='auto')
ax.set(xticks=range(5),xticklabels=[f'{t:g}' for t in swings],yticks=range(4),yticklabels=[f'{s:.3f}' for s in speeds],xlabel='Proposed swing duration (s)',ylabel='Requested forward speed (m/s)',title='First target constraint reached')
ax.set_xticks(np.arange(-.5,5,.5),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
for speed,swing,label,color in [(.015,2.,'.015 m/s · 2 s','#437c61'),(.02,2.,'.020 m/s · 2 s','#24649a'),(.02,1.5,'.020 m/s · 1.5 s','#9c5a29')]:
 p=next(c for c in cases if c['parameters']['requested_forward_left_yaw'][0]==speed and c['parameters']['swing_s']==swing)
 with np.load(H/'results'/(p['parameters']['candidate_id']+'.npz'),allow_pickle=False) as z:
  t=z['time_s']-26;mask=(t>=0)&(t<=8);rate.plot(t[mask],z['command'][mask,0],label=label,color=color,linewidth=2.1)
 rate.scatter([p['result']['reference_stop_latency_s']],[0],color=color,zorder=4)
rate.set(xlabel='Seconds after stop request',ylabel='Filtered planned forward speed (m/s)',title='Unchanged stop filter still takes ~6 s',xlim=(0,8),ylim=(-.001,.021));rate.grid(alpha=.22);rate.legend(frameon=False,loc='upper right',fontsize=9)
fig.suptitle('C geometry: paired target speed study, 7 mm lift',fontsize=17)
fig.supxlabel('CPU planned endpoints only. No new measured flight, support, torque or walking admission.',fontsize=11)
fig.savefig(H/'speed_constraints.png',dpi=180)
