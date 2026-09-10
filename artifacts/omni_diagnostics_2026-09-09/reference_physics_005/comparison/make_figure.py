from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
p=Path(__file__).resolve().parent
c=json.loads((p/'matched_comparison.json').read_text())['link']['400Hz_trapezoid'];q=json.loads((p/'quiet_replay.json').read_text())['trials']
x=np.arange(32);fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
for ax,vals,label,bound in [(axes[0],[np.array(c['baseline004_every_env_m'])*1000,np.array(c['comparison005_every_env_m'])*1000],'Position / velocity-integral difference (mm)',5),(axes[1],[[r['max_joint_velocity_rms_rad_s'] for r in q[k]['quiet_rows']] for k in ['004','005']],'Maximum reported joint velocity RMS (rad/s)',.03)]:
 ax.bar(x-.19,vals[0],width=.38,color='#17605a',label='004 · force timing false')
 ax.bar(x+.19,vals[1],width=.38,color='#bf5631',label='005 · force timing true')
 ax.axhline(bound,color='#971b36',ls='--',lw=1,label='Unchanged 5 mm diagnostic' if ax is axes[0] else 'Unchanged quiet gate')
 ax.set(ylabel=label,xlabel='Matched environment ID',xticks=x);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.17);ax.legend(fontsize=9,loc='upper right')
fig.suptitle('Actual 005 improved base consistency but failed quiet hold in 2 / 32 robots\nStanding-only comparison · no solver adoption or walking admission',fontsize=13)
fig.savefig(p/'all_environment_comparison.png',dpi=170)
