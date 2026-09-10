"""Compare only two measured runs; no synthetic force/trajectory extension."""
from pathlib import Path
import argparse,json,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--original-trace',type=Path,required=True);parser.add_argument('--output-image',type=Path,required=True);args=parser.parse_args()
if args.output_image.exists():raise FileExistsError(args.output_image)
P=args.raw;new=np.load(P/'run/left_strafe/trace.npz');old=np.load(args.original_trace);sub=np.load(P/'run/left_strafe/physics_substeps.npz');refs=json.loads((P/'run/left_strafe/reference_states.json').read_text())
t=new['time_s'][:,0];ot=old['time_s'][:,0];mask=t>=7.8;omask=ot>=7.8
fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
ax[0,0].plot(ot[omask],old['normal_force_world_n'][omask,0,5,2],label='Original RR');ax[0,0].plot(t[mask],new['normal_force_world_n'][mask,0,5,2],label='New RR');ax[0,0].axhline(1,color='r',ls='--',label='1 N support threshold');ax[0,0].axvspan(8.4,8.7,alpha=.15,color='green',label='Correction');ax[0,0].axvline(10.1,color='gray',ls=':');ax[0,0].set(title='RR remains loaded beyond original rejection',ylabel='Measured normal force (N)');ax[0,0].legend(fontsize=8)
for i in [1,3,5]:ax[0,1].plot(t[mask],new['normal_force_world_n'][mask,0,i,2],label=['LF','LM','LR','RF','RM','RR'][i])
ax[0,1].axhline(1,color='r',ls='--');ax[0,1].axvline(11.98,color='gray',ls=':');ax[0,1].set(title='New rejection: LM < 1 N while RF unloads',ylabel='Measured normal force (N)');ax[0,1].legend()
st=sub['time_s'];qnames=list(sub['joint_names']);j=qnames.index('revolute_2_3');sm=(st>=11.35)&(st<=11.8);cm=(t>=11.35)&(t<=11.8)
ax[1,0].plot(st[sm],sub['computed_torque_nm'][sm,0,j],label='400 Hz requested',lw=1);ax[1,0].scatter(t[cm],new['computed_torque_nm'][cm,0,j],label='50 Hz requested',s=15,color='black');ax[1,0].axhline(1.6,color='r',ls='--',label='1.6 N m');ax[1,0].set(title='RF tibia: two brief requested-torque exceedances',ylabel='N m');ax[1,0].legend(fontsize=8)
times=[];offset=[]
for row in refs:
 if 'result'not in row:continue
 r=row['result'];times.append(r['target_time_s']);offset.append(r['diagnostics']['rr_preload_diagnostic']['offset_applied_world_m'][2]*1000)
ax[1,1].plot(times,offset);ax[1,1].axvspan(8.4,8.7,alpha=.15,color='green');ax[1,1].set(xlim=(8.25,8.85),title='Recorded one-shot correction follows declared C2',ylabel='RR world-anchor offset (mm)')
for a in ax.flat:a.set_xlabel('Physical time (s)');a.grid(alpha=.2)
fig.suptitle('RR preload002: measured local improvement, overall trial rejected\n3 landings / 599 controls; no completed motion or quiet-stop phase',fontsize=14)
fig.savefig(args.output_image,dpi=150)
