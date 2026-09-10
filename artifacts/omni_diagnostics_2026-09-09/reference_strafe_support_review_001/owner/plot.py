"""Static diagnostic from recorded 50 Hz forces and derived geometry. No simulation."""
from pathlib import Path
import argparse,json,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--output-image',type=Path,required=True);args=ap.parse_args();assert not args.output_image.exists(), 'Refuse to overwrite image'
z=np.load(P.parents[1]/'tmp/reference_directional_results_002/run/left_strafe/trace.npz');d=np.load(args.analysis/'derived_arrays.npz');r=json.loads((args.analysis/'report.json').read_text())
t=d['time_s'];n=len(t);F=z['normal_force_world_n'][:,0,:,2];feet=z['reference_point_world_m'][:,0];legs=list(z['legs']);sel=np.arange(390,n)
rr=[x for x in r['samples'] if x['control_index']>=391]
fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for l in range(6):ax[0,0].plot(t[sel],F[sel,l],label=legs[l].upper())
ax[0,0].axhline(1,color='k',ls='--',lw=1);ax[0,0].set(title='Measured normal forces (50 Hz)',ylabel='N');ax[0,0].legend(ncol=3)
ax[0,1].plot(t[sel],F[sel,5],label='RR');ax[0,1].axhline(1,color='r',ls='--',label='Unchanged support threshold');ax[0,1].set(title='RR unloads as LM swings',ylabel='N');ax[0,1].legend()
ax[1,0].plot(t[sel],feet[sel,5,2]*1000,label='Actual toe reference point');ax[1,0].plot(t[sel],d['target_FK_measured_body_world_m'][sel,5,2]*1000,label='Target FK in measured body');ax[1,0].plot([x['time_s'] for x in rr],[x['RR']['virtual_reference_world_m'][2]*1000 for x in rr],label='Virtual world target');ax[1,0].plot(t[sel],d['URDF_distal_mesh_min_z_m'][sel]*1000,label='URDF distal mesh min (reconstructed)',ls=':');ax[1,0].set(title='RR heights; none is a penetration sensor',ylabel='mm');ax[1,0].legend(fontsize=8)
for key in ['five_excluding_lm_contact_geometry','four_excluding_lm_rr_contact_geometry']:ax[1,1].plot(t[sel],d['margin_'+key][sel]*1000,label=key.split('_')[0]+' contact points')
ax[1,1].axhline(25,color='r',ls='--',label='Reference margin threshold');ax[1,1].set(title='Projected whole-robot COM margin; geometry only',ylabel='mm');ax[1,1].legend()
for a in ax.flat:
 for ti in [8.42,8.74,9.10]:a.axvline(ti,color='gray',alpha=.25)
 a.set_xlabel('Physical time (s)');a.grid(alpha=.2)
fig.suptitle('Directional002 rejected: RR < 1 N during LM swing\nStatic support area remains large; load distribution is not guaranteed',fontsize=13)
fig.savefig(args.output_image,dpi=150)
