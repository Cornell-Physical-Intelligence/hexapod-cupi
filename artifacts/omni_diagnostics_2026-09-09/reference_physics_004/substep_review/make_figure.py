from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Fresh figure required')
 fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
 for ax,phase,e,title in zip(axes,['standing','wave'],[3,0],['Standing · environment 3 · largest accumulated discrepancy','Wave prefix · environment 0 · rejected before completion']):
  with np.load(a.run/phase/'physics_substeps.npz') as d,np.load(a.run/phase/'trace.npz') as t:
   n=len(t['position_world_m']);pos=d['root_link_position_world_m'][1600:,e].astype(float);vel=d['root_link_velocity_world_mps'][1600:,e].astype(float)
   axis=-t['rotation_world_from_body'][199,e,:,1].astype(float);axis[2]=0;axis/=np.linalg.norm(axis)
   times=np.arange(len(pos))*.0025;actual=(pos-pos[0])@axis
   integrated=np.r_[0,np.cumsum(.5*(vel[1:]+vel[:-1])@axis*.0025)]
   control=np.r_[0,np.cumsum(vel[8::8]@axis*.02)]
   ax.plot(times,actual*1000,label='Actual pose change',color='#17605a',lw=2)
   ax.plot(times,integrated*1000,label='Integrated 400 Hz reported velocity',color='#bf5631',lw=1.5)
   ax.plot(np.arange(n-200+1)*.02,control*1000,label='Integrated 50 Hz last-substep velocity',color='#7445a5',ls='--',lw=1.1)
   ax.set(title=title,xlabel='Seconds after 4 s settle',ylabel='Forward displacement (mm)');ax.grid(alpha=.18);ax.legend(loc='best',fontsize=9)
 fig.suptitle('Pose change and solver-reported velocity are different measurements\nActual 004 · diagnostic only · original acceptance gates unchanged',fontsize=13)
 fig.savefig(a.output,dpi=170)
if __name__=='__main__':main()
