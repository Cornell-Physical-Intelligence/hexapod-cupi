"""Plot actual pair001 measurements, preserving their original timebases and hashes."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'reference_pair_results_001/run/pair'
with np.load(INPUT/'trace.npz') as z: t={k:z[k].copy() for k in z.files}
with np.load(INPUT/'physics_substeps.npz') as z: sub={k:z[k].copy() for k in z.files}
reset=json.loads((INPUT/'pair_reset.json').read_text())
state=json.loads((INPUT/'state.json').read_text())
initial=np.asarray(reset['state']['initial_measured_toes_world_m'])
x=t['time_s'][:,0]-4.0; sx=sub['time_s'];assert len(x)==1100 and len(sx)==8801
assert np.allclose(x,np.arange(1,1101)*.02,atol=1e-9)
assert np.allclose(sx,np.arange(8801)*.0025,atol=1e-9)
lift=(t['reference_point_world_m'][:,0,:,2]-initial[:,2])*1000
request=np.abs(sub['computed_torque_nm'][:,0]).max(axis=1)
applied=np.abs(sub['applied_torque_nm'][:,0]).max(axis=1)
assert float(request.max())==state['diagnostic_result']['substep_review']['max_requested_torque_nm']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,axes=plt.subplots(3,1,figsize=(11.5,8.6),sharex=True,gridspec_kw={'height_ratios':[1,1.2,1]})
fig.subplots_adjust(top=.88,bottom=.10,hspace=.24,left=.09,right=.98)
fig.suptitle('Full C robot: four-foot support during a paired lift',x=.09,ha='left',fontsize=18,fontweight='bold',y=.975)
fig.text(.09,.925,'Actual Isaac measurements · LM/RM lift and return · diagnostic only; no walking policy',fontsize=11,color='#475569')
for ax in axes:
 ax.axvspan(5,7,color='#cbd5e1',alpha=.30,zorder=0)
 ax.axvspan(12,22,color='#d1fae5',alpha=.35,zorder=0)
 ax.grid(axis='y',color='#e2e8f0',linewidth=.6);ax.set_xlim(0,22)
for j,label,color in [(1,'Left middle','#2563eb'),(4,'Right middle','#db2777')]:axes[0].plot(x,lift[:,j],label=label,color=color,lw=2)
axes[0].axhline(2,color='#64748b',ls=':',lw=1,label='2 mm lift guide')
axes[0].set_ylabel('Toe-reference lift\nfrom initial pose (mm)');axes[0].legend(loc='upper right',ncol=3,frameon=False,fontsize=9)
labels=['LF','LM','LR','RF','RM','RR'];colors=['#7c3aed','#2563eb','#ea580c','#0f766e','#db2777','#475569']
for j,(label,color) in enumerate(zip(labels,colors)):axes[1].plot(x,t['reaction_force_world_n'][:,0,j,2],label=label,color=color,lw=1.35,ls='--' if j in (1,4) else '-')
axes[1].set_ylabel('Vertical foot reaction (N)');axes[1].legend(loc='upper right',ncol=6,frameon=False,fontsize=9)
axes[2].plot(sx,request,color='#0f172a',lw=1.6,label='Maximum absolute requested / applied torque')
assert np.array_equal(request,applied)
axes[2].axhline(1.6,color='#b91c1c',ls='--',lw=1.2,label='1.6 N·m diagnostic limit')
axes[2].set_ylim(0,1.8);axes[2].set_ylabel('Motor torque (N·m)');axes[2].set_xlabel('Seconds after the retained 4-second startup')
axes[2].legend(loc='lower left',frameon=False,fontsize=9)
fig.text(.09,.035,'Gray: planned unloaded hold. Green: scored quiet hold. Contacts at 50 Hz; torque at every 400 Hz substep.',fontsize=10,color='#475569')
fig.savefig(HERE/'actual_pair.png',dpi=160,facecolor='white');plt.close(fig)
report={'input_sha256':{n:hashlib.sha256((INPUT/n).read_bytes()).hexdigest() for n in ['trace.npz','physics_substeps.npz','pair_reset.json','state.json']},'control_samples':len(x),'substep_samples':len(sx),'lift_peak_mm':lift[:,[1,4]].max(axis=0).tolist(),'peak_requested_nm':float(request.max()),'requested_applied_maxima_identical':True,'startup_excluded_from_plot_s':4,'not_walking_or_policy_qualification':True}
(HERE/'plot_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}))
