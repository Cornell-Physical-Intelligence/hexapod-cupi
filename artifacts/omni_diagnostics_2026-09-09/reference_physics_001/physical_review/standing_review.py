"""Measured standing001 load distribution and frozen-target geometry analysis."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/reference_physics_adapter_001/source_001/tools'))
from serial_geometry import SerialGeometry,tensor
G=SerialGeometry();LEGS=['lf','lm','lr','rf','rm','rr']
P=ROOT/'tmp/reference_physics_results_001/run/standing/trace.npz';d=np.load(P)
sel=slice(200,None);c=d['distal_contact'][sel];f=d['normal_force_world_n'][sel,:,:,2]
qt=d['joint_target_leg_major_rad'][0];dq=qt-G.q0.numpy()[None]
ft=G.fk(tensor(qt))[0].numpy();fm=d['reference_point_world_m'][sel]
R=d['rotation_world_from_body'][sel];p=d['position_world_m'][sel]
virtual=np.einsum('tnij,nlj->tnli',R,ft)+p[:,:,None,:]
preload=virtual-fm
rpy=Rotation.from_quat(d['quaternion_world_xyzw'][sel].reshape(-1,4)).as_euler('xyz').reshape(800,32,3)
rows=[]
for e in range(32):
    legs=[]
    for i,leg in enumerate(LEGS):
        legs.append(dict(leg=leg,support_fraction=float(c[:,e,i].mean()),force_z_n_quantiles=np.percentile(f[:,e,i],[0,50,100]).tolist(),mean_reference_point_world_z_m=float(fm[:,e,i,2].mean()),mean_virtual_minus_measured_point_z_m=float(preload[:,e,i,2].mean()),joint_target_minus_nominal_rad=dq[e,i].tolist(),target_fk_body_z_m=float(ft[e,i,2])))
    rows.append(dict(replica=e,all_six_support_entire_window=bool(c[:,e].all()),min_support_count=int(c[:,e].sum(-1).min()),mean_root_height_m=float(p[:,e,2].mean()),root_height_peak_to_peak_m=float(np.ptp(p[:,e,2])),max_abs_roll_pitch_rad=float(abs(rpy[:,e,:2]).max()),mean_total_normal_force_n=float(f[:,e].sum(-1).mean()),target_fk_body_z_span_m=float(np.ptp(ft[e,:,2])),legs=legs))
report=dict(scope='Actual frozen reference001 zero-residual standing result; wave not launched.',trace_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),post_settle_interval_s=[float(d['time_s'][200,0]),float(d['time_s'][-1,0])],samples_per_replica=800,passing_six_support_replicas=[r['replica'] for r in rows if r['all_six_support_entire_window']],failed_six_support_replicas=[r['replica'] for r in rows if not r['all_six_support_entire_window']],max_joint_target_range_rad=float(np.ptp(d['joint_target_rad'],axis=0).max()),target_minus_nominal_extrema_rad=[float(dq.min()),float(dq.max())],nominal_target_fk_body_z_span_m=float(np.ptp(G.feet0.numpy()[:,2])),failed_support_force_z_n_quantiles=np.percentile(f[~c],[0,25,50,75,100]).tolist(),measured_reference_height_caution='Named tibia-local reference point height is not mesh ground clearance.',replicas=rows)
out=Path(__file__).parent
(out/'standing_review.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(12,8),layout='constrained')
for ax,values,title,cmap,low,high in [(axes[0],c.mean(0),'Measured support ≥1 N\nfraction of 4–20 s', 'viridis',0,1),(axes[1],f.mean(0),'Mean measured vertical\ncontact force (N)','magma',0,34),(axes[2],preload[:,:,:,2].mean(0)*1000,'Virtual target toe minus\nmeasured toe Z (mm)','coolwarm',-3,3)]:
    im=ax.imshow(values,aspect='auto',origin='upper',cmap=cmap,vmin=low,vmax=high)
    ax.set_title(title);ax.set_xticks(range(6),LEGS);ax.set_yticks(range(32),range(32));ax.set_ylabel('Replica');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Reference001 rejected standing: constant randomized targets unload feet\n17/32 retained six supports; no walking attempt occurred',fontsize=13)
fig.savefig(out/'standing_review.png',dpi=170)
print(json.dumps({k:v for k,v in report.items() if k!='replicas'},indent=2))
