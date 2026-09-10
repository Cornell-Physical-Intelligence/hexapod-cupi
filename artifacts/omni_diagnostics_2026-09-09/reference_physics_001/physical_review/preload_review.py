"""Read previous pilot's initial-policy trace to bound swing clearance risk."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/reference_physics_adapter_001/source_001/tools'))
from serial_geometry import SerialGeometry,tensor
G=SerialGeometry()
P=ROOT/'tmp/omni_velocity_pilot_003_review/initial/diagnostic_trace.npz'
d=np.load(P);names=list(d['joint_names']);order=[names.index(n) for row in G.names for n in row]
meas=d['joint_position_rad'][:,:,order].reshape(-1,12,6,3)
targ=d['joint_target_rad'][:,:,order].reshape(-1,12,6,3)
fm,_,_=G.fk(tensor(meas));ft,_,_=G.fk(tensor(targ))
R=Rotation.from_quat(d['quaternion_world_xyzw'].reshape(-1,4)).as_matrix().reshape(-1,12,3,3)
off=np.einsum('tnij,tnlj->tnli',R,(ft-fm).numpy())
sel=(d['time_s']>=3.)&(d['time_s']<=4.)
print('times',d['time_s'][sel][[0,-1]],'points',sum(sel))
print('range vertical offset all selected mm',np.percentile(off[sel,:,:,2]*1000,[0,25,50,75,100]))
print('env0 legoffset mm at4',off[sel][-1,0]*1000)
print('env0velocity',d['velocity_navigation_mps'][sel][-1,0])
print('targets initialmaxmotion',abs(d['executable_target_velocity_rad_s'][sel]).max())
# The swing is below its nominal peak for most of its duration.
u=np.linspace(0,1,100001);h=.005*64*(u**3-3*u**4+3*u**5-u**6)
thresholds=[.0005,.001,.002,.003,.004,.0045]
interval=[]
for z in thresholds:
    ix=np.where(h>=z)[0];interval.append(dict(required_clearance_m=z,first_time_s=float(2*u[ix[0]]),last_time_s=float(2*u[ix[-1]])))
report=dict(scope='Previous initial-policy trace used only as empirical loaded-deflection context, not current reference-screen measurement. Same frozen serial geometry and Kp30; actual source001 standing will decide.',trace=str(P.relative_to(ROOT)),trace_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),time_window_s=[3.,4.],leg_names=['lf','lm','lr','rf','rm','rr'],virtual_minus_actual_world_z_mm_percentiles=np.percentile(off[sel,:,:,2]*1000,[0,25,50,75,100]).tolist(),env0_virtual_minus_actual_world_m_at4=off[sel][-1,0].tolist(),maximum_initial_policy_target_velocity_rad_s=float(abs(d['executable_target_velocity_rad_s'][sel]).max()),swing_clearance_intervals=interval)
(Path(__file__).parent/'preload_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(interval)
