"""Read-only actual directional002 analysis. No physics writes or gate changes."""
from pathlib import Path
import argparse, hashlib, json, sys, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
RAW=ROOT/'tmp/reference_directional_results_002'
SRC=ROOT/'tmp/reference_directional_adapter_002/source_directional_002'
sys.path.insert(0,str(SRC/'tools'))
from serial_geometry import SerialGeometry,tensor
from wave_math import MassGeometry

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def fk(g,q,pos,R):
 f,J,Ts=g.fk(tensor(q));f=f.numpy();Ts=[T.numpy() for T in Ts]
 world=np.einsum('nij,nlj->nli',R,f)+pos[:,None,:]
 return world,Ts,J.numpy()

def com(m,Ts,pos,R):
 b=np.tile(m.body_mass*m.body_com,(len(pos),1))
 for j,T in enumerate(Ts):
  x=np.einsum('nlij,lj->nli',T[...,:3,:3],m.link_com[:,j])+T[...,:3,3]
  b+=(x*m.link_mass[:,j,None]).sum(1)
 return pos+np.einsum('nij,nj->ni',R,b/m.mass)

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(exist_ok=False,parents=True)
 audit=json.loads((RAW/'remote_audit.json').read_text());verified={p:sha(RAW/p)==h for p,h in audit['raw_payloads'].items()};assert all(verified.values())
 z=np.load(RAW/'run/left_strafe/trace.npz');s=np.load(RAW/'run/left_strafe/physics_substeps.npz');rows=json.loads((RAW/'run/left_strafe/reference_states.json').read_text());refs={e['physical_step']:e['result'] for e in rows if 'result'in e}
 t=z['time_s'][:,0];n=len(t);legs=list(z['legs']);assert legs==['lf','lm','lr','rf','rm','rr']
 g=SerialGeometry();m=MassGeometry(g);names=list(z['joint_names']);order=[names.index(x) for ns in g.names for x in ns]
 q=z['joint_position_leg_major_rad'][:,0];pos=z['position_world_m'][:,0];R=z['rotation_world_from_body'][:,0];feet=z['reference_point_world_m'][:,0];cp=z['contact_point_world_m'][:,0];valid=z['contact_point_valid'][:,0];dist=z['distal_contact'][:,0];F=z['normal_force_world_n'][:,0,:,2]
 np.testing.assert_array_equal(q,z['joint_position_rad'][:,0,order].reshape(n,6,3))
 feetfk,Ts,J=fk(g,q,pos,R);cm=com(m,Ts,pos,R)
 targetfk,_,_=fk(g,z['joint_target_leg_major_rad'][:,0],pos,R)
 st=s['time_s'];st=st[:,0] if st.ndim>1 else st
 sp=s['root_link_position_world_m'][:,0];sr=Rotation.from_quat(s['root_link_quaternion_world_xyzw'][:,0]).as_matrix();sq=s['joint_position_rad'][:,0,order].reshape(-1,6,3)
 sfk,sTs,_=fk(g,sq,sp,sr);scm=com(m,sTs,sp,sr)
 # Static support projection only; contacts remain 50 Hz and are not interpolated.
 margins={key:np.array([m.support_margin(cp[k],ids(k),cm[k]) for k in range(n)]) for key,ids in {
  'threshold_actual':lambda k:np.flatnonzero(dist[k]&valid[k]),
  'five_excluding_lm_contact_geometry':lambda k:np.array([0,2,3,4,5]) if valid[k,[0,2,3,4,5]].all() else [],
  'four_excluding_lm_rr_contact_geometry':lambda k:np.array([0,2,3,4]) if valid[k,[0,2,3,4]].all() else [],
 }.items()}
 # RR mesh minimum in URDF reconstruction, not a PhysX signed-distance measurement.
 import trimesh
 xml=ET.parse(g.urdf_path).getroot();joints={j.get('name'):j for j in xml.findall('joint')};linkname=joints[g.names[5][2]].find('child').get('link');link=next(x for x in xml.findall('link') if x.get('name')==linkname)
 collision=link.find('collision');mesh=collision.find('geometry/mesh');origin=collision.find('origin');meshfile=SRC/'robot/hexapod_mkii_length_study/meshes'/Path(mesh.get('filename')).name
 verts=np.asarray(trimesh.load_mesh(meshfile,process=False).vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
 oR=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix();op=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');verts=verts@oR.T+op
 distalverts=verts[verts[:,1]>=.126*.93];assert len(distalverts)>0
 def minz(v):
  T=Ts[2][:,5];worldR=R@T[:,:3,:3];worldp=pos+np.einsum('nij,nj->ni',R,T[:,:3,3]);return (worldR[:,2,:]@v.T+worldp[:,2,None]).min(1)
 meshmin=minz(verts);distalmin=minz(distalverts)
 desired=np.full_like(pos,np.nan);virt=np.full_like(feet,np.nan);preload=np.full_like(feet,np.nan);mode={};events=[];prev=None
 for k,r in refs.items():
  np.testing.assert_allclose(r['target_time_s'],t[k],atol=3e-12,rtol=0)
  np.testing.assert_allclose(np.array(r['q_ref'])[0].astype(np.float32),z['joint_target_rad'][k,0],atol=0,rtol=0)
  state=r['state'];desired[k]=state['desired_position_world_m'];virt[k]=r['diagnostics']['reference_point_world_m'];preload[k]=state['reference_minus_measured_preload_world_m'];mode[k]=[state['mode'],state['current_leg']]
  now=(state['mode'],state['current_leg'],state['confirmed_touchdowns'])
  if now!=prev:events.append(dict(control_index=k,time_s=t[k],mode=now));prev=now
 rr=5; sample_indices=[199,317,330,333,348,391,392,393,405,419,420,435,436,453,454,464,480,495,503,504]
 def row(k):
  return dict(control_index=k,time_s=t[k],mode=mode.get(k),force_z_n=F[k],support=dist[k],whole_robot_com_world_m=cm[k],support_margin_m={x:y[k] for x,y in margins.items()},body_position_m=pos[k],body_rpy_deg=Rotation.from_matrix(R[k]).as_euler('xyz',degrees=True),desired_body_position_m=desired[k],RR=dict(toe_reference_world_m=feet[k,rr],toe_reference_velocity_world_mps=z['reference_point_velocity_world_mps'][k,0,rr],actual_contact_point_world_m=cp[k,rr],contact_point_valid=valid[k,rr],reaction_force_n=z['reaction_force_world_n'][k,0,rr],friction_xy_n=z['friction_force_xy_n'][k,0,rr],slip_mps=z['distal_contact_slip_mps'][k,0,rr],virtual_reference_world_m=virt[k,rr],target_FK_using_measured_body_m=targetfk[k,rr],latched_preload_m=preload[k,rr],joint_q_rad=q[k,rr],joint_target_rad=z['joint_target_leg_major_rad'][k,0,rr],joint_computed_torque_nm=z['computed_torque_nm'][k,0,order].reshape(6,3)[rr],URDF_reconstructed_mesh_min_z_m=meshmin[k],URDF_reconstructed_distal_mesh_min_z_m=distalmin[k]))
 # fixed contact polygon evaluated against 400 Hz reconstructed COM, never called an actual 400 Hz support measurement.
 last=np.arange(n-100,n);submask=(st>=t[last[0]]-.02-1e-10)&(st<=t[-1]+1e-10);peak=np.abs(s['computed_torque_nm'][submask,0]);idx=np.unravel_index(peak.argmax(),peak.shape);si=np.flatnonzero(submask)[idx[0]]
 w={}
 for label,ix in [('last100',last),('RR_landing_hold',np.arange(420,436)),('LM_force_free',np.arange(454,505))]:
  w[label]=dict(index_first=int(ix[0]),index_last=int(ix[-1]),time_first_s=t[ix[0]],time_last_s=t[ix[-1]],RR_force_min_n=F[ix,rr].min(),RR_force_mean_n=F[ix,rr].mean(),RR_force_max_n=F[ix,rr].max(),all_force_begin_n=F[ix[0]],all_force_end_n=F[ix[-1]],body_delta_m=pos[ix[-1]]-pos[ix[0]],RR_toe_delta_m=feet[ix[-1],rr]-feet[ix[0],rr],RR_contact_point_delta_m=cp[ix[-1],rr]-cp[ix[0],rr],RR_target_FK_measured_body_delta_m=targetfk[ix[-1],rr]-targetfk[ix[0],rr],margins_min_m={x:y[ix].min() for x,y in margins.items()},body_rpy_delta_deg=(Rotation.from_matrix(R[ix[-1]])*Rotation.from_matrix(R[ix[0]]).inv()).as_rotvec()*180/np.pi)
 report=dict(scope='Actual failed left-strafe diagnostic; unchanged 1 N / five-support gates; no new physical admission',raw_verified=len(verified),source_manifest_sha256=audit['source_manifest_sha256'],mass_kg=m.mass,urdf_sha256=g.urdf_sha,mesh_sha256=sha(meshfile),reference_time_and_exact_float32_target_alignment=True,contact_rate_hz=50,substep_rate_hz=400,contact_forces_400Hz_available=False,FK_to_SDK_toe_max_error_m=np.linalg.norm(feetfk-feet,axis=-1).max(),FK_to_SDK_toe_RR_max_error_m=np.linalg.norm(feetfk[:,rr]-feet[:,rr],axis=-1).max(),com_semantics='Whole articulated COM from 19 frozen URDF masses, actual named joint angles and raw XYZW root-link pose; no measured per-link COM storage',mesh_height_semantics='URDF collision mesh vertices transformed using measured root/joints; no PhysX signed-distance or actual penetration readback. Includes reconstruction discrepancy with SDK link poses.',events=events,samples=[row(k) for k in sample_indices],windows=w,last100_400Hz=dict(count=int(submask.sum()),first_s=st[submask][0],last_s=st[submask][-1],maximum_requested_torque_nm=peak.max(),peak_time_s=st[si],peak_joint=names[idx[1]],whole_robot_com_world_min_m=scm[submask].min(0),whole_robot_com_world_max_m=scm[submask].max(0)),final_reference_state=refs[504]['state'])
 (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
 (out/'last100.json').write_text(json.dumps(clean([row(int(k)) for k in last]),indent=2,allow_nan=False)+'\n')
 np.savez_compressed(out/'derived_arrays.npz',time_s=t,whole_robot_com_world_m=cm,target_FK_measured_body_world_m=targetfk,URDF_mesh_min_z_m=meshmin,URDF_distal_mesh_min_z_m=distalmin,**{'margin_'+k:v for k,v in margins.items()})
 print(json.dumps(clean({k:report[k] for k in ['raw_verified','mass_kg','FK_to_SDK_toe_max_error_m','FK_to_SDK_toe_RR_max_error_m','windows','last100_400Hz']}),indent=2))
if __name__=='__main__':main()
