from pathlib import Path
import unittest,tempfile,json,sys,types,importlib.util,ast
import numpy as np
from standing_math import Geometry,servo,classify_contacts
from standing_session import NativeStandingSession
from inspect_core import rotation
HERE=Path(__file__).parent
META=json.loads((HERE/'geometry/geometry.json').read_text())
with np.load(HERE/'geometry/geometry_extrema.npz')as z:CLOUDS={k:z[k]for k in z.files}
MODEL=json.loads(Path('artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/source/model.json').read_text())
NAMES=[x['name']for x in MODEL['joints']];BODIES=[x['name']for x in MODEL['links']]

def frame_poses(q,root):
 # Independent CPU model frame composition; fake native physics is explicitly synthetic.
 poses={'body':(root[:3],rotation(root[3:]))};remaining=list(MODEL['joints'])
 while remaining:
  for j in remaining[:]:
   if j['parent']not in poses:continue
   p,R=poses[j['parent']];a=q[NAMES.index(j['name'])];Z=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
   poses[j['child']]=(p+R@j['xyz'],R@rotation(j['quaternion_xyzw'])@Z);remaining.remove(j)
 from scipy.spatial.transform import Rotation
 return np.array([np.r_[poses[n][0],Rotation.from_matrix(poses[n][1]).as_quat()]for n in BODIES])

class A:
 def __init__(self,x):self.x=np.array(x)
 def numpy(self):return self.x
class View:
 count=1;prim_paths=['/Robot/body']
 def __init__(self):
  self.q=np.zeros((1,18),np.float32);self.dq=self.q.copy();self.f=self.q.copy();self.pt=self.q.copy();self.vt=self.q.copy();self.root=np.array([[0,0,.08161109101311194,0,0,0,1]],np.float32);self.rv=np.zeros((1,6),np.float32)
 def get_dof_positions(self):return A(self.q)
 def get_dof_velocities(self):return A(self.dq)
 def get_dof_actuation_forces(self):return A(self.f)
 def get_dof_projected_joint_forces(self):return A(self.f)
 def get_dof_position_targets(self):return A(self.pt)
 def get_dof_velocity_targets(self):return A(self.vt)
 def get_root_transforms(self):return A(self.root)
 def get_root_velocities(self):return A(self.rv)
 def get_link_transforms(self):return A([frame_poses(self.q[0],self.root[0])])
 def set_dof_positions(self,a,i):self.q=a.numpy().copy()
 def set_dof_velocities(self,a,i):self.dq=a.numpy().copy()
 def set_dof_actuation_forces(self,a,i):self.f=a.numpy().copy()
 def set_dof_position_targets(self,a,i):self.pt=a.numpy().copy()
 def set_dof_velocity_targets(self,a,i):self.vt=a.numpy().copy()
 def set_root_transforms(self,a,i):self.root=a.numpy().copy()
 def set_root_velocities(self,a,i):self.rv=a.numpy().copy()
class Sim:
 def __init__(self,v):self.v=v;self.counter=31;self.physics_sim_view=self
 def get_physics_step_count(self):return self.counter
 def update_articulations_kinematic(self):pass
 def step(self,render=False):self.counter+=1
class Contact:
 filter_count=1;filter_paths=['/Ground'];max_contact_data_count=1024
 def __init__(self,v):self.v=v;self.sensor_paths=['/Robot/'+b for b in BODIES]
 def get_contact_data(self,dt):
  f=np.zeros((1024,1));p=np.zeros((1024,3));n=np.zeros((1024,3));d=np.zeros((1024,1));counts=np.zeros((19,1),np.uint32);starts=counts.copy();pose=self.v.get_link_transforms().numpy()[0]
  for k,s in enumerate(META['shapes']):
   body=s['body'];b=BODIES.index(body);T=np.array(s['shape_to_link']);local=T[:3,:3]@np.array([.13,0,0])+T[:3,3];p[k]=pose[b,:3]+rotation(pose[b,3:])@local;f[k]=12.;n[k,2]=1;counts[b]=1;starts[b]=k
  return[A(x)for x in [f,p,n,d,counts,starts]]
def save(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
class Tests(unittest.TestCase):
 def setUp(self):self.geo=Geometry(META,CLOUDS,BODIES)
 def test_geometry_points_exact_mask_and_shape_rotation(self):
  s=META['shapes'][0];T=np.array(s['shape_to_link']);pose=np.r_[[.2,.3,.4],[0,0,np.sin(.3),np.cos(.3)]];R=rotation(pose[3:])
  for x,want in [(.13,True),(.115,True),(.114999,False),(.132,False)]:
   source=np.array([x,0,0]);world=pose[:3]+R@(T[:3,:3]@source+T[:3,3]);ok,p=self.geo.cap(s['body'],world,pose);self.assertEqual(ok,want);np.testing.assert_allclose(p,source,atol=1e-15)
 def test_servo_exact_parent_each_named_row(self):
  spec=importlib.util.spec_from_file_location('parent_servo','tmp/updated_native_phase_b_design_002/provisional_actuator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  q=np.linspace(-.04,.04,18,dtype=np.float32);dq=np.linspace(-52,52,18,dtype=np.float32);target=np.zeros((2,18),np.float32);kp=np.full(18,12,np.float32);kd=np.full(18,.2,np.float32)
  raw,applied,ceiling=servo(np.tile(q,(2,1)),np.tile(dq,(2,1)),target,kp,kd);old=m.effort(q,dq,target[0],kp,kd)
  np.testing.assert_allclose(raw[0],old['requested_nm'],atol=1e-6,rtol=0);np.testing.assert_allclose(applied[0],old['software_applied_nm'],atol=1e-7,rtol=0)
  self.assertLessEqual(abs(applied).max(),1.60001);self.assertEqual(applied[0,0],0);self.assertEqual(applied[0,-1],0)
 def test_patch_before_aggregate_and_invalid_ranges(self):
  v=View();c=Contact(v);pose=v.get_link_transforms().numpy();data=[a.numpy()for a in c.get_contact_data(.0025)];sm=[(0,b)for b in BODIES]
  result=classify_contacts(data,sm,pose,self.geo,1);self.assertTrue(result['distal_contact'].all());self.assertFalse(result['nonfoot_contact'].any())
  k=0;s=META['shapes'][k];b=BODIES.index(s['body']);T=np.array(s['shape_to_link']);data[1][k]=pose[0,b,:3]+rotation(pose[0,b,3:])@(T[:3,:3]@np.array([.09,0,0])+T[:3,3]);r=classify_contacts(data,sm,pose,self.geo,1);self.assertTrue(r['nonfoot_contact'][0]);self.assertFalse(r['distal_contact'].all())
  data[4][0,0]=1025
  with self.assertRaises(ValueError):classify_contacts(data,sm,pose,self.geo,1)
 def test_initial_clearance_exact_neutral(self):
  v=View();minimum,non=self.geo.clearance(v.get_link_transforms().numpy());self.assertAlmostEqual(minimum[0],.005,places=6);self.assertGreater(non[0],.001)
 def test_session_actual_methods_eight_steps_copied_packet_no_retry_reset(self):
  old=sys.modules.get('warp');sys.modules['warp']=types.SimpleNamespace(array=lambda x,**kw:A(x),uint32=np.uint32,float32=np.float32)
  try:
   with tempfile.TemporaryDirectory()as d:
    v=View();sim=Sim(v);native={'joint_names':NAMES,'body_names':BODIES,'native_max_velocity':np.full((1,18),50.26548).tolist(),'limits':[[[j['lower'],j['upper']]for j in MODEL['joints']]]};session=NativeStandingSession(v,Contact(v),sim,MODEL,native,self.geo,d,save,'cpu')
    first=session.reset_canonical();self.assertFalse(first['contact_valid'].any());first['joint_position_rad'][:]=1
    self.assertTrue((session.observe()['joint_position_rad']==0).all());self.assertEqual(sim.counter,31)
    with self.assertRaises(ValueError):session.reset_canonical()
    packet=session.step_control(np.zeros((1,18),np.float32));self.assertEqual(sim.counter,39);self.assertEqual(session.count,8);self.assertTrue(packet['contact_valid'].all());self.assertTrue(packet['distal_contact'].all());self.assertTrue((packet['computed_torque_nm']==0).all());raw=session.observe_control_substeps();self.assertEqual(len(raw),8);raw[0]['joint_position_rad'][:]=99;self.assertTrue((session.observe_control_substeps()[0]['joint_position_rad']==0).all())
    session.failure='test terminal';before=sim.counter
    with self.assertRaises(RuntimeError):session.step_control(np.zeros((1,18),np.float32))
    self.assertEqual(sim.counter,before);session.close();z=np.load(Path(d)/'substeps_000.npz');self.assertEqual(z['sequence'].tolist(),list(range(8)));self.assertEqual(z['explicit_counter'].tolist(),list(range(32,40)))
  finally:
   if old is None:sys.modules.pop('warp',None)
   else:sys.modules['warp']=old
 def test_clipped_non_toe_extrema_include_original_crossing_edges(self):
  for shape in META['shapes']:
   self.assertEqual(shape['non_toe_plane_intersections'],481)
   T=np.array(shape['shape_to_link']);cloud=CLOUDS['non_toe__'+shape['body']]
   local=(cloud-T[:3,3])@T[:3,:3]
   # Other attached meshes may exist; the tibia clip itself must reach the cut.
   self.assertTrue(np.any(abs(local[:,0]-.115)<1e-10))
 def test_native_getter_fault_retains_actual_clock_and_partial_position(self):
  old=sys.modules.get('warp');sys.modules['warp']=types.SimpleNamespace(array=lambda x,**kw:A(x),uint32=np.uint32,float32=np.float32)
  try:
   with tempfile.TemporaryDirectory()as d:
    v=View();sim=Sim(v);native={'joint_names':NAMES,'body_names':BODIES,'native_max_velocity':np.full((1,18),50.26548).tolist(),'limits':[[[j['lower'],j['upper']]for j in MODEL['joints']]]}
    session=NativeStandingSession(v,Contact(v),sim,MODEL,native,self.geo,d,save,'cpu');session.reset_canonical()
    original=v.get_root_velocities
    def fail_after_step():
     if sim.counter>31:raise RuntimeError('injected late getter')
     return original()
    v.get_root_velocities=fail_after_step
    with self.assertRaises(RuntimeError):session.step_control(np.zeros((1,18),np.float32))
    session.close();partial=json.loads((Path(d)/'failed_partial_step.json').read_text());summary=json.loads((Path(d)/'session.json').read_text())
    self.assertEqual(partial['actual_explicit_counter'],32);self.assertEqual(partial['active_getter'],'get_root_velocities');self.assertIn('joint_position_rad',partial);self.assertEqual(summary['steps'],1);self.assertEqual(summary['captured_steps'],0);self.assertFalse(summary['all_rows_recorded'])
  finally:
   if old is None:sys.modules.pop('warp',None)
   else:sys.modules['warp']=old
 def test_second_substep_pre_getter_failure_is_fresh_without_extra_step(self):
  old=sys.modules.get('warp');sys.modules['warp']=types.SimpleNamespace(array=lambda x,**kw:A(x),uint32=np.uint32,float32=np.float32)
  try:
   with tempfile.TemporaryDirectory()as d:
    v=View();sim=Sim(v);native={'joint_names':NAMES,'body_names':BODIES,'native_max_velocity':np.full((1,18),50.26548).tolist(),'limits':[[[j['lower'],j['upper']]for j in MODEL['joints']]]}
    session=NativeStandingSession(v,Contact(v),sim,MODEL,native,self.geo,d,save,'cpu');session.reset_canonical();original=v.get_dof_velocities;calls=[0]
    def fail_pre_second():
     calls[0]+=1
     if calls[0]==3:raise RuntimeError('pre second step')
     return original()
    v.get_dof_velocities=fail_pre_second
    with self.assertRaises(RuntimeError):session.step_control(np.zeros((1,18),np.float32))
    session.close();partial=json.loads((Path(d)/'failed_partial_step.json').read_text());summary=json.loads((Path(d)/'session.json').read_text())
    self.assertEqual(partial['sequence'],1);self.assertEqual(partial['substep_index'],1);self.assertEqual(partial['active_getter'],'get_dof_velocities');self.assertEqual(sim.counter,32);self.assertEqual(summary['captured_steps'],1);self.assertTrue(summary['all_rows_recorded'])
  finally:
   if old is None:sys.modules.pop('warp',None)
   else:sys.modules['warp']=old
 def test_quiet_scorer_AST_preserved(self):
  parent=ast.parse(Path('tmp/reference_physics_adapter_009/source_009/tools/omni_quiet_review.py').read_text());new=ast.parse((HERE/'quiet_metrics.py').read_text())
  for name in ['quiet_metrics']:
   a=next(x for x in parent.body if isinstance(x,ast.FunctionDef)and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef)and x.name==name);self.assertEqual(ast.dump(a),ast.dump(b))
if __name__=='__main__':unittest.main()
