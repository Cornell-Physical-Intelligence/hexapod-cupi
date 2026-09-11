"""Synthetic native-view semantics: observation parity and retained failed readbacks."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile,types,unittest
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import test_standing as fixture
from standing_session import NativeStandingSession
from solver_diagnostics import collect_legacy_friction,validate_legacy,LINK_FIELD,FLOOR_FIELD
from native_support import save

HERE=Path(__file__).resolve().parent
PARENT_HASHES=json.loads((HERE/'PARENT_STANDING004_FREEZE_SHA256.json').read_text())
spec=importlib.util.spec_from_file_location('standing004_session_diagnostic_test',HERE/'PARENT_STANDING004_SESSION.py')
parent_session=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent_session)

class Shared:
 def __init__(self,value):self.value=value
 def numpy(self):return self.value

@contextmanager
def fake_warp():
 original=sys.modules.get('warp');sys.modules['warp']=types.SimpleNamespace(array=lambda x,**kw:fixture.A(x),uint32=np.uint32,float32=np.float32)
 try:yield
 finally:
  if original is None:sys.modules.pop('warp',None)
  else:sys.modules['warp']=original

@contextmanager
def session(kind=NativeStandingSession):
 with tempfile.TemporaryDirectory()as directory,fake_warp():
  view=fixture.View();sim=fixture.Sim(view);contact=fixture.Contact(view)
  native={'joint_names':fixture.NAMES,'body_names':fixture.BODIES,
   'native_max_velocity':np.full((1,18),50.26548).tolist(),
   'limits':[[[joint['lower'],joint['upper']]for joint in fixture.MODEL['joints']]]}
  geometry=fixture.Geometry(fixture.META,fixture.CLOUDS,fixture.BODIES)
  obj=kind(view,contact,sim,fixture.MODEL,native,geometry,directory,save,'cpu')
  obj.reset_canonical()
  try:yield obj,view,contact,sim,Path(directory)
  finally:
   if not obj.contact_stream.closed:obj.close()

class DiagnosticsTests(unittest.TestCase):
 def test_nonzero_legacy_is_preserved_copied_and_never_set(self):
  values=np.full((2,18),.05,np.float32)
  view=types.SimpleNamespace(count=2,get_dof_friction_coefficients=lambda:Shared(values))
  with tempfile.TemporaryDirectory()as directory:
   result=collect_legacy_friction(view,{'joint_names':fixture.NAMES},directory,save)
   values[:]=99
   self.assertTrue(validate_legacy(result,2,fixture.NAMES))
   self.assertEqual(result['coefficients'],np.full((2,18),.05,np.float32).tolist())
   self.assertEqual(json.loads((Path(directory)/'legacy_friction_readback.json').read_text()),result)
  tree=ast.parse((HERE/'solver_diagnostics.py').read_text())
  self.assertFalse(any(isinstance(node,ast.Attribute)and node.attr.startswith('set_')for node in ast.walk(tree)))

 def test_legacy_getter_failure_and_invalid_values_survive_before_rejection(self):
  for values in [np.zeros((1,17),np.float32),np.full((1,18),np.nan,np.float32)]:
   with tempfile.TemporaryDirectory()as directory:
    view=types.SimpleNamespace(count=1,get_dof_friction_coefficients=lambda:Shared(values))
    with self.assertRaises(ValueError):collect_legacy_friction(view,{'joint_names':fixture.NAMES},directory,save)
    record=json.loads((Path(directory)/'legacy_friction_readback.json').read_text())
    self.assertIn('coefficients',record);self.assertEqual(record['failed_getter']['name'],'get_dof_friction_coefficients')
  def missing():raise AttributeError('synthetic backend unavailable')
  with tempfile.TemporaryDirectory()as directory:
   view=types.SimpleNamespace(count=1,get_dof_friction_coefficients=missing)
   with self.assertRaises(AttributeError):collect_legacy_friction(view,{'joint_names':fixture.NAMES},directory,save)
   self.assertIn('failed_getter',json.loads((Path(directory)/'legacy_friction_readback.json').read_text()))

 def test_diagnostic_buffers_copied_before_another_read_and_endpoint_mutation(self):
  with session()as(obj,view,contact,sim,directory):
   link=np.arange(114,dtype=np.float32).reshape(1,19,6);view.get_link_velocities=lambda:Shared(link)
   snapshot=obj.snap();link[:]=999;self.assertEqual(snapshot[LINK_FIELD][0,0,0],0.)
   matrix=np.zeros((19,1,3),np.float32);data=contact.get_contact_data
   def read_matrix(dt):self.assertEqual(dt,.0025);matrix[:]=7.;return Shared(matrix)
   def read_data(dt):matrix[:]=999.;return data(dt)
   contact.get_contact_force_matrix=read_matrix;contact.get_contact_data=read_data
   result=obj.step_control(np.zeros((1,18),np.float32))
   self.assertTrue((result[FLOOR_FIELD]==7).all())
   self.assertTrue(result['distal_contact'].all());self.assertFalse(result['nonfoot_contact'].any())
   result[FLOOR_FIELD][:]=123;self.assertTrue((obj.observe_control_substeps()[-1][FLOOR_FIELD]==7).all())
   self.assertEqual(sim.counter,39)

 def test_new_link_getter_failure_preserves_actual_poststep_counter_and_prior_fields(self):
  with session()as(obj,view,contact,sim,directory):
   def fail():raise RuntimeError('synthetic link read failure')
   view.get_link_velocities=fail
   with self.assertRaises(RuntimeError):obj.step_control(np.zeros((1,18),np.float32))
   row=json.loads((directory/'failed_partial_step.json').read_text())
   self.assertEqual(row['actual_explicit_counter'],32);self.assertEqual(row['actual_controlled_steps'],1)
   self.assertEqual(row['active_getter'],'get_link_velocities');self.assertIn('joint_velocity_rad_s',row);self.assertIn('root_com_velocity',row)
   self.assertEqual(obj.captured_count,0)

 def test_invalid_floor_shape_retains_matrix_and_prior_link_diagnostic(self):
  with session()as(obj,view,contact,sim,directory):
   contact.get_contact_force_matrix=lambda dt:Shared(np.ones((19,3),np.float32))
   with self.assertRaisesRegex(ValueError,'Diagnostic shape'):obj.step_control(np.zeros((1,18),np.float32))
   row=json.loads((directory/'failed_partial_step.json').read_text())
   self.assertEqual(row['actual_explicit_counter'],32);self.assertEqual(row['active_getter'],'get_contact_force_matrix')
   self.assertEqual(np.asarray(row[FLOOR_FIELD]).shape,(19,3));self.assertIn(LINK_FIELD,row)
   self.assertEqual(obj.captured_count,0)

 def test_instrumentation_preserves_all_parent_common_rows_effort_and_rng(self):
  outcomes=[];random_before=np.random.get_state()
  for kind in [parent_session.NativeStandingSession,NativeStandingSession]:
   with session(kind)as(obj,view,contact,sim,directory):
    # Nonzero, bounded synthetic motion exercises PD; it is not native physics.
    def advance(render=False):
     sim.counter+=1;view.q[:]=np.float32((sim.counter-31)*1e-7);view.dq[:]=np.float32(4e-5)
    sim.step=advance
    for control in range(2):obj.step_control(np.zeros((1,18),np.float32))
    outcomes.append(([{k:np.asarray(v).copy()for k,v in row.items()}for row in obj.rows],view.f.copy(),sim.counter))
  before,after=outcomes
  self.assertEqual(before[2],after[2]);np.testing.assert_array_equal(before[1],after[1])
  for old,new in zip(before[0],after[0]):
   self.assertEqual(set(new)-set(old),{LINK_FIELD,FLOOR_FIELD})
   for key in old:np.testing.assert_array_equal(old[key],new[key],err_msg=key)
  random_after=np.random.get_state()
  self.assertEqual(random_before[0],random_after[0]);np.testing.assert_array_equal(random_before[1],random_after[1]);self.assertEqual(random_before[2:],random_after[2:])

 def test_original_scoring_servo_geometry_and_reset_methods_are_exact(self):
  self.assertEqual(hashlib.sha256((HERE/'PARENT_STANDING004_FREEZE_SHA256.json').read_bytes()).hexdigest(),'037013353a30c5f3751235634aafdb3f204b676837962b14df72c3051c0a2786')
  self.assertEqual(hashlib.sha256((HERE/'PARENT_STANDING004_SESSION.py').read_bytes()).hexdigest(),PARENT_HASHES['standing_session.py'])
  for name in ['standing_math.py','standing_score.py','quiet_metrics.py','servo_candidate.json','geometry/geometry.json','geometry/geometry_extrema.npz','ASSET_SHA256.json','native_support.py']:
   self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),PARENT_HASHES[name],name)
  old=ast.parse((HERE/'PARENT_STANDING004_SESSION.py').read_text());new=ast.parse((HERE/'standing_session.py').read_text())
  for name in ['reset_canonical','set_force','observe_control_substeps','flush']:
   get=lambda tree:next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name==name)
   self.assertEqual(ast.dump(get(old)),ast.dump(get(new)),name)

if __name__=='__main__':unittest.main()
