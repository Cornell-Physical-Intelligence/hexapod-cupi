import unittest,json,tempfile
from pathlib import Path
from canonical_direct_ppo.binding_contract import verify_admissions,require_hash,sha
from canonical_direct_ppo.smoke_config import protocol

from test_support import ROOT,STANDING
from test_solver_fixture import write_native_records,write_member_fixture
from canonical_direct_ppo import source005_solver_diagnostics as diag

def write(path,data):path.write_text(json.dumps(data))

def synthetic_result(directory,n,one_sha=None,model_hash=None):
 """Protocol fixture only. No native arrays/measurement or admission claim escapes tempdir."""
 directory.mkdir();src=json.loads((STANDING/'FREEZE_SHA256.json').read_text());p=protocol()
 identity={k:p[k]for k in ['urdf_sha256','model_sha256','usd_sha256']}
 identity.update(num_envs=n,runtime_binding={'runtime_tree_sha256':sha(STANDING/'FREEZE_SHA256.json')},
  servo_sha256=src['servo_candidate.json'],geometry_sha256=src['geometry/geometry.json'],actuation_state_sha256='a'*64,CPU_SYNTHETIC_FIXTURE_ONLY=True)
 if one_sha is not None:identity['standing_one_state_sha256']=one_sha
 if model_hash is not None:identity['model_sha256']=model_hash
 quiet={k:0 for k in ['max_planar_excursion_m','max_heading_excursion_deg','max_joint_velocity_rms_rad_s','max_joint_position_range_rad','max_target_step_abs_p95_rad_per_20ms','max_requested_torque_saturation_fraction','max_applied_torque_nm','terminations','truncations']}
 quiet.update(pass_=True,failed_bounds=[],window_duration_s=16.);quiet['pass']=quiet.pop('pass_')
 physical={k:0 for k in ['post_settle_missing_six_toe_substeps','all_controlled_nonfoot_substeps','max_applied_all_substeps_nm','max_requested_saturation_fraction_400hz','mean_requested_saturation_fraction_400hz','minimum_non_toe_mesh_floor_m']};physical['minimum_plate_height_m']=.1
 report={'num_envs':n,'substeps':8000,'controls':1000,'all_pass':True,'replicas':[{'quiet':quiet,'physical':physical,'failed_physical_bounds':[],'pass':True}for _ in range(n)]}
 write(directory/'standing_report.json',report)
 write(directory/'session.json',{'captured_steps':8000,'all_rows_recorded':True,'steps':8000,'controls':1000,'reset_count':1,'failure':None,'substep_files':['substeps_000.npz'],'solver_diagnostics':diag.declaration(n,8000)})
 for name in ['control_trace.npz','initial_reset.json','native_readback.json','sdf_readback.json','contact_view.json','contacts.jsonl','substeps_000.npz']:(directory/name).write_text('CPU protocol fixture only')
 write_native_records(directory,n,final=True)
 for name in ['control_trace.npz','substeps_000.npz']:write_member_fixture(directory/name)
 write(directory/'native_errors.json',[])
 outputs={p.name:sha(p)for p in directory.iterdir()if p.name!='native_errors.json'}
 state={'schema':'canonical_native_standing_v1','identity':identity,'status':'completed','inputs_unchanged':True,'errors':[],'native_error_events':[],
  'physical_admission':False,'physics_admitted':False,'training_allowed':False,'explicit_steps_completed':8000,'checks':{'CPU_PROTOCOL_FIXTURE_ONLY':True,'solver_recipe_readback':True,'legacy_friction_observed':True},'outputs':outputs}
 write(directory/'state.json',state);return sha(directory/'state.json')

class BindingChecks(unittest.TestCase):
 def test_pending_bindings_fail_before_reading_native_paths(self):
  with self.assertRaisesRegex(ValueError,'Pending'):verify_admissions('/not-opened','/not-opened','/not-opened',{})
 def test_invented_or_placeholder_hash_is_not_admission(self):
  for value in [None,'PENDING','x'*64,'0'*63,True]:
   with self.assertRaises(ValueError):require_hash(value,'test')
 def test_exact_source_contract_positive_and_wrong_model_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);one=root/'one';many=root/'many';s1=synthetic_result(one,1);s32=synthetic_result(many,32,s1)
   src=json.loads((STANDING/'FREEZE_SHA256.json').read_text())
   b={'standing_source_freeze_sha256':sha(STANDING/'FREEZE_SHA256.json'),'standing1_state_sha256':s1,'standing32_state_sha256':s32,
    'servo_sha256':src['servo_candidate.json'],'geometry_sha256':src['geometry/geometry.json'],
    'policy_adapter_sha256':sha(Path(__file__).parent/'canonical_direct_ppo/adapter.py'),'ready_for_native_dispatch':True}
   lineage=verify_admissions(STANDING,one,many,b);self.assertFalse(lineage['Stage2_complete'])
   wrong=root/'wrong';b['standing32_state_sha256']=synthetic_result(wrong,32,s1,model_hash='b'*64)
   with self.assertRaisesRegex(ValueError,'wrong model'):verify_admissions(STANDING,one,wrong,b)
 def test_changed_raw_and_different_first_admission_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);one=root/'one';many=root/'many';s1=synthetic_result(one,1);s32=synthetic_result(many,32,'b'*64)
   src=json.loads((STANDING/'FREEZE_SHA256.json').read_text())
   b={'standing_source_freeze_sha256':sha(STANDING/'FREEZE_SHA256.json'),'standing1_state_sha256':s1,'standing32_state_sha256':s32,
    'servo_sha256':src['servo_candidate.json'],'geometry_sha256':src['geometry/geometry.json'],
    'policy_adapter_sha256':sha(Path(__file__).parent/'canonical_direct_ppo/adapter.py'),'ready_for_native_dispatch':True}
   with self.assertRaisesRegex(ValueError,'exact standing1'):verify_admissions(STANDING,one,many,b)
   (many/'contacts.jsonl').write_text('CHANGED')
   with self.assertRaisesRegex(ValueError,'Changed input'):verify_admissions(STANDING,one,many,b)

if __name__=='__main__':unittest.main()
