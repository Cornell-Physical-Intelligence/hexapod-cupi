"""Stdlib exact input, scope and terminal contract for provisional native standing."""
from pathlib import Path
import importlib.util,json,hashlib,math

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
_parent=load('_standing_asset','inspection_contract.py');_b=load('_standing_effort','actuation_contract.py')
sha=_parent.sha;read=_parent.read;check_map=_parent.check_map
SCHEMA='canonical_native_standing_v1';DT=.0025;CONTROLS=1000;STEPS=8000;SETTLE=200
ACTUATION_FREEZE='1f02009cb22c1efbe4a0d0c7f7988d9d63b658d78e509ea444b7afafee6178c9'

def inventory(p):return {str(f.relative_to(p)):sha(f)for f in sorted(p.rglob('*'))if f.is_file()}
def mapsha(m):return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_inputs(args):
 own=Path(__file__).resolve().parent;identity=_parent.verify_inputs(args);n=args.num_envs
 if type(n)is not int or n not in (1,32):raise ValueError('Only explicit1 then32 standing')
 admission=Path(args.admission).resolve();expected=read(own/'ACTUATION_SHA256.json');check_map(admission,expected)
 if inventory(admission)!=expected:raise ValueError('Exact actual actuation inventory changed')
 s=read(admission/'state.json');_b.validate_result(admission,s['identity'])
 if s['identity']['inspector_freeze_sha256']!=ACTUATION_FREEZE:raise ValueError('Unreviewed native effort source')
 if Path(args.output).resolve().is_relative_to(admission):raise ValueError('Output overlaps effort evidence')
 identity.update(schema=SCHEMA,num_envs=n,controls=CONTROLS,steps=STEPS,settle_controls=SETTLE,dt=DT,control_dt=.02,substeps_per_control=8,ground=True,gravity=[0.,0.,-9.81],
   runtime_binding={'runtime_tree_sha256':identity['inspector_freeze_sha256'],'scope':'canonical_provisional_native_standing_only'},
   actuation_state_sha256=sha(admission/'state.json'),actuation_inventory_sha256=mapsha(inventory(admission)),
   geometry_sha256=sha(own/'geometry/geometry.json'),servo_sha256=sha(own/'servo_candidate.json'),
   solver_recipe_sha256=sha(own/'solver_recipe.py'),solver_recipe={'position_iterations':32,'velocity_iterations':4},
   model_scope='Uncalibrated provisional48V software effort envelope; no battery confirmation or hardware qualification',
   physical_admission=False,physics_admitted=False,training_allowed=False)
 if n==32:
  if not args.standing_one:raise ValueError('Actual same-source standing1 required before32')
  d=Path(args.standing_one).resolve();old=read(d/'state.json');r=validate_result(d,old['identity'])
  if old['identity']['num_envs']!=1 or old['identity']['runtime_binding']!=identity['runtime_binding']or old['identity']['actuation_state_sha256']!=identity['actuation_state_sha256']:raise ValueError('Standing1 source/effort identity mismatch')
  if Path(args.output).resolve().is_relative_to(d):raise ValueError('Output overlaps standing1')
  identity['standing_one_state_sha256']=r['state_sha256'];identity['standing_one_inventory_sha256']=mapsha(inventory(d))
 elif getattr(args,'standing_one',None):raise ValueError('Standing1 does not consume a previous standing result')
 return identity

def validate_result(directory,identity):
 d=Path(directory);s=read(d/'state.json')
 if s.get('schema')!=SCHEMA or s.get('identity')!=identity:raise ValueError('Wrong native standing identity')
 if s.get('status')!='completed' or s.get('inputs_unchanged')is not True:raise ValueError('Incomplete standing acquisition')
 if s.get('errors')!=[]or s.get('native_error_events')!=[]or read(d/'native_errors.json')!=[]:raise ValueError('Runtime/late native error')
 if any(s.get(k)is not False for k in ['physical_admission','physics_admitted','training_allowed']):raise ValueError('Standing cannot claim broader physics or training admission')
 if s.get('explicit_steps_completed')!=STEPS:raise ValueError('Incomplete standing steps')
 if not s.get('checks')or not all(x is True for x in s['checks'].values()):raise ValueError('Missing/rejected native identities')
 if s['checks'].get('solver_recipe_readback')is not True:raise ValueError('Missing solver recipe readback')
 solver=read(d/'solver_readback.json')
 expected_roots={'/Robot'}|{f'/Robot_{i:03d}'for i in range(1,identity['num_envs'])}
 for phase in ('before','after_authoring','after_reset','after_controlled_steps'):
  values=solver.get(phase,{})
  if set(values)!=expected_roots:raise ValueError('Incomplete solver recipe roots')
  expected=(32,1,[False,False])if phase=='before'else(32,4,[True,True])
  if any((v.get('position_iterations'),v.get('velocity_iterations'),v.get('authored'))!=expected for v in values.values()):raise ValueError('Solver recipe differs:'+phase)
 outputs=s.get('outputs',{});check_map(d,outputs)
 if 'solver_readback.json'not in outputs:raise ValueError('Unsealed solver readback')
 if not {'control_trace.npz','session.json','standing_report.json','initial_reset.json','native_readback.json','sdf_readback.json','contact_view.json','contacts.jsonl'}.issubset(outputs):raise ValueError('Missing standing raw evidence')
 session=read(d/'session.json');report=read(d/'standing_report.json')
 if session.get('captured_steps')!=STEPS or session.get('all_rows_recorded')is not True:raise ValueError('Incomplete captured physics rows')
 if session.get('steps')!=STEPS or session.get('controls')!=CONTROLS or session.get('reset_count')!=1 or session.get('failure')is not None:raise ValueError('Clock/reset/failure state rejected')
 if any(x not in outputs for x in session['substep_files'])or not session['substep_files']:raise ValueError('Missing raw substeps')
 if report.get('num_envs')!=identity['num_envs']or report.get('substeps')!=STEPS or report.get('controls')!=CONTROLS or report.get('all_pass')is not True:raise ValueError('Standing physics/quiet rejected; acquisition is not admission')
 if len(report.get('replicas',[]))!=identity['num_envs']or any(r.get('pass')is not True for r in report['replicas']):raise ValueError('Incomplete per-replica pass')
 bounds={'max_planar_excursion_m':.01,'max_heading_excursion_deg':2.,'max_joint_velocity_rms_rad_s':.03,'max_joint_position_range_rad':.02,'max_target_step_abs_p95_rad_per_20ms':.002,'max_requested_torque_saturation_fraction':.005,'max_applied_torque_nm':1.60001}
 for replica in report['replicas']:
  q=replica.get('quiet',{});p=replica.get('physical',{})
  if q.get('pass')is not True or q.get('failed_bounds')!=[]or replica.get('failed_physical_bounds')!=[]or q.get('terminations')!=0 or q.get('truncations')!=0:raise ValueError('Rejected per-replica bounds')
  for key,bound in bounds.items():
   x=q.get(key)
   if type(x)not in(int,float)or not math.isfinite(x)or x>bound:raise ValueError('Quiet metric contradicts pass:'+key)
  if q.get('window_duration_s',0)<10:raise ValueError('Short quiet window')
  for key in ['post_settle_missing_six_toe_substeps','all_controlled_nonfoot_substeps']:
   if p.get(key)!=0:raise ValueError('Actual support/contact metric contradicts pass')
  for key,limit in [('max_applied_all_substeps_nm',1.60001),('max_requested_saturation_fraction_400hz',.005),('mean_requested_saturation_fraction_400hz',.005)]:
   v=p.get(key)
   if type(v)not in(int,float)or not math.isfinite(v)or v>limit:raise ValueError('Actual motor metric contradicts pass')
  for key,minimum in [('minimum_non_toe_mesh_floor_m',-.001),('minimum_plate_height_m',.055)]:
   v=p.get(key)
   if type(v)not in(int,float)or not math.isfinite(v)or v<minimum:raise ValueError('Actual clearance metric contradicts pass')
 return {'phase':'standing','status':'completed','num_envs':identity['num_envs'],'standing_pass':True,'training_allowed':False,'state_sha256':sha(d/'state.json')}
