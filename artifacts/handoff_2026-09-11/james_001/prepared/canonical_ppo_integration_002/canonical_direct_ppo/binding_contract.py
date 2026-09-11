"""Stdlib admission/lineage barrier, evaluated before an actor or native allocation."""
from pathlib import Path
import hashlib,importlib.util,json
from .smoke_config import protocol


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def files(path):return {str(p.relative_to(path)):sha(p)for p in sorted(path.rglob('*'))if p.is_file()}
def require_hash(value,name):
 if not isinstance(value,str)or len(value)!=64 or any(c not in '0123456789abcdef'for c in value):raise ValueError('Pending/invalid required binding:'+name)
 return value

def verify_directory(directory,digest):
 directory=Path(directory).resolve();require_hash(digest,'source manifest')
 if sha(directory/'FREEZE_SHA256.json')!=digest:raise ValueError('Wrong frozen source')
 expected=read(directory/'FREEZE_SHA256.json')
 actual={str(p.relative_to(directory)):sha(p)for p in directory.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p!=directory/'FREEZE_SHA256.json'}
 if actual!=expected:raise ValueError('Changed frozen source inventory')
 if any((directory/name).is_symlink()or not(directory/name).resolve().is_relative_to(directory)for name in expected):raise ValueError('Unsafe source path')
 return expected


def verify_admissions(standing_source,standing_one,standing32,bindings):
 for name in ['standing_source_freeze_sha256','standing1_state_sha256','standing32_state_sha256','servo_sha256','geometry_sha256','policy_adapter_sha256']:
  require_hash(bindings.get(name),name)
 if bindings.get('ready_for_native_dispatch')is not True:raise ValueError('Preparation is not enabled for native dispatch')
 root=Path(standing_source).resolve();source=verify_directory(root,bindings['standing_source_freeze_sha256'])
 for file,key in [('servo_candidate.json','servo_sha256'),('geometry/geometry.json','geometry_sha256')]:
  if source.get(file)!=bindings[key]:raise ValueError('Wrong admitted servo/geometry')
 spec=importlib.util.spec_from_file_location('_canonical_smoke_standing_contract',root/'standing_contract.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 states=[]
 for n,directory,key in [(1,standing_one,'standing1_state_sha256'),(32,standing32,'standing32_state_sha256')]:
  directory=Path(directory).resolve()
  if sha(directory/'state.json')!=bindings[key]:raise ValueError('Changed exact standing receipt')
  state=read(directory/'state.json');result=module.validate_result(directory,state['identity'])
  identity=state['identity']
  if identity.get('num_envs')!=n or identity['runtime_binding']['runtime_tree_sha256']!=bindings['standing_source_freeze_sha256']or result.get('standing_pass')is not True:raise ValueError('Wrong or rejected native standing evidence')
  for field in ['urdf_sha256','model_sha256','usd_sha256']:
   if identity.get(field)!=protocol()[field]:raise ValueError('Historical/wrong model rejected:'+field)
  if identity.get('servo_sha256')!=bindings['servo_sha256']or identity.get('geometry_sha256')!=bindings['geometry_sha256']:raise ValueError('Standing result parameter identity mismatch')
  states.append(state)
 if states[1]['identity'].get('standing_one_state_sha256')!=bindings['standing1_state_sha256']:raise ValueError('Standing32 was not admitted by exact standing1')
 if states[0]['identity']['actuation_state_sha256']!=states[1]['identity']['actuation_state_sha256']:raise ValueError('Native motor admission mismatch')
 own=Path(__file__).resolve().parent
 if sha(own/'adapter.py')!=bindings['policy_adapter_sha256']:raise ValueError('Unreviewed policy adapter')
 return {'schema':protocol()['schema'],'protocol':protocol(),'standing_source_freeze_sha256':bindings['standing_source_freeze_sha256'],
         'standing1_state_sha256':bindings['standing1_state_sha256'],'standing32_state_sha256':bindings['standing32_state_sha256'],
         'servo_sha256':bindings['servo_sha256'],'geometry_sha256':bindings['geometry_sha256'],'adapter_sha256':bindings['policy_adapter_sha256'],
         'fresh_checkpoint_only':True,'physical_hardware_admitted':False,'Stage2_complete':False}


def bind_consumer_lineage(lineage):
 own=Path(__file__).resolve().parent.parent
 digest=sha(own/'FREEZE_SHA256.json');source=verify_directory(own,digest)
 rsl_sha=sha(own/'RSL_SOURCE_SHA256.json')
 if source.get('RSL_SOURCE_SHA256.json')!=rsl_sha:raise ValueError('Unbound installed-RSL source map')
 return {**lineage,'consumer_source_freeze_sha256':digest,'rsl_source_map_sha256':rsl_sha}
