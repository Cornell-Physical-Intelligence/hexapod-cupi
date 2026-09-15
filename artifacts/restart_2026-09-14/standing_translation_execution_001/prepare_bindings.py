"""Root-reviewed exact host/guard bindings for two fresh diagnostic arms."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREP=BASE/'standing_translation_preparation_001'
GUARD=BASE/'standing_translation_guard_001'
REMOTE='/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
 m=root/'FREEZE_SHA256.json'
 expected=json.loads(m.read_text())
 actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=m}
 assert expected==actual, str(root)
 assert not any(p.is_symlink() for p in root.rglob('*'))
 return sha(m)
source=verify(PREP/'source');host=verify(PREP/'host');guard=verify(GUARD)
module=ast.parse((PREP/'host/launch_standing_spark.py').read_text())
source_pin=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FREEZE' for t in n.targets))
assert source_pin==source
ops=json.loads((BASE/'spark_ownership_001/RESERVATION.json').read_text())
directory=HERE/'bindings';directory.mkdir(exist_ok=False)
record={'source_freeze_sha256':source,'host_freeze_sha256':host,'guard_freeze_sha256':guard,'remote_root':REMOTE,'arms':{}}
for arm in ('origin','xy14_4'):
 output=REMOTE+'/'+arm+'_001'
 native={'schema':'canonical_placement_host_binding_v1','source_freeze_sha256':source,'host_freeze_sha256':host,'placement':arm,'output':output,'native_dispatch_authorized':True,'coordination_sha256':ops['coordination_sha256']}
 nativepath=directory/(arm+'_host.json');nativepath.write_text(json.dumps(native,indent=2)+'\n')
 binding={'schema':'canonical_single_placement_guard_v1','root_review_complete':True,'placement':arm,'source':REMOTE+'/source','host':REMOTE+'/host','native_bindings':REMOTE+'/bindings/'+nativepath.name,'output':output,'asset':'/home/orionh/HEXAPOD_runs/canonical_direct_20260910/asset_001','admission':'/home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_actuation_001/actuation','supervisor_source':'/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009','isaaclab':'/home/orionh/IsaacLab','num_envs':1,'diagnostic_only':True,'max_seconds':1200,'source_freeze_sha256':source,'host_freeze_sha256':host,'native_bindings_sha256':sha(nativepath),'reservation_root':'/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001','policy_sha256':ops['policy_sha256'],'coordination_sha256':ops['coordination_sha256'],**{key:False for key in ('standing_admission','batch_admission','stage2_complete','training_allowed','physical_admission','physics_admitted')}}
 path=directory/(arm+'_guard.json');path.write_text(json.dumps(binding,indent=2)+'\n')
 record['arms'][arm]={'output':output,'native_bindings_sha256':sha(nativepath),'guard_bindings_sha256':sha(path),'unit':'hexapod-placement-'+arm.replace('_','-')+'-001-20260914.service'}
(HERE/'BINDINGS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
