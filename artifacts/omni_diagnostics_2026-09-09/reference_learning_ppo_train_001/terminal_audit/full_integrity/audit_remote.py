from pathlib import Path
import hashlib, importlib.util, json, sys, time
from types import SimpleNamespace
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
spec=importlib.util.spec_from_file_location('prior',Path(__file__).parent/'prior_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
sha=a.sha
maps={k:a.tree(BASE/folder,manifest,h,count) for k,folder,manifest,h,count in a.SPECS}
maps['continuation_guard']=a.tree(BASE/'reference_learning_ppo_continuation_guard_001','FREEZE_SHA256.json','8b03cadd1224b8615d536e2fce2b37b7a6e723410dada59f6292d0dfb50de8f3',7)
run=BASE/'reference_learning_ppo_001';root=run/'inputs/study';assets=a.read(run/'inputs/study_before.sha256.json')
assert len(assets)==550 and not any(p.is_symlink() for p in root.rglob('*'))
assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(assets)
for name,h in assets.items():a.relative(name);assert sha(root/name)==h,name
spec=importlib.util.spec_from_file_location('host',BASE/'reference_learning_ppo_host_001/launch_learning_ppo_spark.py');host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
args=SimpleNamespace(source=BASE/'reference_physics_source_009',run=BASE/'reference_physics_009',device_run=BASE/'reference_device_smoke_001',bridge=BASE/'reference_device_smoke_adapter_001',consumer=BASE/'reference_moving_ppo_source_003',observation=BASE/'reference_policy_observation_005_001',output=run,phase_group='learn',decision_receipt=BASE/'reference_learning_ppo_decision_001.json')
identity=host.validate_inputs(args,phase='train_10',require_standing=True)
assert sha(args.decision_receipt)=='bf01ba95e9d2160ea83564c857cc4972bd90a15e46ec47599f213b2b1e7662a0'
assert sha(run/'campaign.json')=='3175ead5c6e4718abce48ee2b7001187ba46dbff9900eb575f8d21b93de6ee23'
# Verify both historical admission snapshot and the new terminal raw set independently.
verified={}
for name,snapshot in [('initial',Path(__file__).parent/'initial_snapshot.json'),('learning',Path(__file__).parent/'learning_snapshot.json')]:
 if not snapshot.exists() and name=='initial':
  candidates=list(BASE.glob('*audit*json'))
  raise FileNotFoundError('Need exact old audit location; available '+str([p.name for p in candidates]))
 data=a.read(snapshot);n=0
 for f,h in data['raw_payloads'].items():
  if name=='learning':p=Path(data['remote_paths'][f])
  elif f.startswith('run/'):p=run/f.removeprefix('run/')
  elif f.startswith('forecast_pause/'):p=BASE/'forecast_pause_051'/f.removeprefix('forecast_pause/')
  elif f.startswith('preflight/'):p=BASE/f.removeprefix('preflight/')
  else:raise ValueError(f)
  assert sha(p)==h,str(p);n+=1
 verified[name]={'snapshot':str(snapshot),'snapshot_sha256':sha(snapshot),'files':n,'unchanged':True}
print(json.dumps({'checked_unix':time.time(),'audit_read_only':True,'input_maps':maps,'assets_files':550,'assets_unchanged':True,'host_readonly_identity':identity,'decision_sha256':sha(args.decision_receipt),'initial_campaign_sha256':sha(run/'campaign.json'),'raw_snapshot_checks':verified},indent=2))
