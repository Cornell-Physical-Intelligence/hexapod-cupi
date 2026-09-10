from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys,time
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
source=base/'reference_pair_motion_source_001';guard=base/'reference_pair_motion_launch_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,manifest,bound):
 assert sha(root/manifest)==bound
 assert not any(p.is_symlink() for p in root.rglob('*'))
 assert {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/manifest}==json.loads((root/manifest).read_text())
verify(source,'campaign_source_hashes.json','216224f3cb727549d8f70bfc2836f32bdf472f892b37b0c89fa471dc71b23932')
verify(guard,'FREEZE_SHA256.json','2bb6fbbf9b404c05bfff625a715b19c43f60eead679e57ff8786a024188c1d80')
assert sha(source/'tools/launch_pair_motion_spark.py')=='60d51f1e8d71924390add188460f12e157eb114aa3dae617ce254c90fe4d021e'
sys.path.insert(0,str(source/'tools'))
import launch_pair_motion_spark as host
runtime=host.check_source(source)
out=base/'reference_pair_motion_001';pause=base/'forecast_pause_050';assert not out.exists() and not pause.exists()
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md');assert sha(coord)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
unit=subprocess.check_output(['systemctl','--user','show','hexapod-moving-ppo-001-20260910.service','-p','ActiveState','-p','Result','-p','InvocationID'],text=True,timeout=20)
state=dict(line.split('=',1) for line in unit.splitlines());assert state['ActiveState'] in ('failed','inactive') and state['Result']=='exit-code'
assert state.get('InvocationID') in ('','53fa4c8afc7940e0931a77c18d310557')
tree=ast.parse((guard/'launch_guarded_remote.py').read_text());pins=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prior_pins' for t in n.targets))
for name,bound in pins.items():assert sha(base/name)==bound,name
absence=[]
for phase in ('standing','calibration','profile_32'):
 j=json.loads((base/'reference_moving_ppo_001/jobs'/(phase+'.json')).read_text());assert j['cleanup_checked'] and j['container_id']
 for identifier in (j['container_name'],j['container_id']):
  p=subprocess.run(['docker','inspect',identifier],text=True,capture_output=True,timeout=20)
  assert p.returncode and any(x in p.stderr.lower() for x in ('no such object','no such container')),identifier
  absence.append({'identifier':identifier,'absent':True})
report={'passed':True,'checked_unix':time.time(),'source_files':946,'source_manifest_sha256':sha(source/'campaign_source_hashes.json'),'host_sha256':sha(source/'tools/launch_pair_motion_spark.py'),'guard_sha256':sha(guard/'launch_guarded_remote.py'),'guard_freeze_sha256':sha(guard/'FREEZE_SHA256.json'),'runtime':runtime,'coordination_sha256':sha(coord),'previous_unit':state,'previous_receipts':pins,'previous_owned_absent':absence,'fresh_output':str(out),'fresh_pause':str(pause),'GPU_allocated':False}
p=base/'reference_pair_motion_preflight_001.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'source_files':946,'previous_identifiers_absent':len(absence),'preflight':str(p)}))
