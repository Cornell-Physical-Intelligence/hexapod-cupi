from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,subprocess,sys,time
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
hroot=base/'reference_learning_ppo_host_001';groot=base/'reference_learning_ppo_guard_001'
h=load('bound_host',hroot/'launch_learning_ppo_spark.py');g=load('bound_guard',groot/'launch_guarded_remote.py')
h.verify_tree(hroot,'FREEZE_SHA256.json','45f623d298c00136ca18d159261d8df601bd42840ebae467fa94bf38283e6792')
h.verify_tree(groot,'FREEZE_SHA256.json','fc623f16e5a559da3940852892bc3eeef39b6143ddf3587873dcc07480cafc78')
args=SimpleNamespace(source=g.SOURCE,run=g.RUN,device_run=g.DEVICE_RUN,bridge=g.BRIDGE,consumer=g.CONSUMER,observation=g.OBSERVATION,output=g.OUTPUT,phase_group='admission',decision_receipt=None)
assert not g.OUTPUT.exists() and not g.PAUSE.exists()
identity=h.validate_inputs(args)
sys.path.insert(0,str(g.SOURCE/'tools'))
from launch_reference_physics_spark import check_source
runtime=check_source(g.SOURCE)
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
assert sha(coord)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
unit=subprocess.check_output(['systemctl','--user','show','hexapod-pair-motion-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','InvocationID'],text=True,timeout=20)
state=dict(line.split('=',1) for line in unit.splitlines())
assert state['ActiveState'] in ('failed','inactive') and state['Result']=='exit-code'
assert state.get('InvocationID') in ('','aeb33fdfce794cfca50e4227ebfada1d')
tree=ast.parse((groot/'launch_guarded_remote.py').read_text())
pins=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prior_pins' for t in n.targets))
for name,bound in pins.items():assert sha(base/name)==bound,name
absence=[]
for phase in ('standing','paired_forward'):
 j=json.loads((base/'reference_pair_motion_001/jobs'/(phase+'.json')).read_text())
 assert j['cleanup_checked'] and j['container_id']
 for identifier in (j['container_name'],j['container_id']):
  result=subprocess.run(['docker','inspect',identifier],text=True,capture_output=True,timeout=20)
  assert result.returncode and any(x in result.stderr.lower() for x in ('no such object','no such container')),identifier
  absence.append({'identifier':identifier,'absent':True})
report={'passed':True,'checked_unix':time.time(),'identity':identity,'runtime':runtime,'host_sha256':sha(g.HOST),'host_freeze_sha256':sha(hroot/'FREEZE_SHA256.json'),'guard_sha256':sha(groot/'launch_guarded_remote.py'),'guard_freeze_sha256':sha(groot/'FREEZE_SHA256.json'),'coordination_sha256':sha(coord),'previous_unit':state,'previous_receipts':pins,'previous_owned_absent':absence,'fresh_output':str(g.OUTPUT),'fresh_pause':str(g.PAUSE),'phases':h.ADMISSION_PHASES,'GPU_allocated':False}
p=base/'reference_learning_ppo_preflight_001.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'preflight':str(p),'consumer':identity['consumer_freeze_sha256'],'phases':h.ADMISSION_PHASES}))
