from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys

B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
host_freeze,guard_freeze=sys.argv[1:]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(folder,pin):
    p=B/folder; f=p/'FREEZE_SHA256.json';assert sha(f)==pin
    assert not any(x.is_symlink() for x in p.rglob('*'))
    m=json.loads(f.read_text());assert {x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and x!=f}==m
    return len(m)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
counts={'host':verify('direct_omni_train_host_005',host_freeze),'guard':verify('direct_omni_train_extended_caps_guard_002',guard_freeze)}
g=load('retry_guard',B/'direct_omni_train_extended_caps_guard_002/launch_guarded_remote.py')
g.require_final_bindings(); assert not g.OUTPUT.exists() and not g.PAUSE.exists()
g.verify_previous_owner()
g.verify_failed_previous_owner()
h=load('actual_retry_host',g.HOST); identity=g.validate_train_inputs(h)
a=SimpleNamespace(source=g.SOURCE,checkpoint=g.CHECKPOINT,contract=g.CONTRACT,supervisor_source=g.SUPERVISOR_SOURCE,output=g.OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=host_freeze,allocation='extended',branch='caps',smoke=B/'direct_omni_train_smoke_004')
supervisor=h.load_supervisor(a)
meta=supervisor.DIRECT_DEADLINE_ADAPTER
assert meta['phase_deadline_seconds']['train']==1800 and meta['app_ready_deadline_seconds']==90 and meta['cleanup_AST_unchanged']
assert sha(Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md'))=='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
print(json.dumps({'passed':True,'python':sys.version,'payload_counts':counts,'host_freeze':host_freeze,'guard_freeze':guard_freeze,'identity':identity,'actual_installed_host_load_supervisor_passed':True,'adapter':meta,'no_run_owned_called':True,'no_GPU':True,'output_created':g.OUTPUT.exists()},indent=2))
