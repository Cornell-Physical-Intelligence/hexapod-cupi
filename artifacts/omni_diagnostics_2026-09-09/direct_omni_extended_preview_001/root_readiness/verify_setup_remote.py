from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
B = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
g = load('root_preview_guard', B/'direct_omni_extended_preview_guard_001/launch_guarded_remote.py')
g.bind_selection(sys.argv[1:])
g.verify_frozen(B/'direct_omni_extended_preview_guard_001', '4454f8083ad0983f352b9d1e78a7ea537c21835dd53deba4dfe7ec8b2c13c413')
g.require_final_bindings()
assert not g.OUTPUT.exists() and not g.PAUSE.exists()
g.verify_previous_owner()
h = load('root_actual_preview_host', g.HOST)
identity = g.validate_preview_inputs(h)
args = SimpleNamespace(source=g.SOURCE, contract=g.CONTRACT, supervisor_source=g.SUPERVISOR_SOURCE,
    adapter=g.ADAPTER, pilot=g.PILOT, checkpoint=g.CHECKPOINT,
    checkpoint_sha256=g.CHECKPOINT_SHA256, campaign_sha256=g.CAMPAIGN_SHA256,
    output=g.OUTPUT, isaaclab=Path('/home/orionh/IsaacLab'), host_freeze_sha256=g.HOST_FREEZE_SHA256)
parent = h.load_supervisor(args)
assert callable(parent.run_owned)
assert sha(Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')) == '35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
print(json.dumps({'passed':True, 'python':sys.version, 'identity':identity,
    'actual_installed_host_load_supervisor_passed':True, 'no_run_owned_called':True,
    'no_GPU':True, 'output_created':g.OUTPUT.exists(), 'pause_created':g.PAUSE.exists()}, indent=2))
