"""Actual installed host setup, standard library only and no external processes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib, importlib.util, json, platform, subprocess, sys, time
sys.dont_write_bytecode = True
base = Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
prior = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def forbid(*a, **kw):
    raise AssertionError('External process forbidden in actual CPU setup check')
with patch.object(subprocess, 'run', side_effect=forbid), patch.object(subprocess, 'Popen', side_effect=forbid), patch.object(subprocess, 'check_output', side_effect=forbid):
    host = load('_root_actual_host', base/'inspection_host_003/launch_inspection_spark.py')
    args = SimpleNamespace(source=base/'inspection_source_003', asset=base/'asset_001',
        supervisor_source=prior/'reference_physics_source_009', output=base/'native_inspection_003',
        isaaclab=Path('/home/orionh/IsaacLab'))
    args.host_freeze_sha256 = host.verify_own_bundle()
    host.require_fresh_output(args)
    identity = host.verify_inputs(args)
    sys.path.insert(0, str(args.supervisor_source/'tools'))
    original = load('_root_original_supervisor', args.supervisor_source/'tools/launch_reference_physics_spark.py')
    parent = host.load_supervisor(args, identity)
    for name in ('run_owned', 'owned_container'):
        assert getattr(parent, name).__code__ == getattr(original, name).__code__, name
    assert parent.RUNTIME_TREE == identity['inspector_freeze_sha256']
    command = parent.command(args.source, args.output, 'cpu-setup-no-container', 'inspection')
    assert '/inspection/run_inspection.py' in command
    for name in ('run_inspection.py', 'inspect_core.py'):
        compile((args.source/name).read_text(), str(args.source/name), 'exec')
    assert not any(k in sys.modules for k in ('torch', 'numpy', 'isaaclab'))
    assert not args.output.exists()
print(json.dumps({'passed': True, 'observed_unix': time.time(), 'python': platform.python_version(),
    'identity': identity, 'host_freeze_sha256': args.host_freeze_sha256,
    'unmodified_ownership_code_objects': True, 'external_process_calls': 0, 'GPU_calls': 0,
    'native_entry_compiles': True, 'output_created': False}, indent=2))
