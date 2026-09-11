"""Actual installed host setup, standard library only and no external processes."""
from pathlib import Path
from types import SimpleNamespace, CodeType
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
    host = load('_root_actual_host', base/'standing_host_007/launch_standing_spark.py')
    args = SimpleNamespace(source=base/'standing_source_005', asset=base/'asset_001', admission=base/'native_actuation_001/actuation', num_envs=32, standing_one=base/'native_standing_006/standing',
        supervisor_source=prior/'reference_physics_source_009', output=base/'native_standing32_003',
        isaaclab=Path('/home/orionh/IsaacLab'))
    args.host_freeze_sha256 = host.verify_own_bundle()
    host.require_fresh_output(args)
    identity = host.verify_inputs(args)
    sys.path.insert(0, str(args.supervisor_source/'tools'))
    original = load('_root_original_supervisor', args.supervisor_source/'tools/launch_reference_physics_spark.py')
    parent = host.load_supervisor(args, identity)
    adapter = load('_root_deadline_adapter', base/'standing_host_007/deadline_adapter.py')
    proof = adapter.inspect_supervisor(args.supervisor_source/'tools/launch_reference_physics_spark.py')
    code = compile(proof['adapted_function_source'], str((args.supervisor_source/'tools/launch_reference_physics_spark.py').resolve()) + '::' + adapter.SCHEMA, 'exec')
    expected = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'run_owned')
    assert parent.run_owned.__code__ == expected
    assert parent.run_owned.__globals__ is vars(parent)
    assert parent.CANONICAL_DEADLINE_ADAPTER == {k:v for k,v in proof.items() if k not in ('original_function_source','adapted_function_source','_original_module_source')}
    assert parent.owned_container.__code__ == original.owned_container.__code__
    assert parent.RUNTIME_TREE == identity['inspector_freeze_sha256']
    command = parent.command(args.source, args.output, 'cpu-setup-no-container', 'standing')
    assert '/standing/run_standing.py' in command
    assert str(args.admission)+':/admission:ro' in command
    for name in ('run_standing.py', 'standing_session.py', 'standing_contract.py'):
        compile((args.source/name).read_text(), str(args.source/name), 'exec')
    assert not any(k in sys.modules for k in ('torch', 'numpy', 'isaaclab'))
    assert not args.output.exists()
print(json.dumps({'passed': True, 'observed_unix': time.time(), 'python': platform.python_version(),
    'identity': identity, 'host_freeze_sha256': args.host_freeze_sha256,
    'owned_container_original_code_identical': True, 'run_owned_exact_three_seam_adapter': True, 'deadline_adapter_proof': parent.CANONICAL_DEADLINE_ADAPTER, 'external_process_calls': 0, 'GPU_calls': 0,
    'native_entry_compiles': True, 'output_created': False}, indent=2))
