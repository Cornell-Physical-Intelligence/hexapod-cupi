#!/usr/bin/env python3
"""Read-only installed Python3.12 setup proof; execute only after audited32 pass."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import CodeType, SimpleNamespace
from unittest.mock import patch

BASE = Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
HOST_FREEZE = 'efd7169161fa7586fd76741f129a66f66119253e510214080e6d4f33628e2130'
HOST_RUNTIME = '254384b956b602f81a494b57d361cd4858a9382b78ddbcecd488ba1c7d462890'
ADAPTER = 'e3ec9b1ea42ec48b8d2a6724ae25817c944e9ad3c3cd2ace17fcdd8172a2f754'
PPO_FREEZE = '970cd1766366c9bdba3239205695e998da80819d2283ce0e261a3b265ca48183'
STANDING_FREEZE = 'c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
STANDING_ONE = '33920a4af296ed124643fe61a16f4840c19949b5f6e434327d7cd7ea9953cd3f'
SUPERVISOR_MAP = '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE = '9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
COORDINATION = '649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f'
PHASE = 'canonical_ppo_smoke'


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def hash_argument(value):
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise argparse.ArgumentTypeError('An actual audited lowercase SHA-256 is required')
    return value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), 'Wrong module origin')
    return module


def forbid(*args, **kwargs):
    raise AssertionError('External process forbidden in actual CPU setup check')


def readonly_audit(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        writing_mode = isinstance(mode, str) and any(c in mode for c in 'wax+')
        writing_flags = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        require(not (writing_mode or writing_flags), 'Filesystem writes forbidden in CPU setup')
    if event in ('os.mkdir', 'os.rename', 'os.remove', 'os.rmdir', 'os.link', 'os.symlink', 'os.chmod', 'os.chown', 'os.truncate', 'os.utime', 'os.system', 'os.fork', 'os.forkpty', 'os.exec', 'os.posix_spawn', 'subprocess.Popen', 'socket.connect', 'socket.bind'):
        raise AssertionError('Mutation/process/network forbidden in CPU setup: ' + event)


def verify_host_before_import(host_dir):
    require(not host_dir.is_symlink(), 'Symbolic host root')
    require(sha(host_dir / 'FREEZE_SHA256.json') == HOST_FREEZE, 'Host004 freeze mismatch')
    declared = read(host_dir / 'FREEZE_SHA256.json')
    actual = {}
    for path in sorted(host_dir.rglob('*')):
        require(not path.is_symlink(), 'Symbolic host payload')
        if path.is_file() and path.name != 'FREEZE_SHA256.json':
            actual[path.relative_to(host_dir).as_posix()] = sha(path)
    require(len(declared) == 38 and actual == declared, 'Host004 exact38 payload map mismatch')
    require(sha(host_dir / 'launch_ppo_spark.py') == HOST_RUNTIME, 'Host launcher mismatch')
    require(sha(host_dir / 'deadline_adapter.py') == ADAPTER, 'Deadline adapter mismatch')


def expected_command(args):
    mounts = [(args.source, '/ppo'), (args.standing_source, '/standing'), (args.asset, '/asset'),
              (args.admission, '/admission'), (args.standing_one, '/standing_one'),
              (args.standing32, '/standing32'), (args.bindings, '/bindings.json')]
    return ['docker', 'compose', '--env-file', 'docker/.env.base', '-f', 'docker/docker-compose.yaml',
            '--profile', 'base', 'run', '--rm', '--no-deps', '--name', 'cpu-setup-no-container',
            '-w', '/output', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'PYTHONPATH=/ppo:/workspace/isaaclab/source/isaaclab', '-v', str(args.output) + ':/output:rw',
            *[item for source, target in mounts for item in ('-v', str(source) + ':' + target + ':ro')],
            '--entrypoint', '/workspace/isaaclab/_isaac_sim/python.sh', 'isaac-lab-base',
            '/ppo/run_native_smoke.py', '--asset', '/asset', '--admission', '/admission',
            '--standing-source', '/standing', '--standing-one', '/standing_one',
            '--standing32', '/standing32', '--bindings', '/bindings.json',
            '--output', '/output/' + PHASE, '--device', 'cuda:0', '--headless']


def prove(args, requested, host_dir):
    verify_host_before_import(host_dir)
    host = load('_actual_ppo_host004', host_dir / 'launch_ppo_spark.py')
    require(host.SOURCE_FREEZE == PPO_FREEZE and host.STANDING_FREEZE == STANDING_FREEZE, 'Wrong source selection')
    require(host.SUPERVISOR_MAP == SUPERVISOR_MAP and host.SUPERVISOR_CODE == SUPERVISOR_CODE, 'Wrong ownership source')
    require(host.EXPECTED_COORDINATION == COORDINATION and sha(host.COORDINATION) == COORDINATION, 'Coordination changed')
    require(sha(args.standing_one / 'state.json') == STANDING_ONE, 'Wrong native6 state')
    require(sha(args.standing32 / 'state.json') == requested.standing32_state_sha256, 'Actual32 state differs from root audit pin')
    require(sha(args.bindings) == requested.bindings_sha256, 'Enabled binding differs from root pin')
    bindings = read(args.bindings)
    for key, value in {'standing_source_freeze_sha256': STANDING_FREEZE,
                       'standing1_state_sha256': STANDING_ONE,
                       'standing32_state_sha256': requested.standing32_state_sha256,
                       'ready_for_native_dispatch': True}.items():
        require(bindings.get(key) == value, 'Pending/wrong actual binding: ' + key)
    args.host_freeze_sha256 = host.verify_own_bundle(HOST_FREEZE)
    host.require_fresh_output(args)
    identity = host.verify_inputs(args)  # Real full source, assets, actuation,1/32 raw+gate validation; no stub.
    require(identity['policy_lineage']['standing32_state_sha256'] == requested.standing32_state_sha256, 'Wrong32 lineage')
    require(identity['bindings_sha256'] == requested.bindings_sha256, 'Wrong enabled receipt lineage')
    for root, manifest, expected in ((args.source, 'FREEZE_SHA256.json', 74),
                                    (args.standing_source, 'FREEZE_SHA256.json', 109),
                                    (args.supervisor_source, 'campaign_source_hashes.json', 926)):
        require(len(read(root / manifest)) == expected, 'Unexpected mapped source count')
    allocation = {'fresh_neutral_controls': 1000, 'fresh_neutral_substeps': 8000,
                  'policy_controls': 48, 'policy_substeps': 384, 'total_controls': 1048, 'total_substeps': 8384}
    for key, value in allocation.items():
        require(identity.get(key) == value, 'Fixed allocation changed: ' + key)
    for key, value in {'replicas': 32, 'controls_per_update': 24, 'updates': 2, 'transitions': 1536,
                       'actor_width': 405, 'critic_width': 408, 'control_dt': .02, 'physics_dt': .0025,
                       'substeps': 8, 'checkpoint_input': None, 'auto_reset': False,
                       'automatic_continuation': False, 'walking_objective_adopted': False,
                       'command': 'allzero48controls; smoke-only'}.items():
        require(key in identity['protocol'] and identity['protocol'][key] == value, 'Fixed PPO protocol changed: ' + key)

    supervisor = args.supervisor_source / 'tools/launch_reference_physics_spark.py'
    sys.path.insert(0, str(supervisor.parent))
    original = load('_actual_original_supervisor', supervisor)
    adapter = load('_actual_ppo_deadline_adapter', host_dir / 'deadline_adapter.py')
    proof = adapter.inspect_supervisor(supervisor)
    original_module_code = compile(proof['_original_module_source'], str(supervisor.resolve()), 'exec', dont_inherit=True)
    originals = {c.co_name: c for c in original_module_code.co_consts if isinstance(c, CodeType)}
    require(original.run_owned.__code__ == originals['run_owned'], 'Original full-module run_owned CodeType differs')
    require(original.owned_container.__code__ == originals['owned_container'], 'Original owned_container CodeType differs')
    require(original.run_owned.__globals__ is vars(original), 'Original globals mismatch')
    parent = host.load_supervisor(args, identity)
    adapted_module_code = compile(proof['adapted_function_source'], str(supervisor.resolve()) + '::' + adapter.SCHEMA, 'exec', dont_inherit=True)
    expected = next(c for c in adapted_module_code.co_consts if isinstance(c, CodeType) and c.co_name == 'run_owned')
    require(parent.run_owned.__code__ == expected, 'Three-seam adapted CodeType differs')
    require(parent.run_owned.__globals__ is vars(parent), 'Adapted function is not in actual supervisor globals')
    require(parent.run_owned.__defaults__ is None and parent.run_owned.__kwdefaults__ is None and parent.run_owned.__closure__ is None, 'Unexpected adapted defaults/closure')
    require(parent.owned_container.__code__ == original.owned_container.__code__, 'Ownership function changed')
    require(parent.owned_container.__globals__ is vars(parent), 'Ownership globals differ')
    metadata = {key: value for key, value in proof.items() if key not in ('original_function_source', 'adapted_function_source', '_original_module_source')}
    require(parent.CANONICAL_DEADLINE_ADAPTER == metadata, 'Adapter metadata differs')
    require(metadata['source_substitutions'] == 3 and metadata['phase_deadline_seconds'] == {PHASE: 1200}, 'Wrong deadline seams/bound')
    require(metadata['app_ready_deadline_seconds'] == 90 and metadata['cleanup_AST_unchanged'] is True, 'Startup/cleanup changed')
    helper = getattr(parent, adapter.HELPER_NAME)
    require(helper.__code__ == adapter.phase_deadline_seconds.__code__ and helper(args, PHASE) == 1200, 'Wrong installed deadline helper')
    try:
        helper(args, 'unapproved_phase')
    except ValueError:
        pass
    else:
        raise AssertionError('Unapproved deadline phase accepted')
    require(parent.RUNTIME_TREE == PPO_FREEZE == identity['runtime_binding']['runtime_tree_sha256'], 'Wrong actual supervisor runtime binding')
    command_closure = inspect.getclosurevars(parent.command)
    verified_closure = inspect.getclosurevars(parent.verified_source)
    save_closure = inspect.getclosurevars(parent.save)
    require(parent.command.__globals__ is vars(host) and command_closure.nonlocals['args'] is args, 'Wrong launch closure/globals')
    require(parent.verified_source.__globals__ is vars(host) and verified_closure.nonlocals['args'] is args and verified_closure.nonlocals['identity'] is identity, 'Wrong input reverification closure/globals')
    require(parent.save.__globals__ is vars(host) and save_closure.nonlocals['deadline_proof'] == metadata, 'Wrong job metadata closure')
    require(save_closure.nonlocals['original_save'].__globals__ is vars(parent), 'Wrong underlying metadata save globals')
    command = parent.command(args.source, args.output, 'cpu-setup-no-container', PHASE)
    require(command == expected_command(args), 'Exact native command/read-only mounts differ')
    for path in (args.source / 'run_native_smoke.py', args.source / 'canonical_direct_ppo/native_entry_adapter.py',
                 args.standing_source / 'run_standing.py', args.standing_source / 'standing_session.py'):
        compile(path.read_text(), str(path), 'exec', dont_inherit=True)
    forbidden_imports = ('torch', 'numpy', 'isaaclab', 'isaacsim', 'omni')
    require(not any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden_imports for name in sys.modules), 'Native/array runtime imported')
    host.require_fresh_output(args)
    require(not args.output.exists() and not args.output.is_symlink(), 'PPO output created during CPU check')
    return {'passed': True, 'observed_unix': time.time(), 'python': platform.python_version(),
            'identity': identity, 'host_freeze_sha256': HOST_FREEZE, 'host_runtime_sha256': HOST_RUNTIME,
            'actual_standing32_state_sha256': requested.standing32_state_sha256,
            'external_enabled_bindings_sha256': requested.bindings_sha256,
            'mapped_source_counts': {'ppo': 74, 'standing': 109, 'supervisor': 926, 'host': 38},
            'owned_container_original_code_identical': True, 'run_owned_exact_three_seam_adapter': True,
            'actual_function_globals_and_metadata_verified': True, 'deadline_adapter_proof': metadata,
            'command': command, 'native_entry_compiles': True, 'external_process_calls': 0,
            'GPU_calls': 0, 'output_created': False, 'physical_admission': False, 'Stage2_complete': False,
            'scope': 'Installed CPU setup proof after audited standing admission; no native PPO execution.'}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument('--standing32-state-sha256', required=True, type=hash_argument)
    parser.add_argument('--bindings-sha256', required=True, type=hash_argument)
    requested = parser.parse_args()
    require(sys.version_info[:2] == (3, 12), 'Run with actual Spark Python3.12')
    require(sys.flags.no_site == 1 and sys.dont_write_bytecode and sys.flags.optimize == 0, 'Use Python -B -S without optimization')
    sys.addaudithook(readonly_audit)
    args = SimpleNamespace(source=BASE / 'ppo_source_003', standing_source=BASE / 'standing_source_005',
                           asset=BASE / 'asset_001', admission=BASE / 'native_actuation_001/actuation',
                           standing_one=BASE / 'native_standing_006/standing',
                           standing32=BASE / 'native_standing32_004/standing', bindings=BASE / 'ppo_bindings_001.json',
                           supervisor_source=PRIOR / 'reference_physics_source_009',
                           output=BASE / 'canonical_ppo_smoke_001', isaaclab=Path('/home/orionh/IsaacLab'))
    with patch.object(subprocess, 'run', side_effect=forbid), patch.object(subprocess, 'Popen', side_effect=forbid), patch.object(subprocess, 'check_output', side_effect=forbid), patch.object(subprocess, 'check_call', side_effect=forbid), patch.object(subprocess, 'call', side_effect=forbid):
        result = prove(args, requested, BASE / 'ppo_host_004')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
