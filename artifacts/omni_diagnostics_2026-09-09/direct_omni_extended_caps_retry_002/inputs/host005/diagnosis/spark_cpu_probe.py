"""Read-only Spark Python probe; adapter source is sent in memory via stdin.

No compilation output is executed except the explicit trusted adapter module.
The whole supervisor compilation remains code-only. No run_owned call occurs.
"""
import ast
import base64
import copy
import dis
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

sys.dont_write_bytecode = True
payload = json.loads(sys.stdin.read())
def prohibit_writes(event, args):
    if event == 'open' and len(args) > 2 and args[2] & (1 | 2 | 64 | 512 | 1024):
        raise RuntimeError('Unexpected writable open in read-only CPU probe')
    if event in ('subprocess.Popen', 'os.system', 'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir'):
        raise RuntimeError('Unexpected process/filesystem mutation in read-only CPU probe: ' + event)
sys.addaudithook(prohibit_writes)
BASE = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
supervisor = BASE/'reference_physics_source_009'
source = supervisor/'tools/launch_reference_physics_spark.py'
host_path = BASE/'direct_omni_train_host_004/launch_train_spark.py'
adapter_virtual_path = BASE/'direct_omni_train_host_005/deadline_adapter.py'
adapter_bytes = base64.b64decode(payload['adapter_base64'])
assert hashlib.sha256(adapter_bytes).hexdigest() == payload['adapter_sha256']
assert hashlib.sha256(host_path.read_bytes()).hexdigest() == payload['host_launcher_sha256']
spec = importlib.util.spec_from_file_location('exact_host005_unchanged_launcher', host_path)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
args = types.SimpleNamespace(allocation='pilot', supervisor_source=supervisor)
parent = host.load_supervisor(args)
text = source.read_text()
node = next(n for n in ast.parse(text, filename=str(source)).body if isinstance(n, ast.FunctionDef) and n.name == 'run_owned')
def extract(code):
    return next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'run_owned')
isolated = extract(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), str(source), 'exec'))
whole = extract(compile(text, str(source), 'exec', dont_inherit=True))
actual = parent.run_owned.__code__
different = [k for k in dir(actual) if k.startswith('co_') and k != 'co_lnotab' and not callable(getattr(actual, k)) and getattr(actual, k) != getattr(isolated, k)]
old_adapter = host.load_deadline_adapter()
try:
    old_adapter.install(parent, source)
except ValueError as error:
    old_failure = str(error)
else:
    old_failure = None
assert old_failure == 'Loaded run_owned differs from exact verified source'
assert actual != isolated and actual == whole
def short_instructions(code):
    return [(i.offset, i.opname, i.argrepr) for i in dis.get_instructions(code)]
a, b = short_instructions(actual), short_instructions(isolated)
diff_examples = [{'index': i, 'loaded': x, 'isolated': y} for i, (x, y) in enumerate(zip(a, b)) if x != y][:8]
adapter = types.ModuleType('_direct_host004_deadline_adapter')
adapter.__file__ = str(adapter_virtual_path)
exec(compile(adapter_bytes, str(adapter_virtual_path), 'exec', dont_inherit=True), vars(adapter))
original_read_bytes = Path.read_bytes
original_load_module = host.load_module
def read_bytes(path):
    return adapter_bytes if path == adapter_virtual_path else original_read_bytes(path)
def load_module(name, path):
    if name == '_direct_host004_deadline_adapter':
        return adapter
    return original_load_module(name, path)
with patch.object(Path, 'read_bytes', read_bytes), patch.object(host, 'load_module', load_module):
    args.allocation = 'extended'
    repaired = host.load_supervisor(args)
    metadata = repaired.DIRECT_DEADLINE_ADAPTER
    assert repaired.run_owned.__globals__ is vars(repaired)
    assert metadata['adapter_sha256'] == payload['adapter_sha256']
    assert metadata['phase_deadline_seconds']['train'] == 1800
    assert metadata['app_ready_deadline_seconds'] == 90
    args.allocation = 'pilot'
    altered = host.load_supervisor(args)
    code = altered.run_owned.__code__
    altered.run_owned.__code__ = code.replace(co_consts=tuple(601 if type(v) is int and v == 600 else v for v in code.co_consts))
    try:
        adapter.install(altered, source)
    except ValueError as error:
        changed_code_rejection = str(error)
    else:
        raise AssertionError('Altered code was accepted')
    for allocation in ('smoke', 'pilot'):
        args.allocation = allocation
        unchanged = host.load_supervisor(args)
        assert unchanged.run_owned.__code__ == whole
        assert not hasattr(unchanged, 'DIRECT_DEADLINE_ADAPTER')
assert source.read_text() == text
print(json.dumps({
    'passed': True, 'python': sys.version, 'adapter_sha256': payload['adapter_sha256'],
    'host_launcher_sha256': payload['host_launcher_sha256'],
    'supervisor_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'old_failure_reproduced': old_failure, 'loaded_equals_isolated': actual == isolated,
    'loaded_equals_whole_module': actual == whole, 'different_code_fields': different,
    'first_disassembly_differences': diff_examples, 'full_code_comparison_retained': True,
    'actual_host_load_supervisor_extended_passed': True, 'adapter_metadata': metadata,
    'altered_same_global_function_rejected': changed_code_rejection,
    'smoke_pilot_code_unchanged': True, 'no_run_owned_called': True,
    'transport': 'new adapter supplied in memory; unchanged actual host launcher loaded from disk',
    'remote_writes_or_subprocesses': False,
}, indent=2))
