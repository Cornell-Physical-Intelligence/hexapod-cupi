from pathlib import Path
import hashlib, importlib.util, json, math, sys
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory(path, omit):
    assert not any(p.is_symlink() for p in path.rglob('*')), 'Unexpected symlink'
    return {p.relative_to(path).as_posix(): sha(p) for p in path.rglob('*')
            if p.is_file() and p != path/omit and '__pycache__' not in p.parts}

payloads = inventory(root, 'BUNDLE_SHA256.json')
assert payloads == read(root/'BUNDLE_SHA256.json'), 'Published payload mismatch'
provenance = read(root/'PROVENANCE.json')
for name, component in provenance['preserved_components'].items():
    if 'freeze_sha256' in component:
        p = root/name
        assert sha(p/'FREEZE_SHA256.json') == component['freeze_sha256'], name
        assert inventory(p, 'FREEZE_SHA256.json') == read(p/'FREEZE_SHA256.json'), name
raw = read(root/'terminal/RAW_SHA256.json')
assert len(raw) == 32
for name, digest in raw.items():
    assert sha(root/'terminal'/name) == digest, name
assert sum((root/'terminal'/name).stat().st_size for name in raw) == 266352
assert set(raw) == {p.relative_to(root/'terminal').as_posix() for folder in ('run', 'forecast_pause') for p in (root/'terminal'/folder).rglob('*') if p.is_file()}
a = read(root/'terminal/audit.json')
r = read(root/'RESULT.json')
s = read(root/'terminal/run/standing/state.json')
session = read(root/'terminal/run/standing/session.json')
campaign = read(root/'terminal/run/campaign.json')
job = read(root/'terminal/run/jobs/standing.json')
assert a['audit_verified'] and a['errors'] == [] and a['raw_inventory_stable']
assert a['terminal_outcome'] == r['outcome'] == 'authentic_terminal_failure'
assert a['unit']['InvocationID'] == r['invocation'] == 'f3cd14c1078e47c891d97718643b22aa'
assert a['unit']['MainPID'] == '0' and a['unit']['ExecMainStatus'] == '1'
assert a['unit']['ActiveState'] == 'failed'
assert not a['standing_completed'] and not a['native_validation']['passed']
assert s['status'] == campaign['status'] == job['status'] == 'failed'
assert campaign['terminal_inputs_unchanged'] and s['inputs_unchanged']
assert job['cleanup_checked'] and len(a['owned_absence']['identifiers']) == 2
assert all(v['absent'] for v in a['owned_absence']['identifiers'].values())
assert a['restoration']['passed']
assert session['steps'] == 14 and session['captured_steps'] == 13 and session['controls'] == 1
assert session['all_rows_recorded'] is False and session['reset_count'] == 1
assert session['failure'] == "ValueError('Invalid patch normal')"
for flags in (s, campaign, r):
    for name in ('physical_admission', 'training_allowed'):
        assert flags[name] is False
for name in ('source', 'host', 'guard', 'ownership_supervisor', 'asset'):
    assert a['input_'+name] == a['final_input_'+name]
    assert a['input_'+name]['passed']
for label, component in [('source','source'), ('host','host'), ('guard','guard_executed')]:
    assert a['input_'+label]['manifest_sha256'] == provenance['preserved_components'][component]['freeze_sha256']
assert read(root/'root_checks/dispatch_001.json')['returncode'] == 1
assert '--standing-compute-apps' in read(root/'root_checks/dispatch_001.json')['stderr']
no_mutation = json.loads(read(root/'root_checks/first_guard_no_mutation.json')['stdout'])
assert not no_mutation['native_output_exists'] and not no_mutation['pause006_exists']
assert 'LoadState=not-found' in no_mutation['owner']
assert read(root/'root_checks/actual_spark_setup_002.json')['passed']
assert read(root/'root_checks/dispatch_002.json')['returncode'] == 0
spec = importlib.util.spec_from_file_location('_contact_replay', root/'portable_contact_check.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
replay = module.check(root/'terminal/run/standing/failed_contact_buffer.npz')
for key, value in replay.items():
    expected = r['diagnosis'][key]
    assert value == expected or (isinstance(value, float) and math.isclose(value, expected, abs_tol=1e-15)), key
assert sha(root/'terminal/audit.json') == r['terminal_audit_sha256']
assert sha(root/'contact_diagnosis/report.json') == r['diagnosis']['report_sha256']
print(f'PASS {len(payloads)} payloads; all32 raw files; authentic failed standing, exact zero-contact replay; no physical or learning admission')
