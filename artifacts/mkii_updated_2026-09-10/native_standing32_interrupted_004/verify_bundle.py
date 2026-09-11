"""Portable byte/receipt verification; remote-only numeric data is not replayed."""
from pathlib import Path, PurePosixPath
import hashlib, importlib.util, json, re, sys
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root):
    r = root.resolve()
    require(not any(p.is_symlink() for p in r.rglob('*')), 'Symbolic bundle payload')
    manifest = read(r / 'BUNDLE_SHA256.json')
    actual = tree(r)
    actual.pop('BUNDLE_SHA256.json')
    require(actual == manifest, 'Publication payloads differ')
    components = read(r / 'COMPONENTS.json')
    for name, pin in components.items():
        p = r / name
        require(sha(p / 'FREEZE_SHA256.json') == pin['freeze_sha256'], 'Wrong component freeze')
        own = tree(p)
        own.pop('FREEZE_SHA256.json')
        require(own == read(p / 'FREEZE_SHA256.json') and len(own) == pin['payloads'], 'Component bytes differ')
    require(tree(r / 'root_checks') == read(r / 'ROOT_CHECKS_SHA256.json'), 'Root capture bytes differ')
    require(len(read(r / 'ROOT_CHECKS_SHA256.json')) == 12, 'Wrong root capture count')
    t = r / 'terminal'
    audit = read(t / 'audit.json')
    raw = read(t / 'RAW_SHA256.json')
    enc = read(t / 'RAW_ENCODING.json')
    omitted = read(r / 'REMOTE_ONLY_RAW.json')
    result = read(r / 'RESULT.json')
    require(audit['audit_verified'] is True and audit['errors'] == [] and audit['read_only'] is True, 'Audit failed')
    require(audit['expected_invocation'] == 'b33d418438ea438ca0805eea68e70a43', 'Wrong invocation')
    require(audit['unit']['InvocationID'] == audit['expected_invocation'] and audit['unit']['MainPID'] == '0' and audit['unit']['ExecMainStatus'] == '1' and audit['unit']['ActiveState'] == 'failed', 'Wrong terminal owner')
    require(audit['terminal_outcome'] == 'authentic_terminal_failure' and audit['standing_completed'] is False and audit['raw_acquisition_completed'] is False, 'Wrong historical verdict')
    require(raw == audit['raw_inventory'] and audit['raw_inventory_stable'] is True and len(raw) == 34, 'Wrong full remote inventory')
    require(len(enc) == 24 and len(omitted) == 10 and not set(enc) & set(omitted) and set(enc) | set(omitted) == set(raw), 'Raw local/remote partition differs')
    expected_omissions = {'run/standing/contacts.jsonl'} | {'run/standing/substeps_%03d.npz' % i for i in range(9)}
    require(set(omitted) == expected_omissions, 'Wrong remote-only payloads')
    for rel, pin in raw.items():
        path = PurePosixPath(rel)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe raw path')
        if rel in enc:
            e = enc[rel]
            require(e['storage_path'] == rel and e['encoding'] == 'identity', 'Raw transport changed')
            p = t / rel
            require(e['sha256'] == pin['sha256'] and e['size_bytes'] == pin['size_bytes'] and sha(p) == pin['sha256'] and p.stat().st_size == pin['size_bytes'], 'Original local raw differs: ' + rel)
        else:
            e = omitted[rel]
            require(e['sha256'] == pin['sha256'] and e['size_bytes'] == pin['size_bytes'], 'Remote-only metadata differs')
            require(e['remote_path'] == '/home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_standing32_004/' + rel[len('run/'):], 'Wrong preserved remote location')
            require(not (t / rel).exists(), 'Unexpected remote-only local payload')
    local_raw = {p.relative_to(t).as_posix() for folder in ('run', 'forecast_pause') for p in (t / folder).rglob('*') if p.is_file()}
    require(local_raw == set(enc), 'Unexpected/missing local raw')
    require(sum(v['size_bytes'] for v in raw.values()) == 4666042464 and sum(raw[k]['size_bytes'] for k in enc) == 3317848 and sum(v['size_bytes'] for v in omitted.values()) == 4662724616, 'Wrong raw byte totals')
    require(read(t / 'fetch_result.json') == {'verified_raw_files': 24, 'original_bytes': 3317848, 'remote_only_files': 10, 'local_stored_bytes': 3317848}, 'Wrong actual acquisition receipt')
    for name, field in [('source', 'input_source'), ('host', 'input_host'), ('guard', 'input_guard')]:
        expected = {'passed': True, 'payloads': components[name]['payloads'], 'manifest_sha256': components[name]['freeze_sha256']}
        require(audit[field] == expected and audit['final_' + field] == expected, 'Historical input audit differs: ' + name)
    for field, count in [('input_asset', 9), ('input_ownership_supervisor', 926)]:
        require(audit[field]['passed'] is True and audit[field]['payloads'] == count and audit['final_' + field] == audit[field], 'Historical prerequisite audit differs')

    state = read(t / 'run/standing/state.json')
    job = read(t / 'run/jobs/standing.json')
    campaign = read(t / 'run/campaign.json')
    require(state == audit['native_state'] and job == audit['job'] and campaign == audit['campaign'], 'Audit/raw JSON mismatch')
    require(state['status'] == 'running' and state['explicit_steps_completed'] == 0 and state['checks'] == {} and state['errors'] == [], 'Stale native state rewritten')
    require(state['identity']['inspector_freeze_sha256'] == components['source']['freeze_sha256'], 'Wrong source in actual native state')
    require(all(state[k] is False for k in ('physical_admission', 'physics_admitted', 'training_allowed')) and campaign['stage2_complete'] is False and campaign['terminal_inputs_unchanged'] is True, 'Admission/integrity overclaim')
    require(job['status'] == campaign['status'] == 'failed' and job['cleanup_checked'] is True and job['deadline_seconds'] == 1200 and job['app_ready_deadline_seconds'] == 90, 'Wrong interruption or bounds')
    require(job['error'] == campaign['error'] == "RuntimeError('Unrelated CUDA process appeared; yielding this owned job')", 'Interruption text changed')
    require(job['competitors'] == [{'process': '3803709, /home/orionh/ithaca-reconstruction/env/bin/python', 'cgroup': '0::/user.slice/user-1000.slice/session-c11072.scope'}], 'Actual competitor identity differs')
    require(not any('run/standing/' + name in raw for name in ('standing_report.json', 'session.json', 'control_trace.npz')), 'Unexpected finalized native output')
    log = (t / 'run/logs/standing.log').read_text()
    milestones = [int(value) for value in re.findall(r'CANONICAL_STANDING control=(\d+) replicas=32', log)]
    require('REFERENCE_SCREEN_APP_READY' in log and milestones == list(range(100, 1000, 100)), 'Actual startup/progress markers differ')
    auditor = load('_published_interrupted32_auditor', r / 'auditor/audit_remote.py')
    require(auditor.classify(campaign, job, state, audit['unit'], audit['native_validation']) == 'authentic_terminal_failure', 'Original terminal classification differs')
    require(auditor.restoration(read(t / 'forecast_pause/pause.json'), read(t / 'forecast_pause/restored.json'), read(t / 'forecast_pause/launch.json')) == audit['restoration'], 'Original restoration replay differs')
    ids = audit['owned_absence']['identifiers']
    require(set(ids) == {job['container_name'], job['container_id']} and all(row['absent'] is True for row in ids.values()), 'Exact owned cleanup missing')
    require(audit['exclusive_reservation']['verified'] is True and audit['restoration']['scope'] == 'per_job_snapshot_only' and audit['restoration']['persistent_reservation_release_attempted'] is False, 'Wrong historical reservation/restoration scope')
    contract = load('_published_interrupted32_contract', r / 'source/standing_contract.py')
    try:
        contract.validate_result(t / 'run/standing', state['identity'])
    except ValueError as error:
        require(str(error) == 'Incomplete standing acquisition', 'Unexpected original contract rejection: ' + repr(error))
    else:
        raise ValueError('Incomplete acquisition was admitted')
    for key in ('native_acquisition_completed', 'physical_verdict_available', 'standing32_admitted', 'physical_admission', 'training_allowed', 'stage2_complete', 'full_raw_replay_available_in_bundle', 'raw_numeric_replay_performed', 'final_physical_step_count_known'):
        require(result[key] is False, 'Publication overclaim: ' + key)
    require(result['last_logged_control_milestone'] == 900 and result['logged_control_milestones'] == milestones and result['cause_as_recorded'] == job['error'] and result['competitors_as_recorded'] == job['competitors'], 'Publication summary mismatch')
    return {'verified': True, 'bundle_payloads': len(actual), 'original_remote_inventory_files': 34,
            'original_remote_inventory_bytes': 4666042464, 'verified_local_raw_files': 24,
            'verified_local_raw_bytes': 3317848, 'remote_only_files': 10,
            'remote_only_bytes': 4662724616, 'last_logged_control_milestone': 900,
            'authentic_operational_interruption': True, 'original_contract_rejects_incomplete_acquisition': True,
            'full_raw_numeric_replay_performed': False, 'standing32_admitted': False,
            'physical_verdict_available': False, 'training_allowed': False, 'stage2_complete': False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
