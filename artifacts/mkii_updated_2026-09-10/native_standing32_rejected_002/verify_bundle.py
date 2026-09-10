"""Portable static evidence check. Never imports native code or reads remote files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INVOCATION = '1a38495bda9f4fefa4e5585574aabc56'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def inventory(root):
    require(not root.is_symlink(), 'Symbolic root')
    result = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'Symbolic payload: ' + str(p))
        if p.is_file():
            result[p.relative_to(root).as_posix()] = sha(p)
    return result

def summarize(report):
    rows = report['replicas']
    require(report['num_envs'] == 32 and [r['env'] for r in rows] == list(range(32)), 'Replica coverage')
    require(report['controls'] == 1000 and report['substeps'] == 8000, 'Reported acquisition counts')
    require(report['all_pass'] is False, 'False standing admission')
    require(all(r['pass'] == (not r['failed_physical_bounds'] and r['quiet']['pass']) for r in rows), 'Incoherent combined verdict')
    require(all(r['quiet']['pass'] == (not r['quiet']['failed_bounds']) for r in rows), 'Incoherent quiet verdict')
    require(all(r['quiet']['window_samples'] == 800 and r['quiet']['window_duration_s'] == 16.0 for r in rows), 'Quiet window changed')
    physical = [r for r in rows if r['failed_physical_bounds']]
    quiet = [r for r in rows if not r['quiet']['pass']]
    require(all(r['failed_physical_bounds'] == ['six_toe_support'] for r in physical), 'Physical failure categories changed')
    require(all(r['quiet']['failed_bounds'] == ['max_joint_velocity_rms_rad_s'] for r in quiet), 'Quiet failure categories changed')
    return {
        'schema': 'recorded_standing_report_summary_v1',
        'method': 'Aggregate the authenticated native standing_report.json; no local trace or contact rescore.',
        'num_envs': 32, 'controls': 1000, 'substeps': 8000,
        'combined_pass_count': sum(r['pass'] for r in rows),
        'physical_pass_count': 32 - len(physical), 'quiet_pass_count': 32 - len(quiet),
        'support_failure_envs': [r['env'] for r in physical],
        'quiet_failure_envs': [r['env'] for r in quiet],
        'missing_six_toe_substeps_by_env': {str(r['env']): r['physical']['post_settle_missing_six_toe_substeps'] for r in rows},
        'missing_six_toe_env_substeps_total': sum(r['physical']['post_settle_missing_six_toe_substeps'] for r in rows),
        'max_requested_all_substeps_nm': max(r['physical']['max_requested_all_substeps_nm'] for r in rows),
        'max_applied_all_substeps_nm': max(r['physical']['max_applied_all_substeps_nm'] for r in rows),
        'max_saturation_fraction_400hz': max(r['physical']['max_requested_saturation_fraction_400hz'] for r in rows),
        'nonfoot_env_substeps_total': sum(r['physical']['all_controlled_nonfoot_substeps'] for r in rows),
        'max_quiet_sdk_joint_rms_rad_s': max(r['quiet']['max_joint_velocity_rms_rad_s'] for r in rows),
        'max_quiet_joint_position_range_rad': max(r['quiet']['max_joint_position_range_rad'] for r in rows),
        'max_quiet_planar_excursion_m': max(r['quiet']['max_planar_excursion_m'] for r in rows),
        'max_quiet_heading_excursion_deg': max(r['quiet']['max_heading_excursion_deg'] for r in rows),
        'terminations': sum(r['quiet']['terminations'] for r in rows),
        'truncations': sum(r['quiet']['truncations'] for r in rows),
        'gates_unchanged_from_recorded_native_report': report['gates'],
        'training_allowed': False, 'self_contained_raw_replay': False,
    }

def check_semantics(root, audit):
    require(audit['audit_verified'] is True and audit['errors'] == [], 'Remote audit did not verify')
    require(audit['expected_invocation'] == INVOCATION == audit['unit']['InvocationID'], 'Invocation mismatch')
    require(audit['unit']['MainPID'] == '0' and audit['unit']['ExecMainStatus'] == '1' and audit['unit']['ActiveState'] == 'failed', 'Host failure changed')
    require(audit['raw_acquisition_completed'] is True and audit['standing_completed'] is False, 'Acquisition/admission conflated')
    require(audit['physical_admission'] is False and audit['training_allowed'] is False, 'False admission')
    require(audit['native_validation']['passed'] is False, 'Rejected native validator changed')
    campaign = read(root/'curated/run/campaign.json')
    job = read(root/'curated/run/jobs/standing.json')
    state = read(root/'curated/run/standing/state.json')
    report = read(root/'curated/run/standing/standing_report.json')
    session = read(root/'curated/run/standing/session.json')
    require(campaign == audit['campaign'] and job == audit['job'] and state == audit['native_state'] and report == audit['standing_report'], 'Embedded audit/raw JSON differ')
    require(campaign['status'] == job['status'] == 'failed', 'Host campaign/job status changed')
    require(campaign['error'] == job['error'] == "TimeoutError('Standing phase exceeded ten-minute bound')", 'Timeout evidence changed')
    require(job['deadline_seconds'] == 600 and job['app_ready_deadline_seconds'] == 90 and job['cleanup_checked'] is True, 'Deadline/cleanup evidence changed')
    require(state['status'] == 'completed' and state['explicit_steps_completed'] == 8000 and state['standing_pass'] is False, 'Native terminal evidence changed')
    require(state['errors'] == [] and state['inputs_unchanged'] is True, 'Native integrity changed')
    require(session['steps'] == session['captured_steps'] == 8000 and session['controls'] == 1000 and session['all_rows_recorded'] is True, 'Session coverage changed')
    require(session['reset_count'] == 1 and len(session['root_paths']) == 32 and len(session['joint_names']) == 18 and len(session['body_names']) == 19, 'Session identity changed')
    require(session['substep_files'] == ['substeps_%03d.npz' % i for i in range(10)], 'Substep inventory changed')
    require(audit['owned_absence']['recorded_id_available'] is True, 'Missing owned identity')
    identifiers = audit['owned_absence']['identifiers']
    require(set(identifiers) == {job['container_name'], job['container_id']}, 'Exact owned identifiers mismatch')
    require(all(r['absent'] is True for r in identifiers.values()), 'Owned cleanup not proven')
    require(audit['restoration']['passed'] is True and audit['restoration']['persistent_reservation_release_attempted'] is False, 'Restoration scope changed')
    require(read(root/'curated/forecast_pause/restored.json')['persistent_reservation_release_attempted'] is False, 'Persistent reservation released')
    summary = summarize(report)
    require((summary['combined_pass_count'], summary['physical_pass_count'], summary['quiet_pass_count']) == (11, 11, 25), 'Reported pass counts changed')
    require(summary == read(root/'SUMMARY.json'), 'Summary differs from recorded report')
    return summary

def verify(root=ROOT, expected_bundle=None):
    bundle = root/'BUNDLE_SHA256.json'
    require(bundle.is_file(), 'Missing outer bundle')
    if expected_bundle:
        require(sha(bundle) == expected_bundle, 'Wrong outer bundle hash')
    actual = inventory(root); actual.pop('BUNDLE_SHA256.json')
    require(actual == read(bundle), 'Missing, changed or extra publication payloads')
    provenance = read(root/'PROVENANCE.json')
    audit = read(root/'terminal/audit.json')
    for name, row in provenance['components'].items():
        actual = inventory(root/name)
        if 'freeze_sha256' in row:
            require(actual.pop('FREEZE_SHA256.json') == row['freeze_sha256'], 'Wrong '+name+' freeze')
            require(actual == read(root/name/'FREEZE_SHA256.json') and len(actual) == row['payloads'], 'Wrong '+name+' inventory')
            if name in ('source', 'host', 'guard'):
                for prefix in ('input_', 'final_input_'):
                    check = audit[prefix+name]
                    require(check['passed'] is True and check['manifest_sha256'] == row['freeze_sha256'] and check['payloads'] == row['payloads'], 'Remote/local input mismatch')
        else:
            require(actual == row['snapshot'], 'Root/terminal snapshot mismatch')
    raw = read(root/'REMOTE_RAW_INVENTORY.json'); selected = read(root/'RAW_SELECTION.json')
    require(raw == audit['raw_inventory'] and audit['raw_inventory_stable'] is True, 'Complete recorded inventory mismatch')
    require(len(raw) == audit['raw_payloads'] == 36 and sum(r['size_bytes'] for r in raw.values()) == audit['raw_total_bytes'] == 4969155341, 'Raw inventory totals')
    require(sha(root/'terminal/audit.json') == selected['audit_sha256'], 'Selection lost audit binding')
    require(selected['self_contained_raw_replay'] is False, 'False self-contained claim')
    require(set(selected['selected']).isdisjoint(selected['remote_only']) and set(selected['selected']) | set(selected['remote_only']) == set(raw), 'Selection is not exact partition')
    copied = {}
    for category in ('selected', 'remote_only'):
        for name, row in selected[category].items():
            require({k: row[k] for k in ('sha256', 'size_bytes')} == raw[name], 'Selected raw pin mismatch')
            prefix, suffix = name.split('/', 1)
            require(row['remote_path'] == selected['remote_roots'][prefix]+'/'+suffix, 'Wrong remote path')
            if category == 'selected':
                require(0 <= row['size_bytes'] <= 2000000 and row['stored_path'] == 'curated/'+name, 'Unsafe selection')
                path = root/row['stored_path']
                require(path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], 'Changed selected bytes')
                copied[name] = row['sha256']
            else:
                require(row['size_bytes'] > 2000000 and not (root/'curated'/name).exists(), 'Wrong exclusion')
    require(inventory(root/'curated') == copied and len(copied) == 24 and len(selected['remote_only']) == 12, 'Curated inventory mismatch')
    require(sum(raw[k]['size_bytes'] for k in copied) == selected['selected_bytes'] == 3281218, 'Curated total')
    require(sum(r['size_bytes'] for r in selected['remote_only'].values()) == selected['remote_only_bytes'] == 4965874123, 'Remote-only total')
    fetch = read(root/'FETCH_VERIFICATION.json')
    require(fetch['local_copy_sha256'] == copied and fetch['all_selected_size_and_sha256_verified'] is True and fetch['remote_mutation'] is False, 'Fetch receipt mismatch')
    require(fetch['selection_sha256'] == sha(root/'RAW_SELECTION.json') and fetch['audit_sha256'] == sha(root/'terminal/audit.json'), 'Fetch binding mismatch')
    for name, digest in audit['native_state']['outputs'].items():
        require(raw['run/standing/'+name]['sha256'] == digest, 'Native output seal mismatch')
    summary = check_semantics(root, audit)
    return {'verified': True, 'bundle_sha256': sha(bundle), 'publication_payloads': len(read(bundle)),
            'curated_raw_files': 24, 'curated_raw_bytes': 3281218,
            'remote_only_raw_files': 12, 'remote_only_raw_bytes': 4965874123,
            'recorded_combined_pass_count': summary['combined_pass_count'],
            'host_600_second_timeout': True, 'native_acquisition_completed': True,
            'physical_admission': False, 'training_allowed': False,
            'self_contained_raw_replay': False, 'native_or_remote_execution': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-bundle')
    args = parser.parse_args()
    print(json.dumps(verify(expected_bundle=args.expected_bundle), indent=2, sort_keys=True))
