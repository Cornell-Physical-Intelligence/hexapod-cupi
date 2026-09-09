#!/usr/bin/env python3
"""Verify compact replay evidence; optional --raw-dir rehashes retained raw files.

This reads files only and never starts simulation, contacts Spark, or admits PPO.
The default result verifies a recorded remote inventory, not current remote bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder, raw_dir=None):
    folder = Path(folder).resolve()
    inventory = read_json(folder/'remote_inventory.json')
    records = inventory['files']
    if isinstance(records, list):
        require(len({r['relative_path'] for r in records}) == len(records), 'Duplicate remote inventory path')
        records = {r['relative_path']: r for r in records}
    require(isinstance(records, dict) and records, 'Missing remote inventory')
    remote_root = PurePosixPath(inventory['remote_root'])
    for name, record in records.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe inventory path')
        require(record['remote_path'] == str(remote_root/relative), 'Remote path does not match run root')
        require(re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None
                and type(record['bytes']) is int and record['bytes'] >= 0
                and type(record['mtime_ns']) is int and record['mtime_ns'] > 0, 'Incomplete remote file identity')

    manifest = {}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        require(name not in manifest and PurePosixPath(name).name == name, 'Invalid local manifest path')
        require(digest(folder/name) == sha, 'Local checksum mismatch: ' + name)
        manifest[name] = sha
    actual_local = {p.name for p in folder.iterdir() if p.is_file() and p.name != 'SHA256SUMS'}
    require(set(manifest) == actual_local, 'Local manifest does not cover the complete frozen bundle')
    retrieved = read_json(folder/'retrieved_files.json')
    for name, record in retrieved.items():
        path = folder/name
        require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                'Retrieved original differs: ' + name)
        if name in records:
            require(all(record[key] == records[name][key] for key in ('bytes', 'sha256', 'remote_path')),
                    'Retrieved original differs from remote inventory: ' + name)

    report, supervisor = read_json(folder/'report.json'), read_json(folder/'supervisor.json')
    analysis = read_json(folder/'analysis_analysis.json')
    require(report['pass'] is False and report['diagnostic_complete'] is True
            and report['diagnostic_motion'] == 'validation_prefix' and report['errors'] == []
            and report['num_envs'] == 32 and report['steps_completed'] == 1000
            and report['driven_steps_completed'] == 1500 and report['control_trace_samples'] == 2500
            and report['physics_substeps'] == report['physics_samples_observed'] == report['force_writes'] == 40000
            and report['trace_samples'] == 4800 and report['physical_gate_errors']
            and report['terminated_count'] == report['truncated_count'] == 0,
            'Replay completion, coverage or failure evidence differs')
    for value in (report, analysis):
        require(value['simulation_training_admission'] is False and value['hardware_admission'] is False,
                'Diagnostic evidence cannot grant admission')
    require(supervisor['execution'] == supervisor['validator_report_status'] == 'diagnostic_complete'
            and supervisor['cleanup'] == 'removed_exact_id' and supervisor['supervisor_exit_code'] == 0
            and supervisor['source_identity_unchanged_at_finish'] is True
            and supervisor['contract'] == report['contract'], 'Supervisor source/completion/cleanup differs')
    identity = hashlib.sha256(json.dumps(report['contract']['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(identity == report['contract']['sha256'], 'Functional source identity is inconsistent')
    for field in ('source_manifest', 'source_archive'):
        item = supervisor[field]
        require(records[item['file']]['sha256'] == item['sha256'], 'Remote source archive/manifest hash differs')
    require(records[supervisor['source_archive']['file']]['bytes'] == supervisor['source_archive']['compressed_bytes'],
            'Remote source archive byte count differs')

    raw_names = []
    for field, count, start_field, first, total, width in (
            ('trace_files', 6, 'first_physics_sample', 35200, 4800, 838),
            ('control_trace_files', 31, 'first_control_step', 0, 2500, 300)):
        rows = report[field]
        require(len(rows) == count and rows == analysis[field], 'Analysis/report trace inventory differs')
        cursor = first
        for row in rows:
            name = row['file']
            require(row[start_field] == cursor and row['shape'][1:] == [32, width]
                    and row['shape'][0] > 0 and records[name]['sha256'] == row['sha256'],
                    'Trace hash, shape or sample interval differs: ' + name)
            cursor += row['shape'][0]
            raw_names.append(name)
        require(cursor == first + total, 'Trace sample coverage differs')
    require(len(set(raw_names)) == 37 and {name for name in records if name.endswith('.npz')} == set(raw_names),
            'Remote raw NPZ inventory is incomplete or contains an unexplained file')
    require(analysis['analysis_complete'] is True
            and analysis['inputs']['replay_report']['sha256'] == digest(folder/'report.json')
            and analysis['comparability']['exact_runtime_match'] is True
            and analysis['comparability']['exact_ordered_32_reset_positions_match'] is True,
            'Analysis identity or comparability differs')
    original = read_json(folder/'campaign008_report.json')
    require(digest(folder/'campaign008_report.json') == analysis['inputs']['campaign008_report']['sha256']
            and original['runtime_manifest'] == report['runtime_manifest']
            and original['reset_root_positions_m'] == report['reset_root_positions_m'],
            'Campaign008 source report or exact runtime/placement comparison differs')
    require(all(original['windows'][phase] == report['windows'][phase] for phase in ('startup', 'settled')),
            'Standing metric reproduction differs')
    require(all(row['minimum_rad'] == original['individual_motor_positive_minus_negative_rad'][name]
                for name, row in report['individual_motor_response_by_env'].items())
            and len(report['individual_motor_response_by_env']) == 15,
            'One of the 15 recorded direction minima differs from campaign008')
    require(all(original['windows']['driven'][key] == report['windows']['driven'][key]
                for key in ('max_closure_point_m', 'max_demand_nm', 'max_passive_velocity_relation_error_rad_s',
                            'max_closure_relative_point_velocity_m_s')),
            'Recorded motion-failure peak reproduction differs')

    checked = 0
    if raw_dir is not None:
        raw_dir = Path(raw_dir).resolve()
        for name in raw_names:
            path = raw_dir/name
            require(path.resolve().is_relative_to(raw_dir), 'Raw file escapes supplied directory')
            require(path.stat().st_size == records[name]['bytes'] and digest(path) == records[name]['sha256'],
                    'Raw NPZ bytes differ: ' + name)
            checked += 1
    return {'schema': 'hexapod.prefix_replay_evidence_verification.v1',
            'compact_local_files_verified': len(manifest), 'recorded_remote_files': len(records),
            'recorded_remote_npz_files': len(raw_names),
            'recorded_remote_npz_bytes': sum(records[name]['bytes'] for name in raw_names),
            'raw_npz_files_rehashed_this_invocation': checked,
            'raw_npz_bytes_rehashed_this_invocation': sum(records[name]['bytes'] for name in raw_names) if checked else 0,
            'source_commit': supervisor['source_commit'], 'functional_sha256': identity,
            'diagnostic_complete': True, 'primary_pass': False, 'physical_gate_errors': report['physical_gate_errors'],
            'exact_runtime_placements_standing_windows_15_response_minima_and_four_failure_peaks_reproduced': True,
            'simulation_training_admission': False, 'hardware_admission': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--raw-dir', type=Path, help='Retained run directory or downloaded NPZ directory; reads all 37 raw files')
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.raw_dir), indent=2, sort_keys=True))
