"""Portable local verification only: no native execution, network or raw reconstruction."""
import hashlib
import json
from pathlib import Path
from summarize_analysis import summarize

ROOT = Path(__file__).resolve().parent
ANALYZER = 'd1eb068b71ccf01bae0fe26d16f8f24e83be260c5ca4556ec75ba1d0d678faab'
RESULT = '304bccb643f1be8ab95a41d98c85c985e7248652a208e20ac51b71006ff60231'
STANDING = 'c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
AUDIT = '8834ba3d90b713266f4a2dc270e07a7977ca154376862148fac8329147bad4cb'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def inventory(root):
    require(not any(p.is_symlink() for p in root.rglob('*')), 'Symlink in bundle')
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}


def main():
    outer = read('BUNDLE_SHA256.json')['files']
    require(inventory(ROOT) - {'BUNDLE_SHA256.json'} == set(outer), 'Outer inventory mismatch')
    for rel, digest in outer.items():
        require(sha(ROOT / rel) == digest, 'Changed payload: ' + rel)
    source = read('source/FREEZE_SHA256.json')
    require(len(source) == 15 and sha(ROOT / 'source/FREEZE_SHA256.json') == ANALYZER, 'Analyzer identity')
    require(inventory(ROOT / 'source') == set(source) | {'FREEZE_SHA256.json'}, 'Complete source inventory')
    for rel, digest in source.items():
        require(sha(ROOT / 'source' / rel) == digest, 'Source payload: ' + rel)
    copies = read('COPY_VERIFICATION.json')
    require(len(copies['selected_copies']) == 24 and copies['no_original_raw_copied'] is True, 'Copy selection')
    for row in copies['selected_copies']:
        p = ROOT / row['public_copy']
        require(sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'Copy mismatch')
    a = read('actual/analysis.json'); audit = read('source/inputs/audit.json')
    require(sha(ROOT / 'actual/analysis.json') == RESULT and (ROOT / 'actual/analysis.json').stat().st_size == 22361724, 'Actual result identity')
    require(sha(ROOT / 'source/inputs/audit.json') == AUDIT == a['audit_sha256'], 'Actual audit binding')
    require(sha(ROOT / 'source/inputs/SOURCE005_FREEZE_SHA256.json') == STANDING == a['source_freeze_sha256'], 'Standing source binding')
    require(len(read('source/inputs/SOURCE005_FREEZE_SHA256.json')) == 109 == a['source_payloads'], 'Standing source count')
    require(a['analyzer_sha256'] == sha(ROOT / 'source/analyze_remote.py'), 'Analyzer runtime binding')
    require(a['numeric_reader_sha256'] == sha(ROOT / 'source/numeric_evidence.py'), 'Reader runtime binding')
    require(a['CPU_only'] is True and a['read_only'] is True and a['raw_and_source_reverified_after_analysis'] is True, 'Read-only replay scope')
    require(a['native_report_unchanged'] == audit['standing_report'], 'Original report changed')
    require(a['original_standing_pass'] is False and a['original_support_counts_exact'] is True, 'Original rejection/counts')
    require(audit['raw_payloads'] == 39 and audit['raw_total_bytes'] == 5019294627, 'Full remote inventory')
    require(len(a['consumed_raw_inputs']) == 18, 'Consumed selection')
    for rel, value in a['consumed_raw_inputs'].items():
        require(value == audit['raw_inventory']['run/standing/' + rel], 'Consumed input binding: ' + rel)
    require(a['exact_pre_post_angle_scalar_comparisons'] == 3686400 and a['patch_FP32_product_FP64_sum_exact_rows_all32'] == 360, 'Replay counts')
    require(len(a['events']) == 73 and len(a['used_slot_count_all8000_rows']) == 8000, 'Event/slot rows')
    execution = read('actual/execution.json')
    require(execution['returncode'] == 0 and execution['error'] is None and execution['source_freeze_sha256'] == ANALYZER, 'Successful CPU execution')
    for rel, value in execution['files'].items():
        p = ROOT / 'actual' / rel
        require(sha(p) == value['sha256'] and p.stat().st_size == value['size_bytes'], 'Execution result file')
    transfer = read('actual/transfer.json'); transferred = json.loads(transfer['stdout'])
    require(transfer['returncode'] == 0 and transferred['verified_files'] == 16 and transferred['manifest_sha256'] == ANALYZER, 'Source transfer')
    s = summarize(a)
    require(s == read('SUMMARY.json'), 'Compact extraction mismatch')
    r = s['original_admission_unchanged']
    require((r['combined_pass'], r['physical_pass'], r['quiet_pass']) == (10, 10, 24), 'Original admission counts')
    require(r['original_quiet_window_s'] == [16.0] and r['original_quiet_samples_50Hz'] == [800], 'Original quiet window')
    require(s['diagnostic_rates_same_16s_400Hz']['joint_environment_pairs'] == 576, 'Rate pair count')
    final = read('actual/final_reservation_readback.json')
    require(final['verified'] is True and final['native_calls'] == 0 and final['mutations'] == 0 and final['gpu_compute_processes'] == [], 'Final read-only observation')
    require(final['reservation']['masked_user_units'] == 31 and final['reservation']['queue_lock_held'] is True and final['reservation']['release_attempted'] is False, 'Final reservation observation')
    print(json.dumps({'verified': True, 'payloads': len(outer), 'payload_bytes': sum((ROOT / p).stat().st_size for p in outer), 'bundle_sha256': sha(ROOT / 'BUNDLE_SHA256.json'), 'complete_analyzer_payloads': 15, 'actual_result_sha256': RESULT, 'remote_original_raw_files': 39, 'local_original_raw_files': 0, 'original_report_preserved': True, 'original_all_pass': False}, sort_keys=True))


if __name__ == '__main__':
    main()
