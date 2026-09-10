"""Verify published diagnostics and reproduce compact interpretation; no native/raw replay."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(p.read_text())

def inventory(p):
    require(not p.is_symlink(), 'Symbolic root')
    result = {}
    for f in sorted(p.rglob('*')):
        require(not f.is_symlink(), 'Symbolic payload')
        if f.is_file(): result[f.relative_to(p).as_posix()] = sha(f)
    return result

def verify(root=ROOT, expected=None):
    manifest = root/'BUNDLE_SHA256.json'
    if expected: require(sha(manifest) == expected, 'Wrong bundle identity')
    actual = inventory(root); actual.pop('BUNDLE_SHA256.json')
    require(actual == read(manifest), 'Missing/changed/extra publication payload')
    provenance = read(root/'PROVENANCE.json')
    require(provenance['full_raw_replay_available'] is False and provenance['publication_native_or_remote_execution'] is False, 'Scope changed')
    for name, row in provenance['components'].items():
        mapping = inventory(root/name)
        if 'freeze_sha256' in row:
            require(mapping.pop('FREEZE_SHA256.json') == row['freeze_sha256'], 'Changed '+name+' freeze')
            require(mapping == read(root/name/'FREEZE_SHA256.json') and len(mapping) == row['payloads'], 'Changed '+name+' files')
        else: require(mapping == row['snapshot'], 'Changed root snapshot')
    for name, row in provenance['metadata_inputs'].items():
        p = root/'inputs'/name
        require(sha(p) == row['sha256'] and p.stat().st_size == row['size_bytes'], 'Changed compact input')
    audit = read(root/'analyzer002/original_audit.json')
    require(sha(root/'analyzer001/original_audit.json') == sha(root/'analyzer002/original_audit.json'), 'Analyzer audit mismatch')
    require(audit['audit_verified'] is True and audit['errors'] == [] and audit['raw_acquisition_completed'] is True and audit['standing_completed'] is False, 'Terminal semantics changed')
    full = read(root/'inputs/FULL32_RAW_SHA256.json')
    require(full == audit['raw_inventory'] and len(full) == 36 and sum(v['size_bytes'] for v in full.values()) == 4969155341, 'Full remote inventory changed')
    selection = read(root/'inputs/RAW32_SELECTION.json')
    require(selection['self_contained_raw_replay'] is False and selection['full_raw_files'] == 36 and selection['remote_only_files'] == 12, 'Raw availability changed')
    previous = read(root/'inputs/REJECTED32_BUNDLE_SHA256.json')
    require(previous['terminal/audit.json'] == sha(root/'analyzer002/original_audit.json'), 'Terminal publication binding')
    require(previous['REMOTE_RAW_INVENTORY.json'] == sha(root/'inputs/FULL32_RAW_SHA256.json'), 'Raw map publication binding')
    require(previous['source/FREEZE_SHA256.json'] == sha(root/'inputs/SOURCE003_FREEZE_SHA256.json'), 'Source003 publication binding')
    require((root/'actual001/analysis.json').stat().st_size == 0, 'Invented initial result')
    require('ValueError: Raw patch aggregate differs from NPZ' in (root/'actual001/analysis.stderr').read_text(), 'Initial precision failure missing')
    require(sha(root/'actual001/analysis.stderr') == sha(root/'analyzer002/failure_001/actual_analysis.stderr'), 'Failure receipt changed')
    require(sha(root/'analyzer001/analyze_remote.py') == sha(root/'analyzer002/failure_001/analyze_remote.py'), 'Old analyzer not preserved')
    require(sha(root/'analyzer001/FREEZE_SHA256.json') == sha(root/'analyzer002/PARENT_FREEZE_SHA256.json'), 'Parent source binding')
    require((root/'actual002/analysis.stderr').stat().st_size == 0, 'Actual002 stderr changed')
    analysis = read(root/'actual002/analysis.json')
    require(analysis['read_only'] is True and analysis['CPU_only'] is True and analysis['used_raw_reverified_after_analysis'] is True, 'Analysis scope/integrity changed')
    require(analysis['self_sha256'] == sha(root/'analyzer002/analyze_remote.py') and analysis['helper_sha256'] == sha(root/'analyzer002/numeric_evidence.py'), 'Actual analyzer source mismatch')
    require(analysis['original_audit_sha256'] == sha(root/'analyzer002/original_audit.json') and analysis['source_freeze_sha256'] == sha(root/'inputs/SOURCE003_FREEZE_SHA256.json'), 'Actual source/audit binding')
    require(analysis['native_standing_pass'] is False and analysis['host_error'] == audit['campaign']['error'], 'Actual rejection changed')
    for name, row in analysis['all_raw_inputs_used'].items():
        require(row == full['run/standing/'+name], 'Consumed32 raw pin mismatch')
    single = read(root/'inputs/SINGLE_RAW_SHA256.json')
    for name, row in analysis['single_inputs_used'].items():
        require(row == single['run/standing/'+name], 'Consumed single raw pin mismatch')
    require(analysis['contact_stream_original_sha256'] == full['run/standing/contacts.jsonl']['sha256'], 'Contact stream binding')
    mismatch = analysis['first_analyzer001_unrounded_product_mismatch']
    require(mismatch['corrected_exact_match'] is True and mismatch['corrected_aggregate'] == mismatch['stored_NPZ_aggregate'] and mismatch['old_aggregate'] != mismatch['stored_NPZ_aggregate'], 'Precision correction evidence changed')
    require(mismatch['source_product_dtype'] == 'float32' and mismatch['source_accumulator_dtype'] == 'float64' and mismatch['old_analyzer_product_dtype'] == 'float64', 'Arithmetic distinction changed')
    report = read(root/'interpretation/REPORT.json')
    require(report['analysis_sha256'] == sha(root/'actual002/analysis.json') and report['audit_sha256'] == sha(root/'analyzer002/original_audit.json'), 'Interpretation actual binding')
    # The frozen summarizer is a standard-library-only file with one explicit input.
    # Run in memory after verifying its exact owner manifest; it performs no raw/native I/O.
    argv = sys.argv[:]; capture = io.StringIO()
    try:
        sys.argv = [str(root/'interpretation/summarize.py'), str(root/'actual002/analysis.json')]
        with redirect_stdout(capture): runpy.run_path(str(root/'interpretation/summarize.py'), run_name='__main__')
    finally: sys.argv = argv
    require(json.loads(capture.getvalue()) == report, 'Compact interpretation failed exact reproduction')
    require(report['events'] == 74 and len(report['pass_envs']) == 11 and report['event_summary']['exact128_inactive_zero_patches'] == 74, 'Event counts changed')
    require(report['event_summary']['all_six_forces_zero'] == 0 and report['event_summary']['affected_leg_all_sdk_zero'] == 0, 'Hypothesis counterexample changed')
    references = read(root/'interpretation/REFERENCED_INPUTS.json')
    relocation = {'tmp/canonical_native_standing32_terminal_002/audit.json':'analyzer002/original_audit.json',
                  'tmp/canonical_standing32_failure_analysis_002/FREEZE_SHA256.json':'analyzer002/FREEZE_SHA256.json',
                  'tmp/canonical_standing32_failure_analysis_root_001/analysis.stderr':'actual001/analysis.stderr',
                  'tmp/canonical_standing32_failure_analysis_root_002/analysis.json':'actual002/analysis.json',
                  'tmp/canonical_standing32_failure_analysis_root_002/analysis.stderr':'actual002/analysis.stderr',
                  'tmp/canonical_standing32_failure_analysis_root_002/transfer.json':'actual002/transfer.json'}
    require(set(references) == set(relocation), 'Missing interpretation reference')
    for name, row in references.items():
        p = root/relocation[name]
        require(sha(p) == row['sha256'] and p.stat().st_size == row['size_bytes'], 'Interpretation reference mismatch')
    physics = read(root/'physics_review/report.json')
    require(physics['one_interval_reconstruction_all8000'] is True and physics['one_servo_replay_all8000'] is True, 'Single-origin receipt changed')
    require((physics['pass_count'],physics['support_failed_count'],physics['SDK_quiet_failed_count']) == (11,21,7), 'Independent recorded counts changed')
    require(physics['translation_float32_classifier_test']['comparisons'] == 32768 and physics['translation_float32_classifier_test']['category_flips'] == 0, 'Rounding study changed')
    known_metadata = {sha(root/'inputs/SOURCE003_FREEZE_SHA256.json'),sha(root/'inputs/SINGLE_RAW_SHA256.json'),sha(root/'analyzer002/original_audit.json')}
    known_metadata.update(v['sha256'] for v in single.values()); known_metadata.update(read(root/'inputs/SINGLE_BUNDLE_SHA256.json').values())
    require(all(h in known_metadata for h in physics['inputs'].values()), 'Independent review input unbound to published metadata')
    disposition = read(root/'fable_review/DISPOSITION.json'); launch = read(root/'fable_review/FABLE_LAUNCH.json'); final = read(root/'fable_review/fable_final.json')
    require(disposition['prompt_sha256'] == sha(root/'fable_review/prompt.md') == launch['prompt_sha256'] and disposition['final_sha256'] == sha(root/'fable_review/fable_final.json'), 'Partner bytes changed')
    require(disposition['model'] == launch['model'] == 'claude-fable-5-1' and disposition['effort'] == launch['effort'] == 'max', 'Partner selection changed')
    require(disposition['session_id'] == launch['session_id'] == final['session_id'] and final['is_error'] is False and read(root/'fable_review/FABLE_EXIT.json')['returncode'] == 0, 'Partner result mismatch')
    require(launch['tools_disabled'] is True and launch['MCP_disabled'] is True, 'Partner tool scope changed')
    require(disposition['source004_freeze_sha256'] == sha(root/'inputs/SOURCE004_FREEZE_SHA256.json'), 'Future recipe source mismatch')
    require(disposition['independent_source_review_freeze_sha256'] == sha(root/'solver_source_review/FREEZE_SHA256.json') and disposition['independent_physics_review_freeze_sha256'] == sha(root/'physics_review/FREEZE_SHA256.json'), 'Partner independent review binding')
    require(disposition['actual32_audit_sha256'] == sha(root/'analyzer002/original_audit.json'), 'Partner actual binding')
    actual_max = max(r['quiet']['max_joint_velocity_rms_rad_s'] for r in audit['standing_report']['replicas'])
    require(disposition['actual_max_joint_velocity_rms_50hz_rad_s'] == actual_max == 0.09956584870815277, 'Corrected actual maximum mismatch')
    return {'verified':True,'bundle_sha256':sha(manifest),'payloads':len(actual),
            'initial_precision_failure_preserved':True,'actual_analyzer_source_bound':True,
            'compact_interpretation_exactly_reproduced':True,'support_events':74,
            'full_raw_inventory_files':36,'full_raw_inventory_bytes':4969155341,
            'full_raw_replayed_locally':False,'native_or_remote_execution':False,
            'physical_admission':False,'training_allowed':False}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-bundle');args=p.parse_args()
    print(json.dumps(verify(expected=args.expected_bundle),indent=2,sort_keys=True))
