from pathlib import Path
import hashlib, json, subprocess, sys
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def verify(root, manifest):
    m = read(root/manifest)
    assert not any(p.is_symlink() for p in root.rglob('*'))
    actual = {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p != root/manifest}
    assert actual == m, (str(root), 'inventory mismatch')
    return len(m)
count = verify(R, 'BUNDLE_SHA256.json')
assert sha(R/'curated/FREEZE_SHA256.json') == '3bb38f020885c77b030dcce69b292c5ee61891686c0755cce3e586b292ad14fa'
subprocess.run([sys.executable, '-B', '-S', str(R/'curated/verify_bundle.py')], check=True)
assert sha(R/'analyzer/FREEZE_SHA256.json') == 'be6625b977aa8ee333ecf1aa744bed99ba16d991dca0f3679c53414bbdc8bd94'
verify(R/'analyzer', 'FREEZE_SHA256.json')
A = R/'cpu_analysis/remote'
receipt = read(A/'execution_receipt.json')
assert receipt['execution_receipt_passed'] and receipt['analyzer_evidence_verified'] and receipt['analyzer_errors'] == []
for name, bound in receipt['analysis_outputs_sha256'].items(): assert sha(A/'analysis'/name) == bound
assert receipt['complete_raw_trees_unchanged'] and receipt['analyzer_unchanged'] and receipt['terminal_audit_unchanged']
before, after = read(A/'remote_inputs_before.json'), read(A/'remote_inputs_after.json')
assert before == after
audit_path = R/'curated/audit/remote_terminal_audit_001.json'
assert sha(audit_path) == 'd337846ef9d983e7a17a17a992367a6248cf8cad67adf476958b1e152f271e7a'
audit = read(audit_path)
assert audit['passed'] and not audit['errors'] and all(audit['checks'].values())
expected = {k.removeprefix('run/'):v for k,v in audit['inventory'].items() if k.startswith('run/')}
assert before['campaign'] == expected
assert receipt['campaign_sha256'] == audit['campaign']['campaign_sha256']
assert audit['campaign']['updates'] == 500 and audit['campaign']['actual_transitions'] == 12288000
assert len(audit['ordinary_autosaves']) == 500 and len(audit['owned_absence']) == 12
assert sha(R/'curated/raw/run/train/policy/final.pt') == 'c376a0a4eb04d54396b4fd6171fe173167767463245213cce7c2cc1d3a2877cf'
assert len(audit['campaign']['quiet_failed_env_ids']) == 48
print(json.dumps({'passed':True, 'payloads':count, 'remote_analysis_receipt_verified':True,
    'local_full_training_replay':False, 'remote_live_probe_performed':False, 'Stage2_complete':False}, indent=2))
