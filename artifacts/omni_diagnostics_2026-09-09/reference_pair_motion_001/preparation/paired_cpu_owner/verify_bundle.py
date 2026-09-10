"""Read-only portable payload, final source identity and history verification."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    frozen=json.loads((H/'FREEZE_SHA256.json').read_text())
    files=frozen['files']
    for rel,wanted in files.items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts,rel
        assert sha(H/rel)==wanted,rel
    report=json.loads((H/'report_002.json').read_text())
    for rel,wanted in report['runtime_sources_sha256'].items():assert sha(H/rel)==wanted,rel
    for rel,wanted in report['geometry_and_residual_sources']['oracle_files'].items():assert sha(H/rel)==wanted,rel
    assert sha(H/'tests_002.log')==report['tests']['log_sha256']
    assert sha(H/'tests_initial.log')==report['tests']['initial_failed_expectation_log_sha256']
    old=json.loads((H/'report.json').read_text())
    history=json.loads((H/'history_before_handoff_guard/SHA256.json').read_text())
    for rel,wanted in history.items():
        assert sha(H/'history_before_handoff_guard'/rel)==wanted,rel
        assert old['runtime_sources_sha256'][rel]==wanted,rel
    assert report['cases'][0]['controls']==2200 and report['cases'][0]['failure'] is None
    assert all(c['failure'] is not None for c in report['cases'][1:])
    assert not report['new_moving_pair_physics_admitted'] and not report['PPO_ready']
    print(json.dumps({'verified_payloads':len(files),'final_runtime_identity':True,'historical_runtime_identity':True,'scope':'CPU-only prototype, no physics admission'},indent=2))
if __name__=='__main__':main()
