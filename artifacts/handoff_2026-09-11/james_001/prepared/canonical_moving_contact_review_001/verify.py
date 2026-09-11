"""Portable standard-library immutable review verification; no admission grant."""
from pathlib import Path
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'FREEZE_SHA256.json').read_text())
    expected = manifest['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name != 'FREEZE_SHA256.json'}
    if actual != set(expected):
        raise ValueError('Changed exact payload inventory')
    for name, wanted in expected.items():
        if digest(root / name) != wanted:
            raise ValueError('Changed payload: ' + name)
    inputs = json.loads((root / 'INPUTS.json').read_text())['files']
    for name, record in inputs.items():
        path = root / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Changed input snapshot: ' + name)
    report = json.loads((root / 'REPORT.json').read_text())
    if report['runtime_adoption'] or report['physical_admission'] or report['stage2_pass']:
        raise ValueError('Review is not native or formal admission')
    rsl = json.loads((root / 'inputs/RSL_SOURCE_SHA256.json').read_text())
    if rsl['algorithms/ppo.py'] != digest(root / 'inputs/RSL_ppo.py'):
        raise ValueError('Replay source differs from actual PPO source binding')
    parent = json.loads((root / 'inputs/MOVING_FREEZE_SHA256.json').read_text())['files']
    if parent['contact_contract.py'] != digest(root / 'PARENT_contact_contract.py'):
        raise ValueError('Parent contact facts source differs')
    final = json.loads((root / 'fable/fable_final.json').read_text())
    launch = json.loads((root / 'fable/FABLE_LAUNCH.json').read_text())
    if final.get('is_error') or final['session_id'] != launch['session_id']:
        raise ValueError('Partner consultation not a successful matching session')
    if digest(root / 'fable/prompt.md') != launch['prompt_sha256']:
        raise ValueError('Changed partner prompt')
    print(json.dumps({'status': 'verified', 'payloads': len(expected),
                      'input_snapshots': len(inputs), 'tests_reported': 9,
                      'runtime_adoption': False, 'physical_admission': False,
                      'stage2_pass': False}, sort_keys=True))


if __name__ == '__main__':
    main()
