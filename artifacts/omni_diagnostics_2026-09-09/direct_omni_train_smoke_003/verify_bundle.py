"""Portable local verification/replay of an immutable terminal publication."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
ANALYZER = 'e4db5c7acfec918931b2a444b338fb8bc51d575c6ab1f858e31edd2384edb098'
SOURCE = 'ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
CHECKPOINT = 'ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def relative(name):
    p = PurePosixPath(name)
    check(not p.is_absolute() and '..' not in p.parts, f'Invalid relative path: {name}')
    return name


def inventory(root, exclude=None):
    check(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),
          f'Symlink in {root}')
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*')
            if p.is_file() and p != exclude}


def verify_manifest(root, expected=None, count=None):
    p = root / 'FREEZE_SHA256.json'
    if expected:
        check(sha(p) == expected, f'Changed manifest: {root}')
    m = read(p)
    for name in m:
        relative(name)
    check(inventory(root, p) == m, f'Changed/incomplete/unlisted payloads: {root}')
    if count is not None:
        check(len(m) == count, f'Wrong payload count: {root}')
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true', help='Run unchanged analyzer with NumPy in temporary output')
    args = parser.parse_args()
    outer = verify_manifest(ROOT)
    verify_manifest(ROOT / 'analyzer', ANALYZER, 22)
    pins = read(ROOT / 'SOURCE_PINS.json')
    check(pins['source_manifest_sha256'] == SOURCE and pins['source_files'] == 599,
          'Wrong source pin')
    audit_path = ROOT / 'terminal/remote_terminal_audit.json'
    check(sha(audit_path) == pins['remote_terminal_audit_sha256'], 'Wrong terminal audit')
    audit = read(audit_path)
    raw = {}
    total = 0
    for directory in ('run', 'pause'):
        for name, bound in inventory(ROOT / 'terminal' / directory).items():
            key = directory + '/' + name
            path = ROOT / 'terminal' / key
            raw[key] = {'sha256': bound, 'bytes': path.stat().st_size}
            total += path.stat().st_size
    check(raw == audit['inventory'], 'Raw files differ from remote observation')
    check(len(raw) == audit['files'] == 52 and total == audit['bytes'] == 76234469,
          'Raw count/bytes differ')
    owner = dict(row.split('=', 1) for row in audit['owner']['stdout'].splitlines() if '=' in row)
    check(audit['owner']['exit_code'] == 0 and owner['MainPID'] == '0'
          and owner['ExecMainStatus'] == '0' and owner['ActiveState'] == 'inactive',
          'Remote owner was not terminal exit0')
    check(len(audit['cleanup']) == 8 and all(row['absent'] is True for row in audit['cleanup']),
          'Incomplete historical owned-container absence')
    check(read(ROOT / 'terminal/pause/restored.json') == audit['restored'],
          'Restoration receipt differs')
    check(audit['restored']['restored_unix'] > 0, 'Missing restoration time')
    for remote, bound in audit['pins'].items():
        head, tail = remote.split('/', 1)
        folder = {'direct_omni_train_smoke_003': 'run', 'forecast_pause_059': 'pause'}.get(head)
        check(folder is not None, 'Unexpected pinned remote scope')
        check(sha(ROOT / 'terminal' / folder / relative(tail)) == bound,
              f'Historical pin differs: {remote}')
    campaign = read(ROOT / 'terminal/run/campaign.json')
    check(campaign == audit['campaign'], 'Audited campaign differs')
    check(campaign['status'] == 'completed' and campaign['PPO_updates_completed'] == 2
          and campaign['bounded_campaign_complete'] is True
          and campaign['terminal_inputs_unchanged'] is True
          and campaign['Stage2_complete'] is False, 'Wrong bounded campaign result')
    check(campaign['identity']['source_manifest_sha256'] == SOURCE, 'Wrong actual source')
    check(sha(ROOT / 'terminal/run/train/policy/final.pt') == CHECKPOINT
          and pins['final_checkpoint_sha256'] == CHECKPOINT, 'Wrong final checkpoint')
    report = read(ROOT / 'analysis/report.json')
    check(report['evidence_verified'] is True and report['errors'] == []
          and report['reviewed_inputs_unchanged'] is True and report['Stage2_complete'] is False,
          'Analysis evidence failure')
    check(report['final_stop']['quiet_passed_replicas'] == 0
          and report['final_stop']['total_replicas'] == 48,
          'Unexpected original quiet result')
    check(report['training']['updates_completed'] == 2
          and report['training']['transitions'] == 1536
          and report['training']['strict_reload']['passed'] is True,
          'Unexpected integration result')
    optimization = report['optimizer_diagnostics']
    check(optimization['minibatches_retained'] == 40
          and optimization['sparse_gradient_rows_retained'] == 2
          and optimization['diagnostic_inventory_complete'] is True,
          'Missing optimizer diagnostics')
    original = read(ROOT / 'analysis/INPUTS_SHA256.json')
    mapping = read(ROOT / 'PORTABLE_INPUTS.json')['mapping']
    check(len(mapping) == len(original), 'Missing relocated analyzer input')
    check({row['original_absolute_path']: row['sha256'] for row in mapping} == original,
          'Relocated input map differs')
    for row in mapping:
        check(sha(ROOT / relative(row['bundle_relative_path'])) == row['sha256'],
              'Relocated input hash differs')
    local = read(ROOT / 'LOCAL_VERIFICATION.json')
    check(inventory(ROOT / 'terminal') == local['terminal_copy_map'],
          'Terminal auxiliary file changed')
    replay_result = 'not requested'
    if args.replay:
        with tempfile.TemporaryDirectory(prefix='hexapod_smoke003_replay_') as temporary:
            output = Path(temporary) / 'analysis'
            command = [sys.executable, '-B', str(ROOT / 'analyzer/analyze.py'),
                       '--campaign', str(ROOT / 'terminal/run'),
                       '--cold', str(ROOT / 'cold_baseline'), '--output', str(output)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(command, text=True, capture_output=True, env=env)
            check(result.returncode == 0, 'Portable analyzer failed: ' + result.stderr)
            check(read(output / 'report.json') == report, 'Portable numerical report differs')
            check((output / 'REPORT.md').read_bytes() == (ROOT / 'analysis/REPORT.md').read_bytes(),
                  'Portable human report differs')
            replay_result = 'exact report.json and REPORT.md match'
        verify_manifest(ROOT)
    print(json.dumps({'result': 'PASS', 'payloads': len(outer),
                      'manifest_sha256': sha(ROOT / 'FREEZE_SHA256.json'),
                      'raw_files': len(raw), 'raw_bytes': total,
                      'PPO_updates_completed': 2, 'quiet_passes': '0/48',
                      'portable_replay': replay_result, 'Stage2_complete': False}, indent=2))


if __name__ == '__main__':
    main()
