"""Read-only terminal inventory. No GPU, Docker, signal or filesystem mutation."""
import argparse, hashlib, json
from pathlib import Path

BASE = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
RUN = BASE / 'direct_omni_cold_001'
DISPATCH = {'unit': 'hexapod-direct-omni-cold-001-20260910.service',
            'invocation': 'ae3b95c756db4aa4b639ee4230e8c958', 'pause': 'forecast_pause_053',
            'source_manifest_sha256': '4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b',
            'host_freeze_sha256': 'efbeac2a0bfb7ebf93efb9ec4dce33363607092b30a7e81ac6351950e1fa6083',
            'checkpoint_sha256': '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def inventory(run):
    if run != RUN or run.is_symlink():
        raise ValueError('Only the exact cold001 output is permitted')
    campaign = json.loads((run / 'campaign.json').read_text())
    for key in ('source_manifest_sha256', 'checkpoint_sha256'):
        if campaign.get('identity', {}).get(key) != DISPATCH[key]: raise ValueError('Wrong campaign ' + key)
    if campaign.get('host_freeze_sha256') != DISPATCH['host_freeze_sha256']: raise ValueError('Wrong host identity')
    if campaign.get('status') not in ('completed', 'failed', 'stopped') or not campaign.get('finished_unix'):
        raise ValueError('Campaign is not terminal')
    jobs = {}
    for name in ('standing', 'baseline'):
        path = run / 'jobs' / (name + '.json')
        if path.exists():
            job = json.loads(path.read_text())
            if job.get('phase') != name or not job.get('finished_unix') or job.get('cleanup_checked') is not True:
                raise ValueError('Nonterminal or unaudited phase job: ' + name)
            jobs[name] = job
    if not jobs:
        raise ValueError('Missing actual job receipts')
    files = {}
    for path in sorted(run.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in raw outputs')
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'run': str(run), 'expected_dispatch': DISPATCH, 'campaign_status': campaign['status'], 'files': files,
            'bytes': sum(v['bytes'] for v in files.values()), 'actual_jobs': jobs,
            'scope': 'Terminal producer receipts and raw hashes only; root owns independent source/assets/absence/restoration audit.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=RUN)
    print(json.dumps(inventory(p.parse_args().run), indent=2, allow_nan=False))
