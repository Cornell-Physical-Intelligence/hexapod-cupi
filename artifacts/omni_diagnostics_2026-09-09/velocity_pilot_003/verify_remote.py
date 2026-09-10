"""Read-only source/result hash audit. Never takes locks or signals jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SOURCE = BASE / 'omni_velocity_source_003'
MODE = sys.argv[1]
assert MODE in ('probe', 'pilot')
OUTPUT = BASE / f'omni_velocity_{MODE}_003'
PAUSE = BASE / ('forecast_pause_021' if MODE == 'probe' else 'forecast_pause_022')
PREP = BASE / ('omni_velocity_launch_preparation_003' if MODE == 'probe' else 'omni_velocity_pilot_preparation_003')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not str(p.relative_to(root)).startswith('inputs/study/')}

def command(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'command': args, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}

campaign = json.loads((OUTPUT / 'campaign.json').read_text())
assert campaign['status'] in ('completed', 'completed_needs_review', 'rejected', 'failed'), campaign['status']
assert (PAUSE / 'restored.json').exists(), 'Wait for owned restoration before freezing evidence'
manifest = json.loads((SOURCE / 'campaign_source_hashes.json').read_text())
assert digest(SOURCE / 'campaign_source_hashes.json') == '00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25'
changed = [p for p, h in manifest.items() if not (SOURCE / p).is_file() or digest(SOURCE / p) != h]
extra = sorted(str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*')
               if p.is_file() and str(p.relative_to(SOURCE)) not in manifest
               and str(p.relative_to(SOURCE)) != 'campaign_source_hashes.json')
if MODE == 'probe':
    assets = json.loads((OUTPUT / 'inputs/study_after_flat.sha256.json').read_text())
    asset_root = OUTPUT / 'inputs/study'
else:
    assets = json.loads((OUTPUT / 'inputs_before.sha256.json').read_text())
    asset_root = OUTPUT / 'inputs'
    # Pilot's exact-input map contains the asset plus admission/calibration/runner receipts.
asset_changed = [p for p, h in assets.items() if not (asset_root / p).is_file() or digest(asset_root / p) != h]
jobs = []
for p in sorted((OUTPUT / 'jobs').glob('*.json')):
    job = json.loads(p.read_text())
    if not job.get('container_id'):
        continue
    by_id = command(['docker', 'inspect', job['container_id']])
    by_name = command(['docker', 'inspect', job['container_name']])
    jobs.append({'report_path': str(p.relative_to(OUTPUT)), 'phase': job.get('phase'),
                 'by_id': by_id, 'by_name': by_name})
result = {
    'observed_unix': time.time(), 'mode': MODE,
    'scope': 'Read-only post-terminal identity, exact-container absence and restoration audit; no locks acquired and no assumption that a later job is idle.',
    'source': str(SOURCE), 'output': str(OUTPUT),
    'source_manifest_sha256': digest(SOURCE / 'campaign_source_hashes.json'),
    'source_files_verified': len(manifest), 'source_changed': changed, 'source_extra_files': extra,
    'admitted_inputs_root': str(asset_root), 'admitted_input_files_verified': len(assets),
    'admitted_inputs_changed': asset_changed,
    'unit': command(['systemctl', '--user', 'show', f'hexapod-omni-velocity-{MODE}-003-20260910.service',
                     '--property=ActiveState,SubState,Result,ExecMainStatus']),
    'owned_jobs': jobs,
    'owned_containers_absent': all(j[k]['returncode'] != 0 and 'no such' in j[k]['stderr'].lower()
                                   for j in jobs for k in ('by_id', 'by_name')),
    'run_files_sha256': hashes(OUTPUT), 'pause_files_sha256': hashes(PAUSE),
    'preparation_files_sha256': hashes(PREP),
    'restoration': json.loads((PAUSE / 'restored.json').read_text()),
    'source_origin_sha256': digest(SOURCE / 'source_origin.json'),
}
if MODE == 'probe':
    result['real_rsl_cpu_files_sha256'] = hashes(BASE / 'omni_velocity_cpu_003')
result['integrity_passed'] = not changed and not extra and not asset_changed
print(json.dumps(result, indent=2))
