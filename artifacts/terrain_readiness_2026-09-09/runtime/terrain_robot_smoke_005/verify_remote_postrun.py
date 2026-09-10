"""Read-only historical source, asset, cleanup and live ownership snapshot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SOURCE = BASE / 'terrain_robot_source_005'
OUTPUT = BASE / 'terrain_robot_smoke_005'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'command': args, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}

manifest = json.loads((SOURCE / 'campaign_source_hashes.json').read_text())
before = json.loads((OUTPUT / 'inputs/study_before_flat.sha256.json').read_text())
after = json.loads((OUTPUT / 'inputs/study_after_flat.sha256.json').read_text())
source_changed = [p for p, h in manifest.items() if not (SOURCE / p).is_file() or sha(SOURCE / p) != h]
asset_changed = [p for p, h in after.items() if not (OUTPUT / 'inputs/study' / p).is_file() or sha(OUTPUT / 'inputs/study' / p) != h]
source_extras = sorted(str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*')
                       if p.is_file() and str(p.relative_to(SOURCE)) not in manifest
                       and p.name != 'campaign_source_hashes.json')
owned = []
for phase in ('flat', 'terrain'):
    job = json.loads((OUTPUT / f'jobs/{phase}.json').read_text())
    owned.append({'phase': phase, 'job': job,
                  'inspect_by_id': command(['docker', 'inspect', job['container_id']]),
                  'inspect_by_name': command(['docker', 'inspect', job['container_name']])})

unit = command(['systemctl', '--user', 'show',
                'hexapod-terrain-robot-smoke-005-20260910.service',
                '--property=ActiveState,SubState,Result,ExecMainStatus'])
campaign = json.loads((OUTPUT / 'campaign.json').read_text())
assert campaign['status'] == 'completed' and campaign['terrain_standing_passed']
assert 'ActiveState=inactive' in unit['stdout']
locks = {}
for filename in ('/opt/wx/gpu.lock', '/tmp/hexapod-isaac-gpu.lock'):
    descriptor = os.open(filename, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks[filename] = 'free_at_snapshot_released_immediately'
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except BlockingIOError:
            locks[filename] = 'held_at_snapshot_owner_not_assumed'
    finally:
        os.close(descriptor)

result = {
    'observed_unix': time.time(),
    'scope': 'Post-terminal historical terrain005 verification and separate instantaneous live ownership snapshot; no job was signaled or launched.',
    'source': str(SOURCE), 'output': str(OUTPUT),
    'source_manifest_sha256': sha(SOURCE / 'campaign_source_hashes.json'),
    'source_files_verified': len(manifest), 'source_changed': source_changed,
    'source_extra_files': source_extras,
    'admitted_asset_files_verified': len(after), 'asset_changed': asset_changed,
    'before_after_flat_asset_maps_equal': before == after,
    'owned_jobs': owned, 'unit': unit,
    'forecast_restoration': json.loads((BASE / 'forecast_pause_020/restored.json').read_text()),
    'forecast_timer_snapshot': command(['systemctl', '--user', 'show',
         'stormscope-dispatch.timer', 'stormscope-scout.timer', '--property=Id,ActiveState,SubState']),
    'cuda_snapshot': command(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']),
    'docker_snapshot': command(['docker', 'ps', '--format', '{{.ID}} {{.Names}} {{.Image}}']),
    'lock_snapshot': locks,
    'run_files_sha256': {str(p.relative_to(OUTPUT)): sha(p) for p in sorted(OUTPUT.rglob('*'))
                         if p.is_file() and not str(p.relative_to(OUTPUT)).startswith('inputs/study/')},
    'pause_files_sha256': {str(p.relative_to(BASE / 'forecast_pause_020')): sha(p)
                          for p in sorted((BASE / 'forecast_pause_020').rglob('*')) if p.is_file()},
    'outer_guard_sha256': sha(BASE / 'terrain_robot_smoke_005_preparation/launch_guarded_remote.py'),
    'source_origin_sha256': sha(SOURCE / 'source_origin.json'),
}
result['integrity_passed'] = not source_changed and not source_extras and not asset_changed and before == after
result['owned_containers_absent'] = all(item[key]['returncode'] != 0 and 'no such' in item[key]['stderr'].lower()
                                      for item in owned for key in ('inspect_by_id', 'inspect_by_name'))
print(json.dumps(result, indent=2))
