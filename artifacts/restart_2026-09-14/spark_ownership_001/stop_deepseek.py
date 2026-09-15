"""One exact, user-authorized workload deferral; private backups stay on Spark."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001')
UNIT = 'dsv41-inference.service'
UNIT_PATH = Path('/home/orionh/.config/systemd/user') / UNIT

def run(*args, check=True):
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if check and p.returncode:
        raise RuntimeError(f'{args[0]} exited {p.returncode}: {p.stderr[:500]}')
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def show():
    fields = ('Id', 'ActiveState', 'SubState', 'MainPID', 'ControlGroup', 'Restart',
              'FragmentPath', 'DropInPaths', 'UnitFileState', 'TriggeredBy')
    result = run('systemctl', '--user', 'show', UNIT, *['--property='+f for f in fields])
    return dict(line.split('=', 1) for line in result['stdout'].splitlines() if '=' in line)

def save(name, value):
    path = ROOT / name
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    os.chmod(path, 0o600)

ROOT.mkdir(parents=True, exist_ok=False, mode=0o700)
assert UNIT_PATH.is_file() and not UNIT_PATH.is_symlink(), 'Unit identity changed'
before = show()
assert before['FragmentPath'] == str(UNIT_PATH)
assert before['DropInPaths'] == '' and before['Id'] == UNIT
unit_bytes = UNIT_PATH.read_bytes()
unit_sha = hashlib.sha256(unit_bytes).hexdigest()
backup = ROOT / 'dsv41-inference.service.original'
backup.write_bytes(unit_bytes)
os.chmod(backup, UNIT_PATH.stat().st_mode & 0o777)
coord = Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
(ROOT / 'SPARK_COMPUTE_COORDINATION.before.md').write_bytes(coord.read_bytes())
receipt = {
    'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'authorization': ['you can stop all other processes on the spark',
                      'take full ownership of the spark, you have my permissions'],
    'scope': 'Exclusive HEXAPOD user compute; preserve SSH, OS services, outputs and exact restart state',
    'unit_before': before, 'unit_sha256': unit_sha,
    'unit_backup': str(backup), 'unit_original_mode': oct(UNIT_PATH.stat().st_mode & 0o777),
    'gpu_before': run('nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'),
}
save('before.json', receipt)
receipt['stop'] = run('systemctl', '--user', 'stop', UNIT)
after_stop = show()
assert after_stop['ActiveState'] == 'inactive' and after_stop['MainPID'] == '0', after_stop
assert not UNIT_PATH.is_symlink() and hashlib.sha256(UNIT_PATH.read_bytes()).hexdigest() == unit_sha
# Atomic rename keeps the exact original; the persistent mask rejects restarts.
UNIT_PATH.rename(ROOT / 'dsv41-inference.service.removed')
UNIT_PATH.symlink_to('/dev/null')
receipt['daemon_reload'] = run('systemctl', '--user', 'daemon-reload')
receipt['unit_after'] = show()
assert receipt['unit_after']['UnitFileState'] == 'masked'
assert receipt['unit_after']['ActiveState'] == 'inactive' and receipt['unit_after']['MainPID'] == '0'
receipt['mask_target'] = os.readlink(UNIT_PATH)
receipt['gpu_after'] = run('nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader')
receipt['memory_after'] = run('free', '-m')
receipt['locks_after'] = run('lslocks', '--output', 'PID,COMMAND,PATH')
receipt['old_pids_absent'] = {pid: not Path('/proc/'+pid).exists() for pid in ('2521668','2521678')}
assert all(receipt['old_pids_absent'].values()), 'An identified descendant remains'
receipt['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt['native_run_started'] = False
receipt['restoration'] = 'Only on later explicit release: verify current mask target and backup SHA; restore exact original unit and recorded prior active state. No automatic restart.'
save('receipt.json', receipt)
print(json.dumps(receipt, indent=2))
