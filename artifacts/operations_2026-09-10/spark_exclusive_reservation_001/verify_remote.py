from pathlib import Path
import hashlib, json, subprocess, time

base = Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001')
paths = [base / p for p in ['ACTIVE', 'scheduler_before.json', 'initial_receipt.json', 'ollama_before.json', 'ollama_deferral.json']]
units = [f'stormscope-{name}.{kind}' for name in ['dispatch', 'scout', 'monitor', 'publish', 'verify'] for kind in ['timer', 'service']] + ['ollama.service']
paths += [Path('/home/orionh/.config/systemd/user') / (u + '.d') / '90-hexapod-exclusive-reservation-20260910.conf' for u in units]
coord = Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
assert hashlib.sha256(coord.read_bytes()).hexdigest() == '649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f'
assert json.loads((base / 'ACTIVE').read_text())['exclusive'] is True
states = {}
for unit in units:
    states[unit] = dict(line.split('=', 1) for line in subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ConditionResult', '-p', 'DropInPaths', '-p', 'NeedDaemonReload'], text=True).splitlines())
    assert states[unit]['ActiveState'] in ['inactive', 'failed']
    assert '90-hexapod-exclusive-reservation-20260910.conf' in states[unit]['DropInPaths']
    assert states[unit]['NeedDaemonReload'] == 'no'
for p in paths[5:]:
    assert 'ConditionPathExists=!' + str(base / 'ACTIVE') in p.read_text()
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
print(json.dumps({'verified': True, 'checked_unix': time.time(), 'units': states,
    'files': {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'text': p.read_text()} for p in paths},
    'coordination_sha256': hashlib.sha256(coord.read_bytes()).hexdigest(), 'cuda_processes_raw': gpu,
    'manual_launch_hardware_block_claimed': False, 'mutation_performed': False}))
