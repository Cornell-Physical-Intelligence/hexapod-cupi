from pathlib import Path
import hashlib, json, os, subprocess, time

base = Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001')
unit = 'ollama.service'
fragment = Path('/home/orionh/.config/systemd/user/ollama.service')
dropin = fragment.parent / (unit + '.d') / '90-hexapod-exclusive-reservation-20260910.conf'
assert (base / 'ACTIVE').is_file()
assert not dropin.exists()
assert not (base / 'ollama_deferral.json').exists()
def show():
    return dict(line.split('=', 1) for line in subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState',
         '-p', 'MainPID', '-p', 'InvocationID', '-p', 'FragmentPath', '-p', 'UnitFileState'], text=True).splitlines())
before = show()
assert before['FragmentPath'] == str(fragment)
assert before['InvocationID'] == '41e50638c7114e9c93e5ac1185726c42'
assert before['MainPID'] == '2491618'
assert before['ActiveState'] == 'active'
proc = Path('/proc') / before['MainPID']
exe = os.readlink(proc / 'exe')
assert exe == '/home/orionh/.config/Nvidia Corporation/Personal AI Router/engine-bin/ollama/bin/ollama'
start_ticks = (proc / 'stat').read_text().split(') ', 1)[1].split()[19]
cgroup = (proc / 'cgroup').read_text()
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
assert not gpu.strip(), 'Unexpected active CUDA workload; inspect before stopping inference service'
record = {'unit': unit, 'before': before, 'executable': exe, 'pid_start_ticks': start_ticks,
          'cgroup': cgroup, 'fragment_sha256': hashlib.sha256(fragment.read_bytes()).hexdigest(),
          'authorization': 'make sure our job takes full control of the spark at all times',
          'original_unit_modified': False, 'outputs_deleted': False, 'created_unix': time.time()}
(base / 'ollama_before.json').write_text(json.dumps(record, indent=2) + '\n')
dropin.parent.mkdir(exist_ok=True)
dropin.write_text('[Unit]\n# User-authorized exclusive HEXAPOD reservation.\nConditionPathExists=!' + str(base / 'ACTIVE') + '\n')
subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
assert show()['InvocationID'] == before['InvocationID']
assert (proc / 'stat').read_text().split(') ', 1)[1].split()[19] == start_ticks
subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
after = show()
assert after['ActiveState'] == 'inactive' and after['MainPID'] == '0'
record.update(after=after, dropin_path=str(dropin), dropin_sha256=hashlib.sha256(dropin.read_bytes()).hexdigest(),
              completed_unix=time.time(), release='Restore recorded active state only after explicit user release')
(base / 'ollama_deferral.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
