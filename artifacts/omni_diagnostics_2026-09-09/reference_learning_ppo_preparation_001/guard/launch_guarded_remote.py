"""Dispatch only the three-phase learning admission; never allocate a PPO update."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
sys.dont_write_bytecode = True
import time

BASE = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SOURCE = BASE / 'reference_physics_source_009'
OUTPUT = BASE / 'reference_learning_ppo_001'
PAUSE = BASE / 'forecast_pause_051'
UNIT = 'hexapod-learning-ppo-admission-001-20260910.service'
HOST = BASE / 'reference_learning_ppo_host_001/launch_learning_ppo_spark.py'
RUN = BASE / 'reference_physics_009'
BRIDGE = BASE / 'reference_device_smoke_adapter_001'
CONSUMER = BASE / 'reference_moving_ppo_source_003'
DEVICE_RUN = BASE / 'reference_device_smoke_001'
OBSERVATION = BASE / 'reference_policy_observation_005_001'
HOST_SHA256 = '004bc9eb3cd984e35d229d3dd62afbfb295664a0b9c3cbf6f86ab2a91c6ff79d'
HOST_FREEZE_SHA256 = '45f623d298c00136ca18d159261d8df601bd42840ebae467fa94bf38283e6792'
CONSUMER_FREEZE_SHA256 = 'fd9bef87dda976f9b229e7541408d24e674245ca2c51231dfce9075b81d38334'
PREVIOUS_UNIT = 'hexapod-pair-motion-001-20260910.service'
PREVIOUS_INVOCATION = 'aeb33fdfce794cfca50e4227ebfada1d'



def call(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def require_final_bindings():
    for name, value in (('host', HOST_SHA256), ('host freeze', HOST_FREEZE_SHA256),
                        ('consumer freeze', CONSUMER_FREEZE_SHA256)):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Final reviewed binding is still pending: ' + name)


def verify_frozen(root, expected):
    manifest = root / 'FREEZE_SHA256.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Wrong reviewed frozen bundle: ' + str(root))
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise RuntimeError('Frozen input symbolic substitution')
    declared = json.loads(manifest.read_text())
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and p != manifest}
    if actual != declared:
        raise RuntimeError('Frozen source changed or has unlisted files: ' + str(root))


def verify_previous_owner():
    previous = call(['systemctl','--user','show',PREVIOUS_UNIT,
                     '-p','ActiveState','-p','SubState','-p','InvocationID'])
    fields = dict(line.split('=', 1) for line in previous.splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive', 'failed'):
        raise RuntimeError('Previous reference owner has not exited')
    if fields.get('InvocationID') != PREVIOUS_INVOCATION:
        raise RuntimeError('Previous owner invocation changed')
    prior_pins = {'reference_pair_motion_001/campaign.json': '99517a4cc7794acf8a72a6b938c329f35e1e70b6f5f46f84d1a3f400bcca00ad', 'forecast_pause_050/restored.json': 'a4e6b2992e657e2f25db68f33b3efd4557b265b93407662de4164a9344235105', 'reference_pair_motion_001/jobs/standing.json': 'd87fb6d59f4ce8de155bcef649edda0f5cee9a0abc4e05eaeb340b579bf46eb5', 'reference_pair_motion_001/jobs/paired_forward.json': '2243fe933be09682f85c65a4946b3eedefc1ce8278b3b7c01fa97a5454030c9a'}
    for name, bound in prior_pins.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != bound:
            raise RuntimeError('Previous campaign/job/restoration receipt changed: ' + name)
    for phase in ('standing', 'paired_forward'):
        job = json.loads((BASE/'reference_pair_motion_001/jobs'/(phase+'.json')).read_text())
        if job.get('cleanup_checked') is not True or not job.get('container_id') or not job.get('container_name'):
            raise RuntimeError('Previous job cleanup lacks immutable identity')
        for identifier in (job['container_name'], job['container_id']):
            found = subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],
                                   text=True,capture_output=True,timeout=20)
            if found.returncode == 0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):
                raise RuntimeError('Previous owned container not proven absent: ' + identifier)


def main():
    if PAUSE.exists() or OUTPUT.exists():
        raise RuntimeError('Fresh pause/output names required')
    require_final_bindings()
    sys.path.insert(0, str(SOURCE / 'tools'))
    from launch_reference_physics_spark import check_source
    from launch_length_study_spark import preflight
    check_source(SOURCE)
    if hashlib.sha256((SOURCE/'campaign_source_hashes.json').read_bytes()).hexdigest() != '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e':
        raise RuntimeError('Wrong reviewed reference source')
    if hashlib.sha256(HOST.read_bytes()).hexdigest() != HOST_SHA256:
        raise RuntimeError('Wrong reviewed learning-admission host')
    verify_frozen(HOST.parent, HOST_FREEZE_SHA256)
    verify_frozen(CONSUMER, CONSUMER_FREEZE_SHA256)
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('learning_host', HOST)
    learning_host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_host)
    if learning_host.CONSUMER_FREEZE != CONSUMER_FREEZE_SHA256:
        raise RuntimeError('Host and guard bind different consumers')
    learning_host.validate_inputs(SimpleNamespace(source=SOURCE, run=RUN, device_run=DEVICE_RUN,
        bridge=BRIDGE, consumer=CONSUMER, observation=OBSERVATION, output=OUTPUT,
        phase_group='admission', decision_receipt=None))
    verify_previous_owner()
    coordination = Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md').read_bytes()
    if hashlib.sha256(coordination).hexdigest() != 'fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234':
        raise RuntimeError('Coordination changed; review it before launch')
    locks, restorer, launched = [], None, False
    try:
        units = {name: call(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID'])
            for name in ('stormscope-dispatch.timer', 'stormscope-scout.timer', 'stormscope-dispatch.service', 'stormscope-scout.service')}
        active_services = [name for name, value in units.items() if name.endswith('.service')
            and ('ActiveState=active' in value or 'ActiveState=activating' in value)]
        parents = {int(next(line.split('=', 1)[1] for line in units[name].splitlines() if line.startswith('MainPID=')))
                   for name in active_services}
        parents.discard(0)
        gpu = call(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
        for line in gpu.splitlines():
            pid = int(line.split(',', 1)[0]); seen = set()
            while pid > 1 and pid not in parents and pid not in seen:
                seen.add(pid)
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                except FileNotFoundError:
                    pid = 0
                    break
                pid = int(next(v.split()[1] for v in status.splitlines() if v.startswith('PPid:')))
            if pid != 0 and pid not in parents:
                raise RuntimeError('Unrelated CUDA process; no forecasting pause or candidate launch: ' + line)
        snapshot = {'compute_processes_before_authorized_pause': gpu,
            'containers': call(['docker', 'ps', '--format', '{{.ID}} {{.Names}} {{.Image}}']),
            'forecast_services_authorized_for_graceful_stop': active_services,
            'service_stop_semantics': {name: call(['systemctl','--user','show',name,'-p','KillMode','-p','KillSignal','-p','TimeoutStopUSec']) for name in active_services}}
        PAUSE.mkdir()
        record = dict(user_authorized_pause=True, created_unix=time.time(), units=units, unit=UNIT,
            output=str(OUTPUT), source=str(SOURCE), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reason='Fresh standing, calibration and learning_recovery_32 admission only; exit before any PPO updates or separate reviewed continuation',
            current_instruction='9 September C-study authorization supersedes older physical-model pause notes')
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
        restorer = PAUSE / 'resume_forecasting.py'
        restorer.write_text('''from pathlib import Path
import fcntl,json,subprocess,sys,time
p=Path(__file__).parent
r=json.loads((p/'pause.json').read_text())
if '--stop-owner' in sys.argv:
    subprocess.run(['systemctl','--user','stop',r['unit']],check=True,timeout=240)
with (p/'restore.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (p/'restored.json').exists():raise SystemExit(0)
    cleanup=[]
    for job in sorted((Path(r['output'])/'jobs').glob('*.json')):
        j=json.loads(job.read_text()); name=j.get('container_name'); identity=j.get('container_id')
        if not name:continue
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identity or name],text=True,capture_output=True,timeout=20)
        if q.returncode:
            if 'no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower():continue
            raise RuntimeError('Owned container absence is unknown; keep forecasting paused: '+q.stderr.strip())
        fields=q.stdout.strip().split()
        if len(fields)!=3 or fields[1]!='/'+name or (identity and fields[0]!=identity):raise RuntimeError('Owned cleanup identity mismatch')
        if fields[2]=='true':subprocess.run(['docker','stop','--time','20',fields[0]],check=True,timeout=30,capture_output=True)
        after=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',fields[0]],text=True,capture_output=True,timeout=20)
        if after.returncode:
            if 'no such object' not in after.stderr.lower() and 'no such container' not in after.stderr.lower():raise RuntimeError('Post-stop container absence unknown')
        elif after.stdout.strip()!=fields[0]+' /'+name+' false':raise RuntimeError('Owned container not verified stopped')
        cleanup.append(fields[0])
    active=[name for name,state in r['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state]
    if active:subprocess.run(['systemctl','--user','start',*active],check=True,timeout=30)
    result={'restored_unix':time.time(),'timers':active,'owned_cleanup_checked':cleanup}
    t=p/'restored.tmp';t.write_text(json.dumps(result,indent=2)+'\\n');t.replace(p/'restored.json')
''')
        # Arm recovery before mutating timer state. It stops only this exact
        # owner unit if needed, then restores only previously active timers.
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-051', '--on-active=40m',
            '/usr/bin/python3', str(restorer), '--stop-owner'], check=True)
        active = [name for name, state in units.items() if name.endswith('.timer') and 'ActiveState=active' in state]
        if active:
            subprocess.run(['systemctl', '--user', 'stop', *active], check=True, timeout=30)
        record['gracefully_stopped_services'] = active_services
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
        if active_services:
            subprocess.run(['systemctl', '--user', 'stop', *active_services], check=True, timeout=120)
        if any('ActiveState=active' in call(['systemctl', '--user', 'show', name, '-p', 'ActiveState'])
               or 'ActiveState=activating' in call(['systemctl', '--user', 'show', name, '-p', 'ActiveState'])
               for name in units if name.endswith('.service')):
            raise RuntimeError('Forecast service raced the timer pause; preserve it and restore timers')
        for path in ('/opt/wx/gpu.lock', '/tmp/hexapod-isaac-gpu.lock'):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record['post_pause_preflight'] = preflight()
        record['paused_unix'] = time.time()
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
    except Exception:
        if restorer is not None:
            subprocess.run(['/usr/bin/python3', str(restorer)], check=True, timeout=120)
        raise
    finally:
        for fd in reversed(locks):
            os.close(fd)
    try:
        cmd = ['systemd-run', '--user', '--unit=' + UNIT, '--property=RuntimeMaxSec=1920',
            '--property=TimeoutStopSec=180', '--property=KillMode=process', '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 ' + str(restorer),
            '--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3', str(HOST), '--phase-group', 'admission',
            '--source', str(SOURCE), '--run', str(RUN), '--device-run', str(DEVICE_RUN),
            '--bridge', str(BRIDGE), '--consumer', str(CONSUMER), '--observation', str(OBSERVATION),
            '--output', str(OUTPUT)]
        subprocess.run(cmd, check=True, timeout=30)
        launched = True
        (PAUSE / 'launch.json').write_text(json.dumps(dict(command=cmd, launched_unix=time.time()), indent=2) + '\n')
        print(json.dumps(dict(unit=UNIT, output=str(OUTPUT), pause=str(PAUSE), source=str(SOURCE))))
    finally:
        if not launched:
            # A failed/timed-out systemd-run client does not prove that the
            # owner unit was never started. Stop that exact owner before the
            # restorer examines its jobs and resumes forecasting timers.
            subprocess.run(['/usr/bin/python3', str(restorer), '--stop-owner'], check=True, timeout=300)


if __name__ == '__main__':
    main()
