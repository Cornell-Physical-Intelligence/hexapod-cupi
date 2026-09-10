"""Dispatch only this fresh bounded moving PPO pilot under the user's existing pause authorization."""
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
OUTPUT = BASE / 'reference_moving_ppo_001'
PAUSE = BASE / 'forecast_pause_049'
UNIT = 'hexapod-moving-ppo-001-20260910.service'
HOST = BASE / 'reference_moving_ppo_launch_001/launch_moving_ppo_spark.py'
RUN = BASE / 'reference_physics_009'
BRIDGE = BASE / 'reference_device_smoke_adapter_001'
CONSUMER = BASE / 'reference_moving_ppo_source_001'
DEVICE_RUN = BASE / 'reference_device_smoke_001'
OBSERVATION = BASE / 'reference_policy_observation_005_001'


def call(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def main():
    if PAUSE.exists() or OUTPUT.exists():
        raise RuntimeError('Fresh pause/output names required')
    sys.path.insert(0, str(SOURCE / 'tools'))
    from launch_reference_physics_spark import check_source
    from launch_length_study_spark import preflight
    check_source(SOURCE)
    if hashlib.sha256((SOURCE/'campaign_source_hashes.json').read_bytes()).hexdigest() != '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e':
        raise RuntimeError('Wrong reviewed reference source')
    if hashlib.sha256(HOST.read_bytes()).hexdigest() != '2fcf3dcbf7b66ab0726e97dafd215144fb807b45e3d10bef156271ef4767e4b5':
        raise RuntimeError('Wrong reviewed bounded PPO host')
    host_manifest = HOST.parent / 'FREEZE_SHA256.json'
    if hashlib.sha256(host_manifest.read_bytes()).hexdigest() != 'f8cb69359f9a9bbab1b812c603bce25c2de42e8692d44c5552a1374bd6279f49':
        raise RuntimeError('Wrong reviewed bounded PPO host bundle')
    host_files = json.loads(host_manifest.read_text())
    host_actual = {str(p.relative_to(HOST.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in HOST.parent.rglob('*') if p.is_file() and p != host_manifest}
    if host_actual != host_files or any(p.is_symlink() for p in HOST.parent.rglob('*')):
        raise RuntimeError('PPO host or executable helper changed')
    sys.path.insert(0, str(HOST.parent))
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('ppo_host', HOST)
    ppo_host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppo_host)
    ppo_host.validate_inputs(SimpleNamespace(source=SOURCE, run=RUN, device_run=DEVICE_RUN, bridge=BRIDGE, consumer=CONSUMER, observation=OBSERVATION, output=OUTPUT))
    previous = call(['systemctl','--user','show','hexapod-rr-preload-002-20260910.service','-p','ActiveState'])
    if 'ActiveState=active' in previous or 'ActiveState=activating' in previous:
        raise RuntimeError('Previous reference owner has not exited; do not overlap reference screen')
    if not (BASE/'forecast_pause_048/restored.json').is_file():
        raise RuntimeError('Previous reference forecasting restoration not yet recorded')
    prior_pins = {'reference_rr_preload_002/campaign.json': '8ecbecac7f7c200e3e65766c9484269d9865ccdf9e0e5b2422b6ce28d4876402', 'reference_rr_preload_002/jobs/standing.json': '311b34c725f6a4179f43cb9e310f4be99a7fa127c9e328dcd595857787e9431c', 'reference_rr_preload_002/jobs/left_strafe.json': '03dccaa351838b1eb3875a3cfc04196911a615e66fe241d1133508533fce2d51', 'forecast_pause_048/restored.json': '93afad70957f24d33345d17c47ef1267dbc6ef8262c851fb45944c06cab18a88'}
    for name,bound in prior_pins.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=bound:
            raise RuntimeError('Previous campaign/job/restoration receipt changed: '+name)
    for phase in ('standing','left_strafe'):
        job=json.loads((BASE/'reference_rr_preload_002/jobs'/ (phase+'.json')).read_text())
        if job.get('cleanup_checked') is not True or not job.get('container_id'):
            raise RuntimeError('Previous job cleanup lacks immutable identity')
        for identifier in (job['container_name'],job['container_id']):
            found=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
            if found.returncode==0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):
                raise RuntimeError('Previous owned container not proven absent: '+identifier)
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
            reason='Fresh standing, calibration,32-replica timing,ten moving PPO updates and matched screens; no automatic25/128 allocation',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-049', '--on-active=100m',
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
        cmd = ['systemd-run', '--user', '--unit=' + UNIT, '--property=RuntimeMaxSec=5520',
            '--property=TimeoutStopSec=180', '--property=KillMode=process', '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 ' + str(restorer),
            '--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3', str(HOST),
            '--source', str(SOURCE), '--run', str(RUN), '--device-run', str(DEVICE_RUN), '--bridge', str(BRIDGE), '--consumer', str(CONSUMER), '--observation', str(OBSERVATION), '--output', str(OUTPUT)]
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
