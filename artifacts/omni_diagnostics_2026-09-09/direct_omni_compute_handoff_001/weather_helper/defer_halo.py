"""Defer exactly one identified weather replay; no GPU reservation or timer edits."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time

HERE=Path(__file__).resolve().parent
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
OUTPUT=BASE/'forecast_halo_defer_001'
UNIT='stormscope-halo104-1010-20260910.service'
FALLBACK='stormscope-halo-defer-001'
INVOCATION='441bd8a01ae1421d8b0cf60283fd200e'
WORK=Path('/home/orionh/stormscope-halo-tests-20260910')
WEATHER_OUTPUT=WORK/'halo104-1010'
WEATHER_LOG=WORK/'halo104-1010.log'
SCRIPT=WORK/'halo_replay.py'
SCRIPT_SHA='08a46f047a1ba54a46d2886a27cc8cd918e16a51d83b8b1e193c8e6e17157df3'
FRAGMENT_SHA='6310615d0e68ca40bbd78e3e3fe40323d8b263b30a965b15e793703d20d04b59'
CGROUP='/user.slice/user-1000.slice/user@1000.service/app.slice/'+UNIT
SHELL_COMMAND='exec /opt/wx/venv-models/bin/python -u halo_replay.py --root '+str(WEATHER_OUTPUT)+' > halo104-1010.log 2>&1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def run(argv,**kwargs):return subprocess.run(argv,text=True,capture_output=True,timeout=kwargs.pop('timeout',30),**kwargs)

def bundle():
    manifest=HERE/'FREEZE_SHA256.json';expected=read(manifest)
    actual={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts}
    if actual!=expected or any(p.is_symlink() for p in HERE.rglob('*')):raise RuntimeError('Supplemental helper bundle changed')
    return sha(manifest)

def inspect_weather():
    properties=['LoadState','ActiveState','SubState','InvocationID','MainPID','ControlGroup','WorkingDirectory','FragmentPath','ExecStart','KillMode','KillSignal','TimeoutStopUSec','RuntimeMaxUSec','Restart']
    q=run(['systemctl','--user','show',UNIT,*sum((['-p',p] for p in properties),[])],check=True)
    return dict(line.split('=',1) for line in q.stdout.splitlines() if '=' in line)

def same_definition(state):
    if SCRIPT.is_symlink() or sha(SCRIPT)!=SCRIPT_SHA:raise RuntimeError('Weather script changed')
    fragment=Path(state.get('FragmentPath',''))
    expected={'WorkingDirectory':str(WORK),'KillMode':'control-group','KillSignal':'15','TimeoutStopUSec':'1min 30s','RuntimeMaxUSec':'2h','Restart':'no'}
    if any(state.get(k)!=v for k,v in expected.items()) or not fragment.is_file() or sha(fragment)!=FRAGMENT_SHA:
        raise RuntimeError('Weather unit definition changed')
    if 'argv[]=/bin/bash -c '+SHELL_COMMAND+' ;' not in state.get('ExecStart',''):
        raise RuntimeError('Weather ExecStart changed')

def matching_active(state):
    if state.get('ActiveState') not in ('active','activating'):
        if state.get('InvocationID') not in ('',INVOCATION,None):raise RuntimeError('A replacement weather invocation was observed')
        return False
    if state.get('InvocationID')!=INVOCATION or state.get('MainPID')!='3544965' or state.get('ControlGroup')!=CGROUP:
        raise RuntimeError('Do not stop a replacement weather job')
    same_definition(state)
    if Path('/proc/3544965/cgroup').read_text().strip()!='0::'+CGROUP:
        raise RuntimeError('Weather process ancestry changed')
    return True

def stopped_original(state):
    if state.get('ActiveState') not in ('inactive','failed') or state.get('MainPID','0')!='0':
        raise RuntimeError('Weather replay is not stopped')
    if state.get('InvocationID') not in ('',INVOCATION,None):raise RuntimeError('Weather owner changed after stop')
    cgroup=Path('/sys/fs/cgroup'+CGROUP+'/cgroup.procs')
    if cgroup.exists() and cgroup.read_text().strip():raise RuntimeError('Weather cgroup still contains processes')

def tree(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Archive input contains a symlink')
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def archive(record):
    stopped_original(inspect_weather())
    if record.get('archive'):
        p=OUTPUT/record['archive'];expected=read(p/'SHA256.json')
        if sha(p/'SHA256.json')!=record['archive_manifest_sha256']:
            raise RuntimeError('Preserved archive manifest changed')
        actual=tree(p);actual.pop('SHA256.json')
        if actual!=expected:raise RuntimeError('Preserved archive changed')
        return
    index=1
    while (OUTPUT/f'archive_{index:03d}').exists():index+=1
    p=OUTPUT/f'archive_{index:03d}';p.mkdir()
    before=tree(WEATHER_OUTPUT);log_before=sha(WEATHER_LOG)
    if sha(SCRIPT)!=SCRIPT_SHA:raise RuntimeError('Weather source changed before archive')
    shutil.copytree(WEATHER_OUTPUT,p/'outputs');shutil.copy2(WEATHER_LOG,p/'halo104-1010.log');shutil.copy2(SCRIPT,p/'halo_replay.py')
    (p/'unit_fragment.service').write_text(record['initial_unit_fragment'])
    if (tree(WEATHER_OUTPUT)!=before or tree(p/'outputs')!=before or sha(WEATHER_LOG)!=log_before
        or sha(p/'halo104-1010.log')!=log_before or sha(SCRIPT)!=SCRIPT_SHA
        or sha(p/'halo_replay.py')!=SCRIPT_SHA or sha(p/'unit_fragment.service')!=FRAGMENT_SHA):
        raise RuntimeError('Archive changed during copy; preserved partial archive, no restart')
    hashes=tree(p);save(p/'SHA256.json',hashes)
    # A final atomic full-frame output may coexist with an interrupted temporary.
    # Preserve every file, without interpreting a temporary as a completed frame.
    record.update(archive=p.name,archive_manifest_sha256=sha(p/'SHA256.json'),status='deferred',archive_complete=True)
    save(OUTPUT/'state.json',record)

def defer():
    identity=bundle()
    if OUTPUT.exists():raise RuntimeError('Fresh supplemental output required')
    state=inspect_weather()
    if not matching_active(state):return {'status':'skipped_completed_naturally','stopped':False}
    OUTPUT.mkdir();fragment=Path(state['FragmentPath']).read_text()
    record={'status':'preparing','created_unix':time.time(),'helper_freeze_sha256':identity,
            'unit':UNIT,'original_invocation':INVOCATION,'previously_active':True,
            'initial_unit':state,'initial_unit_fragment':fragment,'archive_complete':False,
            'no_midframe_resume':True,'GPU_reservation':False}
    save(OUTPUT/'state.json',record)
    # Recovery is armed before asking the exact weather service to stop.
    run(['systemd-run','--user','--unit='+FALLBACK,'--on-active=90m',
         '/usr/bin/python3','-B',str(HERE/'defer_halo.py'),'restore'],check=True)
    record['fallback_armed']=True;save(OUTPUT/'state.json',record)
    current=inspect_weather()
    if not matching_active(current):
        record.update(status='skipped_completed_naturally',previously_active=False);save(OUTPUT/'state.json',record);return record
    record['stop_requested']=True;save(OUTPUT/'state.json',record)
    run(['systemctl','--user','stop',UNIT],check=True,timeout=120)
    stopped_original(inspect_weather());record['stopped_unix']=time.time();save(OUTPUT/'state.json',record)
    archive(record)
    return record

def prove_hexapod_absent():
    units=run(['systemctl','--user','list-units','--all','--type=service',
               '--state=active,activating,deactivating','--no-legend','--plain','hexapod-*.service'],check=True).stdout.strip()
    if units:raise RuntimeError('HEXAPOD owner active; weather restart deferred')
    gpu=run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],check=True).stdout.strip()
    if gpu:raise RuntimeError('CUDA owner present; weather restart deferred')
    containers=run(['docker','ps','--format','{{.ID}} {{.Names}}'],check=True).stdout.strip()
    if any('hexapod-' in row for row in containers.splitlines()):raise RuntimeError('HEXAPOD container present')
    checked=[]
    for path in sorted(BASE.glob('direct_omni_*/jobs/*.json')):
        job=read(path);name=job.get('container_name');identity=job.get('container_id')
        if not name:continue  # Contact-log audit siblings do not own containers.
        for item in (name,identity):
            if not item:continue
            q=run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',item])
            if q.returncode==0 or not any(s in q.stderr.lower() for s in ('no such object','no such container')):
                raise RuntimeError('HEXAPOD exact owned-container absence unknown: '+item)
            checked.append(item)
    return {'active_hexapod_units':units,'compute_processes':gpu,'exact_owned_names_ids_absent':checked}

def await_restart():
    """Bounded proof that the exact restarted process acquired its own lock."""
    deadline=time.monotonic()+90.;invocation=None
    lock_stat=Path('/opt/wx/gpu.lock').stat()
    while time.monotonic()<deadline:
        state=inspect_weather()
        if state.get('ActiveState') not in ('active','activating'):
            raise RuntimeError('Restarted weather exited before acquiring its GPU lock; preserve archive, no retry')
        current=state.get('InvocationID');pid=int(state.get('MainPID','0'))
        if not current or current==INVOCATION or (invocation and current!=invocation):
            raise RuntimeError('Restart invocation missing or changed')
        invocation=current
        if (state.get('ControlGroup')!=CGROUP or state.get('WorkingDirectory')!=str(WORK)
            or 'argv[]=/bin/bash -c '+SHELL_COMMAND+' ;' not in state.get('ExecStart','')):
            raise RuntimeError('Restarted unit differs from the exact replay')
        for line in Path('/proc/locks').read_text().splitlines():
            fields=line.split()
            if len(fields)<8 or fields[1:4]!=['FLOCK','ADVISORY','WRITE'] or fields[4]!=str(pid):continue
            major,minor,inode=fields[5].split(':')
            if (int(major,16),int(minor,16),int(inode))==(os.major(lock_stat.st_dev),os.minor(lock_stat.st_dev),lock_stat.st_ino):
                return {'invocation':current,'main_pid':pid,'own_weather_lock_verified':True,
                        'proof':'kernel /proc/locks PID/device/inode; replay completion remains unverified'}
        time.sleep(.25)
    raise RuntimeError('Weather restart did not acquire its own lock within90s; no retry')

def restore():
    identity=bundle();record=read(OUTPUT/'state.json')
    if record['helper_freeze_sha256']!=identity:raise RuntimeError('Helper changed since deferral')
    if (OUTPUT/'restored.json').exists():return read(OUTPUT/'restored.json')
    if not record.get('previously_active') or not record.get('stop_requested'):
        return {'status':'no_deferred_weather_to_restart'}
    archive(record)  # Complete/verify immutable snapshot before any restart.
    state=inspect_weather();stopped_original(state)
    if sha(SCRIPT)!=SCRIPT_SHA:raise RuntimeError('Weather source changed; no restart')
    locks=[]
    try:
        for path in ('/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock'):
            fd=os.open(path,os.O_RDONLY);locks.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        absence=prove_hexapod_absent()
        state=inspect_weather();stopped_original(state)
        if state.get('LoadState')=='not-found':
            # Recreate only the captured transient unit, with the exact original
            # argv/working directory/runtime and unchanged default stop behavior.
            command=['systemd-run','--user','--unit='+UNIT,'--working-directory='+str(WORK),
                '--property=RuntimeMaxSec=2h','--property=KillMode=control-group',
                '--property=KillSignal=15','--property=TimeoutStopSec=90','--property=Restart=no',
                '/bin/bash','-c',SHELL_COMMAND]
            method='recreate_collected_exact_transient_unit'
        else:
            same_definition(state);command=['systemctl','--user','start',UNIT];method='start_existing_exact_unit'
        # The replay itself acquires gpu.lock nonblocking. Release the brief
        # verification locks before launch so we cannot make our own child fail.
        for fd in reversed(locks):os.close(fd)
        locks.clear()
        run(command,check=True,timeout=30)
        startup=await_restart()
        result={'status':'restarted_own_lock_verified','restored_unix':time.time(),'unit':UNIT,
                'method':method,'command':command,'absence':absence,
                'archive':record['archive'],'archive_manifest_sha256':record['archive_manifest_sha256'],
                'startup':startup,'recomputes_original_replay':True,'midframe_resume':False}
        save(OUTPUT/'restored.json',result);return result
    finally:
        for fd in reversed(locks):os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['defer','restore']);a=p.parse_args()
    # Supplemental operation serialization only; no long-lived GPU reservation.
    lock_path=OUTPUT.parent/(OUTPUT.name+'.operation.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            print(json.dumps(defer() if a.action=='defer' else restore(),indent=2))
        except Exception as exc:
            if OUTPUT.exists():
                save(OUTPUT/(a.action+'_failure_'+str(time.time_ns())+'.json'),
                     {'time':time.time(),'action':a.action,'error':repr(exc),'restart_not_guaranteed':True})
            raise

if __name__=='__main__':main()
