"""Read-only remote retrieval; refuses to overwrite preserved originals."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
       '-o', 'ControlPath=/tmp/hexapod_profiler_readonly.sock', '-o',
       'ConnectTimeout=15', '-o', 'BatchMode=yes', 'orionh@100.82.166.9', 'python3 -']
ROOT = '/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1'
CAMPAIGN = ROOT + '/campaign_001/fourbar-campaign-20260907T004524Z-8592e14a'
NOMINAL = CAMPAIGN + '/nominal/hexapod-fourbar-validate-20260907T004734Z-56533deb'
FILES = {f'raw/nominal/{name}': f'{NOMINAL}/{name}' for name in (
    'admitted', 'container.log', 'cpu_asset_audit.json', 'report.json',
    'source.SHA256SUMS', 'supervisor.json')}
FILES['raw/campaign.json'] = CAMPAIGN + '/campaign.json'
FILES.update({f'raw/run/{name}': f'{ROOT}/{name}' for name in (
    'launch_001.json', 'host_001.log', 'capture_followup_launch_001.json',
    'continuation_launch_001.json', 'capture_followup_001.launcher.log',
    'continuation_001.launcher.log', 'capture_followup_001/state.json',
    'continuation_001/state.json')})
REMOTE_ONLY = {'nominal_source_archive': NOMINAL + '/source.tar.gz'}


def remote(mode):
    script = """
from pathlib import Path
import base64,datetime,hashlib,json,os,time
files = FILES
remote_only = REMOTE_ONLY
mode = MODE
def inspect(path, include):
    p=Path(path);before=p.stat();h=hashlib.sha256();chunks=[]
    with p.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):
            h.update(data)
            if include:chunks.append(data)
    after=p.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise RuntimeError('Source changed while read: '+path)
    value={'path':path,'sha256':h.hexdigest(),'bytes':after.st_size,
        'mtime_ns':after.st_mtime_ns,'inode':after.st_ino}
    if include:value['base64']=base64.b64encode(b''.join(chunks)).decode('ascii')
    return value
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'mode':mode,'files':{k:inspect(v,mode=='transfer') for k,v in files.items()},
    'remote_only':{k:inspect(v,False) for k,v in remote_only.items()},'process_snapshot':{}}
result['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
result['boottime_s']=time.clock_gettime(time.CLOCK_BOOTTIME)
for label,pid,start in [('campaign',2021559,105778155),('capture_follower',2030164,105810847),('continuation',2075580,106045128)]:
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        actual=int(fields[19]);same=actual==start
        item={'pid':pid,'expected_start_ticks':start,'actual_start_ticks':actual,'same_process':same,'state':fields[0]}
        if same:item['argv']=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if v]
    except FileNotFoundError:item={'pid':pid,'expected_start_ticks':start,'process_absent':True}
    result['process_snapshot'][label]=item
print(json.dumps(result))
""".replace('FILES', repr(FILES)).replace('REMOTE_ONLY', repr(REMOTE_ONLY)).replace('MODE', repr(mode))
    completed = subprocess.run(SSH, input=script, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout), completed.stderr


def main():
    if (HERE/'raw').exists() or (HERE/'retrieval.json').exists():
        raise ValueError('Refusing to overwrite existing retrieval')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before, stderr_before = remote('transfer')
    for relative, record in before['files'].items():
        data=base64.b64decode(record.pop('base64'), validate=True)
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Transferred bytes differ: '+relative)
        output=HERE/relative;output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:stream.write(data)
    after, stderr_after = remote('verify_after_transfer')
    comparisons=[]
    for group in ('files','remote_only'):
        for relative, record in before[group].items():
            current=after[group][relative]
            stable=record==current
            local_hash=hashlib.sha256((HERE/relative).read_bytes()).hexdigest() if group=='files' else None
            comparisons.append({'group':group,'name':relative,'remote_unchanged':stable,
                'local_sha256':local_hash,'local_matches_remote_after_transfer':local_hash==current['sha256'] if group=='files' else None})
    passed=all(r['remote_unchanged'] and r['local_matches_remote_after_transfer'] is not False for r in comparisons)
    result={'schema':'hexapod.remote_failure_evidence_retrieval.v1','started_utc':started,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ssh_argv':SSH,
        'remote_mutation':False,'gpu_import_or_job':False,'pass':passed,
        'before':before,'after':after,'comparisons':comparisons,
        'ssh_stderr_before':stderr_before,'ssh_stderr_after':stderr_after}
    (HERE/'retrieval.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':passed,'raw_files':len(FILES),'remote_only_files':len(REMOTE_ONLY)},indent=2))
    if not passed:raise ValueError('Remote evidence changed; preserve this attempt and investigate')


if __name__=='__main__':main()
