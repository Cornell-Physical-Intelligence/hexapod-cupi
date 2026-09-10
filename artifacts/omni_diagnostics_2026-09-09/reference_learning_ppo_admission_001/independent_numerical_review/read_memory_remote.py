"""Read-only one-shot memory receipt for the exact active recovery job."""
from pathlib import Path
import json
import subprocess
import time

base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_learning_ppo_001')
job=base/'jobs/learning_recovery_32.json'
report={'unix':time.time(),'job_path':str(job),'read_only':True,'memory_scope':'one snapshot; peak only if cgroup reports it'}
if not job.exists():
    report['status']='phase_not_created'
else:
    data=json.loads(job.read_text());report['job']=data
    identifier=data.get('container_id')
    if not identifier:report['status']='owned_id_not_recorded'
    else:
        result=subprocess.run(['docker','inspect',identifier],capture_output=True,text=True)
        if result.returncode:
            report.update(status='inspect_unavailable',inspect_error=result.stderr)
        else:
            item=json.loads(result.stdout)[0]
            assert item['Id']==identifier and item['Name'].lstrip('/')==data['container_name']
            report['owned_container_state']=item['State']
            if not item['State']['Running']:report['status']='owned_container_not_running'
            else:
                pid=item['State']['Pid'];cgroups=Path('/proc/%d/cgroup'%pid).read_text();report['cgroup']=cgroups
                unified=next(line.split(':',2)[2] for line in cgroups.splitlines() if line.startswith('0::'))
                root=Path('/sys/fs/cgroup')/unified.lstrip('/')
                report['cgroup_values']={k:(root/k).read_text().strip() for k in ('memory.current','memory.peak','memory.max','memory.events') if (root/k).is_file()}
                stats=subprocess.run(['docker','stats','--no-stream','--format','{{json .}}',identifier],capture_output=True,text=True)
                report['docker_stats_returncode']=stats.returncode;report['docker_stats_stdout']=stats.stdout;report['status']='sampled_owned_recovery'
print(json.dumps(report,indent=2))
