"""Local publication integrity, repository-relative parent checks and CPU replay."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,os,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
ANALYZER='e4db5c7acfec918931b2a444b338fb8bc51d575c6ab1f858e31edd2384edb098'
COMPARISON='e33727444175639f5aee3cdc01ae7191d5643baa3ce189c63cce43d6b7e9f432'
SOURCE='ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
CHECKPOINT='195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173'
def check(v,message):
    if not v:raise ValueError(message)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def rel(s):
    p=PurePosixPath(s);check(not p.is_absolute() and '..' not in p.parts,'Unsafe path '+s);return s
def inventory(p,exclude=None):
    check(not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*')),'Symlink in '+str(p))
    return {x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and x!=exclude}
def freeze(p,bound=None,count=None):
    f=p/'FREEZE_SHA256.json'
    if bound:check(sha(f)==bound,'Manifest changed '+str(p))
    d=read(f)
    for k in d:rel(k)
    check(inventory(p,f)==d,'Changed/extra/missing payload '+str(p))
    if count is not None:check(len(d)==count,'Wrong payload count')
    return d
def run(args):
    r=subprocess.run(args,text=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    check(r.returncode==0,'CPU replay failed: '+r.stderr)
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path);p.add_argument('--replay',action='store_true');a=p.parse_args()
    external=read(ROOT/'EXTERNAL_PARENT_INPUTS.json')
    repo=a.repo_root
    if repo is None:
        repo=next((x for x in ROOT.parents if (x/external['CAPS_report_repository_relative_path']).is_file()),None)
    check(repo is not None,'Supply --repo-root to resolve already published parent artifacts')
    repo=repo.resolve();outer=freeze(ROOT);freeze(ROOT/'analyzer',ANALYZER,22);freeze(ROOT/'comparison',COMPARISON,7)
    parents={}
    for row in external['parents']:
        folder=repo/rel(row['repository_relative_path']);f=folder/'FREEZE_SHA256.json'
        check(sha(f)==row['manifest_sha256'],'Published parent manifest differs')
        parents[folder]=read(f)
    def parent_file(path,bound):
        check(path.is_file() and not path.is_symlink() and sha(path)==bound,'External parent payload changed '+str(path))
        candidates=[(r,m) for r,m in parents.items() if path.is_relative_to(r)]
        check(len(candidates)==1,'Ambiguous/unbound parent path')
        r,m=candidates[0];check(m[path.relative_to(r).as_posix()]==bound,'Parent map payload mismatch')
    caps=repo/rel(external['CAPS_report_repository_relative_path']);parent_file(caps,external['CAPS_report_sha256'])
    check(sha(caps)==sha(ROOT/'comparison/caps_report.json'),'Copied CAPS report differs')
    oldinputs=caps.parent/'INPUTS_SHA256.json';parent_file(oldinputs,external['CAPS_input_map_sha256']);original=read(oldinputs)
    check(len(external['CAPS_raw_inputs'])==len(original)==32,'Incomplete CAPS input map')
    check({r['original_path']:r['sha256'] for r in external['CAPS_raw_inputs']}==original,'CAPS mapped input identities differ')
    for row in external['CAPS_raw_inputs']:
        if 'repository_relative_path' in row:parent_file(repo/rel(row['repository_relative_path']),row['sha256'])
        else:check(sha(ROOT/rel(row['bundle_relative_path']))==row['sha256'],'Mapped cold input differs')
    auditpath=ROOT/'terminal/remote_terminal_audit.json';audit=read(auditpath);pins=read(ROOT/'SOURCE_PINS.json')
    check(sha(auditpath)==pins['terminal_audit_sha256'],'Wrong terminal audit')
    full=audit['inventory'];excluded=set(audit['remote_only_intermediate_autosaves'])
    check(len(full)==audit['files']==122 and sum(x['bytes'] for x in full.values())==audit['bytes']==443634816,'Full inventory differs')
    check(len(excluded)==50 and all(k.startswith('run/train/policy/model_') and k.endswith('.pt') for k in excluded),'Invalid omitted files')
    local={}
    for directory in ('run','pause'):
        for name,bound in inventory(ROOT/'terminal'/directory).items():
            key=directory+'/'+name;local[key]={'sha256':bound,'bytes':(ROOT/'terminal'/key).stat().st_size}
    check(set(local)==set(full)-excluded,'Selected inventory differs')
    check(all(local[k]==full[k] for k in local),'Selected raw size/hash mismatch')
    check(len(local)==audit['selected_files']==72 and sum(v['bytes'] for v in local.values())==audit['selected_bytes']==224646566,'Selected count/bytes differ')
    selection=read(ROOT/'PUBLICATION_SELECTION.json');check(selection['remote_only_intermediate_autosaves']==audit['remote_only_intermediate_autosaves'],'Omission selection differs')
    check(inventory(ROOT/'terminal')==read(ROOT/'LOCAL_VERIFICATION.json')['terminal_copy_map'],'Copied auxiliary file differs')
    owner=dict(row.split('=',1) for row in audit['owner']['stdout'].splitlines() if '=' in row)
    check(audit['owner']['exit_code']==0 and owner['MainPID']=='0' and owner['ExecMainStatus']=='0' and owner['ActiveState']=='inactive','Owner not historical terminal exit0')
    check(len(audit['cleanup'])==12 and all(x['absent'] is True for x in audit['cleanup']),'Incomplete owned absence')
    check(read(ROOT/'terminal/pause/restored.json')==audit['restored'] and audit['restored']['restored_unix']>0,'Restoration mismatch')
    for remote,bound in audit['pins'].items():
        head,tail=remote.split('/',1);folder={'direct_omni_train_pilot_quiet_priority_001':'run','forecast_pause_060':'pause'}.get(head)
        check(folder is not None and sha(ROOT/'terminal'/folder/rel(tail))==bound,'Historical pin mismatch')
    campaign=read(ROOT/'terminal/run/campaign.json');check(campaign==audit['campaign'],'Campaign/audit mismatch')
    check(campaign['status']=='completed' and campaign['PPO_updates_completed']==50 and campaign['terminal_inputs_unchanged'] is True and campaign['Stage2_complete'] is False,'Wrong campaign result')
    check(pins['source_manifest_sha256']==campaign['identity']['source_manifest_sha256']==SOURCE,'Wrong actual source')
    report=read(ROOT/'analysis/report.json');check(report['errors']==[] and report['evidence_verified'] is True and report['reviewed_inputs_unchanged'] is True,'Analysis evidence failure')
    check(report['training']['updates_completed']==50 and report['training']['transitions']==1228800 and report['training']['strict_reload']['passed'] is True,'Wrong bounded training result')
    check(report['training']['checkpoint_sha256']==pins['final_checkpoint_sha256']==sha(ROOT/'terminal/run/train/policy/final.pt')==CHECKPOINT,'Wrong final checkpoint')
    check(report['final_stop']['quiet_passed_replicas']==0 and report['final_stop']['total_replicas']==48,'Changed quiet verdict')
    check(sha(ROOT/'analysis/report.json')==sha(ROOT/'comparison/quiet_report.json'),'Comparison new report differs')
    mapping=read(ROOT/'PORTABLE_INPUTS.json')['mapping'];original=read(ROOT/'analysis/INPUTS_SHA256.json')
    check(len(mapping)==len(original) and {r['original_absolute_path']:r['sha256'] for r in mapping}==original,'New analyzer input map differs')
    for row in mapping:check(sha(ROOT/rel(row['bundle_relative_path']))==row['sha256'],'Portable analysis input differs')
    replay='not requested'
    if a.replay:
        with tempfile.TemporaryDirectory(prefix='quiet_priority_pilot_replay_') as temp:
            out=Path(temp)/'analysis'
            run([sys.executable,'-B',str(ROOT/'analyzer/analyze.py'),'--campaign',str(ROOT/'terminal/run'),'--cold',str(ROOT/'cold_baseline'),'--output',str(out)])
            check(read(out/'report.json')==report,'Numerical report replay differs')
            check((out/'REPORT.md').read_bytes()==(ROOT/'analysis/REPORT.md').read_bytes(),'Human report replay differs')
            result=Path(temp)/'comparison.json';run([sys.executable,'-B',str(ROOT/'comparison/compare.py'),'--output',str(result)])
            check(read(result)==read(ROOT/'comparison/comparison.json'),'Descriptive comparison replay differs')
            replay='exact numerical report, human report and CAPS comparison equality'
        freeze(ROOT)
    print(json.dumps({'result':'PASS','payloads':len(outer),'manifest_sha256':sha(ROOT/'FREEZE_SHA256.json'),
                      'selected_files':72,'selected_bytes':224646566,'full_remote_files':122,'remote_only_autosaves':50,
                      'parent_CAPS_inputs_verified':32,'updates':50,'quiet_passes':'0/48','replay':replay,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
