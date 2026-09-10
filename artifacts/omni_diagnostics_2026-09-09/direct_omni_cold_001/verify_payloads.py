"""Portable, read-only verification. Python standard library only; no network."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def safe(root,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or str(rel)!=name:raise ValueError('Unsafe relative path')
    p=root.joinpath(*rel.parts)
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Missing or substituted payload: '+name)
    return p
def verify_map(root,mapping):
    for name,expected in mapping.items():
        if sha(safe(root,name))!=expected:raise ValueError('Changed payload: '+name)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--preparation',type=Path);a=p.parse_args();root=a.root.resolve()
    bundle=read(root/'BUNDLE_SHA256.json')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=set(bundle)|{'BUNDLE_SHA256.json'}:raise ValueError('Publication inventory mismatch')
    verify_map(root,bundle)
    refs=read(root/'REFERENCES.json')
    if a.preparation:
        prep=a.preparation.resolve();manifest=prep/'BUNDLE_SHA256.json'
        if sha(manifest)!=refs['preparation']['bundle_sha256']:raise ValueError('Wrong preparation bundle')
        verify_map(prep,read(manifest))
    for name,bound in refs['frozen_reviews'].items():
        manifest=safe(root,name)
        if sha(manifest)!=bound:raise ValueError('Wrong nested freeze')
        verify_map(manifest.parent,read(manifest))
    evidence=root/'evidence';audit=read(evidence/'root_terminal_audit.json')
    verify_map(evidence,audit['raw_payloads'])
    if len(audit['raw_payloads'])!=23 or sum(audit['raw_sizes_bytes'].values())!=5312991:raise ValueError('Wrong raw scope')
    for name,size in audit['raw_sizes_bytes'].items():
        if safe(evidence,name).stat().st_size!=size:raise ValueError('Wrong raw byte count')
    local=read(evidence/'root_all_raw_verified.json')
    if local['audit_sha256']!=sha(evidence/'root_terminal_audit.json') or not local['all_verified']:raise ValueError('Wrong root local audit binding')
    if not audit['all_input_trees_verified'] or not audit['standing_unchanged']:raise ValueError('Missing full root integrity proof')
    if (audit['source_files'],audit['supervisor_files'],audit['legacy_runtime_files'])!=(589,926,16):raise ValueError('Wrong input scope')
    if audit['expected_invocation']!='ae3b95c756db4aa4b639ee4230e8c958':raise ValueError('Wrong historical invocation')
    if not any(x.get('USER_INVOCATION_ID')==audit['expected_invocation'] for x in audit['historical_journal']):raise ValueError('Missing historical unit identity')
    if audit['unit']['ActiveState'] not in ('inactive','failed') or audit['unit']['Result']!='success':raise ValueError('Unexpected terminal unit')
    for phase in ('standing','baseline'):
        job=read(evidence/'run/jobs'/f'{phase}.json')
        if not job['cleanup_checked']:raise ValueError('Job cleanup failed')
        for key in ('container_name','container_id'):
            if not job.get(key) or not audit['owned_names_IDs_absent'][job[key]]['absent']:raise ValueError('Missing exact owned absence')
    campaign=read(evidence/'run/campaign.json')
    if campaign['status']!='completed' or not campaign['baseline_complete'] or campaign['PPO_updates_completed']!=0:raise ValueError('Wrong completed diagnostic scope')
    standing=read(evidence/'run/standing_immutable.sha256.json');verify_map(evidence/'run/standing',standing)
    if read(evidence/'forecast_pause/restored.json')['restored_unix']!=1789052038.71282:raise ValueError('Wrong restoration')
    history=refs['historical_comparator']
    if sha(safe(root,history['path']))!=history['sha256']:raise ValueError('Wrong historical comparator')
    new=read(evidence/'run/baseline/diagnostics.json');old=read(root/history['path'])
    if new['checkpoint_sha256']!=old['checkpoint_sha256'] or new['checkpoint_sha256']!=refs['checkpoint_sha256']:raise ValueError('Wrong checkpoint binding')
    if (new['overrides']['target_slew_rad_per_20ms'],old['overrides']['target_slew_rad_per_20ms'])!=(.04,.03):raise ValueError('Mixed control profiles')
    print(json.dumps({'passed':True,'publication_payloads':len(bundle),'raw_payloads':23,'raw_bytes':5312991,
                      'complete_48_replica_diagnostic':True,'preparation_payloads_reverified':bool(a.preparation),
                      'PPO_updates_completed':0,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
