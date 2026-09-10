"""Pure inventory planner: never copies/deletes raw or dispatches analysis."""
import argparse,json,re
from pathlib import PurePosixPath,Path
AUDIT_LOCAL_CAP=64*1024*1024
GIT_SINGLE_FILE_CAP=100_000_000  # Conservative decimal boundary.
CHUNK_BYTES=64*1024*1024
RESERVE_BYTES=256*1024*1024
AUDIT_RAW={'train/training_trace.npz','train/training_joint_trace.npz','train/training_events.json'}

def plan(inventory,free_bytes):
    rows=[]
    for name,item in sorted(inventory.items()):
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path)!=name:raise ValueError('Unsafe inventory path')
        size=item['bytes'];digest=item['sha256']
        if type(size) is not int or size<0 or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid inventory size/hash')
        relative=name.removeprefix('run/')
        ordinary=bool(re.fullmatch(r'train/policy/model_\d+\.pt',relative))
        # All other files, especially eval traces and selected decisions, are retained.
        category='remote_only_ordinary_autosave' if ordinary else ('remote_only_required_training_raw' if relative in AUDIT_RAW and size>AUDIT_LOCAL_CAP else 'local_required')
        row={'path':name,'bytes':size,'sha256':digest,'disposition':category}
        if category=='remote_only_required_training_raw':row['claim_limit']='Required by frozen analyzer004; only a verified full remote analysis may claim replay of this omitted input.'
        if category=='local_required' and size>GIT_SINGLE_FILE_CAP:row['publication_encoding']={'kind':'lossless_chunks','chunk_bytes':CHUNK_BYTES,'whole_file_sha256':digest,'required_storage_is_not_reduced':True}
        rows.append(row)
    selected=sum(r['bytes'] for r in rows if r['disposition']=='local_required')
    return {'schema':'extended_acquisition_plan_v1','inventory_rows':rows,'total_remote_bytes':sum(r['bytes'] for r in rows),'selected_local_bytes':selected,'omitted_remote_only_bytes':sum(r['bytes'] for r in rows if r['disposition']!='local_required'),'required_free_bytes_with_reserve':selected+RESERVE_BYTES,'available_local_free_bytes':free_bytes,'fits_local_budget':free_bytes>=selected+RESERVE_BYTES,'local_training_replay_blocked_by_omitted_raw':any(r['disposition']=='remote_only_required_training_raw' for r in rows),'local_training_replay_verified':False,'remote_full_replay_performed':False,'admission':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--free-bytes',type=int,required=True);a=p.parse_args()
    print(json.dumps(plan(json.loads(a.inventory.read_text()),a.free_bytes),indent=2))
