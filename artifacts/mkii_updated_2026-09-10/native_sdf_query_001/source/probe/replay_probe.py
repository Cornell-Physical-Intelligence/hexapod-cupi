"""Portable CPU replay of a separately pinned native result; no simulator import."""
import argparse
import json
from pathlib import Path
import numpy as np
from mesh_oracle import frame_prediction
from native_probe import sha,save,rotation,API_SHA
from score_probe import score,finite


def replay(run,expected_state_sha256):
    run=Path(run);own=Path(__file__).resolve().parent
    manifest=json.loads((own/'FREEZE_SHA256.json').read_text())
    for name,digest in manifest.items():
        if sha(own/name)!=digest:raise ValueError('Changed probe bundle: '+name)
    if sha(run/'state.json')!=expected_state_sha256:raise ValueError('Unbound actual state')
    state=json.loads((run/'state.json').read_text())
    if state['identity']['probe_freeze_sha256']!=sha(own/'FREEZE_SHA256.json'):raise ValueError('Wrong actual probe lineage')
    for name,digest in state['outputs'].items():
        if Path(name).name!=name or sha(run/name)!=digest:raise ValueError('Changed or invalid raw payload')
    actual={p.name for p in run.iterdir()if p.is_file()and p.name!='state.json'}
    if actual!=set(state['outputs']):raise ValueError('Unexpected raw inventory')
    if state['status']!='completed':
        return {'raw_verified':True,'status':'failed_prefix_preserved','actual_errors':state['errors'],
                'queries_completed':state['queries_completed'],'contact_admitted':False,'standing_admitted':False}
    if state['queries_completed']!=12 or not state['inputs_unchanged']:raise ValueError('Incomplete native result')
    before=json.loads((run/'before.json').read_text());after=json.loads((run/'after.json').read_text())
    if before!=after:raise ValueError('Observed native state changed')
    fixture=json.loads((own/'fixture/fixture.json').read_text());p=np.asarray(fixture['points'],dtype=np.float32)
    anchors=np.asarray(fixture['semantic_anchor_indices']);raw=[];repeated=[];world=[];links=[]
    with np.load(own/'fixture/mesh.npz',allow_pickle=False)as z:v=z['vertices'];f=z['faces']
    for k,item in enumerate(fixture['shapes']):
        view=json.loads((run/f'view_{k}.json').read_text())
        if view['object_paths']!=[item['path']]or view['count']!=1 or view['max_num_points']!=len(p) or not view['check']or view['provider_sha256']!=API_SHA:raise ValueError('Wrong native view or provider')
        with np.load(run/f'query_{k}.npz',allow_pickle=False)as z:
            order=np.roll(np.arange(len(p)),k*17)
            if not np.array_equal(z['input_order'],order) or not np.array_equal(z['repeat_input_order'],order[::-1]):raise ValueError('Changed query ordering')
            if not np.array_equal(z['input_points'],p[order][None])or not np.array_equal(z['repeat_input_points'],p[order[::-1]][None]):raise ValueError('Wrong native input')
            raw.append(finite(z['raw'],(1,len(p),4))[0][np.argsort(order)])
            repeated.append(finite(z['repeat_raw'],(1,len(p),4))[0][np.argsort(order[::-1])])
        pose=finite(before['link_pose'][0][before['link_names'].index(item['link'])],(7,))
        t=np.eye(4);t[:3,:3]=rotation(pose[3:]);t[:3,3]=pose[:3];local=np.asarray(item['shape_to_link'])
        world.append(frame_prediction(p[anchors],v,f,t@local));links.append(frame_prediction(p[anchors],v,f,local))
    result=score(raw,repeated,fixture,world,links)
    recorded=json.loads((run/'report.json').read_text())
    result['recorded_score_exact_replay']=all(recorded.get(key)==value for key,value in result.items())
    result['raw_verified']=True;result['state_sha256']=expected_state_sha256
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--state-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();destination=args.output.resolve()
    if destination.exists()or destination.is_relative_to(Path(__file__).resolve().parent)or destination.is_relative_to(args.run.resolve()):raise ValueError('Output must be fresh and outside inputs')
    save(destination,replay(args.run,args.state_sha256))
