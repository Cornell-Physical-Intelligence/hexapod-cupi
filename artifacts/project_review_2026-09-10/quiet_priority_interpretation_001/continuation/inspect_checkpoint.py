"""Read-only CPU inventory for the continuation decision; no trainer/dispatch code."""
import argparse, hashlib, json
from pathlib import Path
import torch

def inspect(path):
    path=Path(path);before=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['optimizer_state_dict']
    result={'checkpoint_sha256':before,'bytes':path.stat().st_size,'keys':sorted(saved),
            'iter':saved.get('iter'),'infos':saved.get('infos'),
            'actor_keys':list(saved['actor_state_dict']),'critic_keys':list(saved['critic_state_dict']),
            'optimizer_entries':len(state['state']),
            'optimizer_fields':sorted({k for v in state['state'].values() for k in v}),
            'group_learning_rates':[g['lr'] for g in state['param_groups']],
            'adam_steps':sorted({v['step'].item() if torch.is_tensor(v['step']) else v['step'] for v in state['state'].values()}),
            'normalizers':{}}
    for key in ('actor_state_dict','critic_state_dict'):
        result['normalizers'][key]={name:{'shape':list(value.shape),'dtype':str(value.dtype),
            'scalar':value.item() if value.numel()==1 else None}
            for name,value in saved[key].items() if name.startswith('obs_normalizer.')}
    if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise ValueError('Input changed')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('checkpoint',type=Path)
    print(json.dumps(inspect(parser.parse_args().checkpoint),indent=2,allow_nan=False))
