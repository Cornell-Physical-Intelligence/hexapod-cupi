"""Read-only CPU checkpoint/data comparison, no optimization or simulation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.paper_walk.learner import PPOLearner,Config,exact_equal,file_sha
OUT=Path(__file__).parent
request=json.loads((OUT/'REQUEST.json').read_text())
torch.set_num_threads(2)
with np.load(request['bc_dataset'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
obs=torch.from_numpy(data['observations']);target=torch.from_numpy(data['actions']);vel=torch.from_numpy(data['velocity_targets_navigation_mps'])
selection=json.loads((ROOT/'artifacts/paper_bc_data_002/SELECTION.json').read_text())
subsets={'all':np.ones(len(obs),bool),'original_steady':data['source_kind']==0,'screened_onset':data['source_kind']==1,'zero_reset_settle_added':data['source_kind']==2,'all_zero_command':(data['commands']==0).all(-1),'zero_first_three_seconds':data['zero_phase']==0,'zero_final_one_second':data['zero_phase']==1}
results={}
for fit in ['paper_bc_fit_001','paper_bc_fit_002']:
    learner=PPOLearner(None,request['prior'],OUT/(fit+'_read_only'),Config(num_envs=32,initial_std=.1,seed=20260914),device='cpu')
    checkpoint=ROOT/'artifacts'/fit/'bc_checkpoint_update000000.pt';learner.load(checkpoint)
    with torch.inference_mode():pred,estimated=learner.model.actor(obs)
    result={'checkpoint_sha256':file_sha(checkpoint),'subsets':{}}
    for name,mask in subsets.items():
        m=torch.from_numpy(mask);error=pred[m]-target[m];delta=target[m]-obs[m,213:];ve=estimated[m]-vel[m]
        result['subsets'][name]={'rows':int(m.sum()),'actor_mse':float(error.square().mean()),'copy_previous_mse':float(delta.square().mean()),'zero_action_mse':float(target[m].square().mean()),'action_absmax':float(pred[m].abs().max()),'estimator_velocity_rmse_mps':float(ve.square().mean().sqrt()),'estimator_axis_rmse_mps':ve.square().mean(0).sqrt().tolist(),'true_velocity_axis_rms_mps':vel[m].square().mean(0).sqrt().tolist(),'named_joint_action_mse':dict(zip(selection['joint_names'],error.square().mean(0).tolist()))}
    results[fit]=result
initial=[]
for fit in ['paper_bc_fit_001','paper_bc_fit_002']:initial.append(torch.load(ROOT/'artifacts'/fit/'initial_fresh_checkpoint.pt',map_location='cpu',weights_only=False))
assert exact_equal(initial[0],initial[1])
results.update(initial_checkpoint_content_exact=True,bc_dataset_sha256=file_sha(request['bc_dataset']),amp_prior_sha256=file_sha(request['prior']),estimator_velocity_supervised=False,scope='All inputs are actual recorded native observations; no closed-loop claim. Estimator output was optimized as an action latent, not velocity. Future velocity targets are prehold body-origin navigation velocity, not COM.')
(OUT/'DATA_CONTRAST.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps({fit:{name:{k:v for k,v in m.items() if k in ['actor_mse','copy_previous_mse','estimator_velocity_rmse_mps']} for name,m in results[fit]['subsets'].items()} for fit in ['paper_bc_fit_001','paper_bc_fit_002']},indent=2))
