"""Run in exact Isaac image, CPU only, with installed RSL API."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from omni_repair_training import load_repair_checkpoint
root=Path(__file__).parents[1]
cfg=yaml.safe_load((root/'installed/agent.yaml').read_text())
cfg['device']='cpu'
obs=TensorDict({'policy':torch.zeros(4,315),'critic':torch.zeros(4,318)},batch_size=[4])
env=SimpleNamespace(get_observations=lambda:obs,num_actions=18,num_envs=4,cfg={},device='cpu')
runner=OnPolicyRunner(env,cfg,device='cpu')
options={'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}
report=load_repair_checkpoint(runner,root/'original/policy.pt',options)
assert torch.isfinite(runner.get_inference_policy()(obs)).all()
import json
print('INSTALLED_CPU_SMOKE '+json.dumps(report))
