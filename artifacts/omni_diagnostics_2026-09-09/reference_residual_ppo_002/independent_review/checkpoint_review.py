"""Strict CPU readback of actual terminal checkpoints; no training or physics."""
from pathlib import Path
import hashlib,importlib.metadata,json,sys
import numpy as np
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parent;TMP=ROOT.parent;consumer=TMP/'reference_residual_ppo_source_002'
sys.path.insert(0,str(consumer))
from residual_runner import initialize_scratch,reload_checkpoint,runner_config,contract
from rsl_rl.runners import OnPolicyRunner
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
class RecordedObservationEnv:
 def __init__(self):
  with np.load(ROOT/'run/smoke/final_observation.npz',allow_pickle=False) as z:self.obs={k:torch.from_numpy(z[k].copy()) for k in ('policy','critic')}
  self.num_envs=32;self.num_actions=18;self.device='cpu';self.max_episode_length=4500;self.episode_length_buf=torch.zeros(32,dtype=torch.long);self.cfg={'scope':'Read-only recorded-observation checkpoint review'}
 def get_observations(self):return TensorDict({k:v.clone() for k,v in self.obs.items()},batch_size=[32])
 def step(self,*_):raise RuntimeError('No physical/optimization steps permitted in checkpoint review')
def main():
 assert importlib.metadata.version('rsl-rl-lib')=='5.0.1'
 import rsl_rl
 for row in read(consumer/'RSL_SOURCE_PARITY.json')['sources']:assert sha(Path(rsl_rl.__file__).parent.parent/row['relative_file'])==row['installed_Spark_sha256']
 torch.set_num_threads(1);run=ROOT/'run';state=read(run/'smoke/state.json');identity=state['policy_identity'];assert identity==contract(identity['scope'],identity['bindings'])
 assert identity['bindings']['consumer_source_sha256']=='e9a1d6d687504fd1323d8addbcb962c3025745a258f71b9f076879778eedaa0b'
 assert identity['bindings']['calibration_sha256']==sha(run/'smoke/calibration.json')
 env=RecordedObservationEnv();results={};actor_states={}
 for which in ('initial','final'):
  path=run/'smoke'/(which+'.pt');assert sha(path)==state[which+'_checkpoint']['checkpoint_sha256']==read(run/('evaluate_'+which)/'state.json')['checkpoint_sha256']
  runner=OnPolicyRunner(env,runner_config('cpu'),log_dir=None,device='cpu');initialize_scratch(runner)
  receipt=reload_checkpoint(runner,path,identity);runner.alg.eval_mode()
  with torch.inference_mode():action=runner.alg.actor(env.get_observations(),stochastic_output=False)
  assert torch.isfinite(action).all()
  if which=='initial':assert torch.equal(action,torch.zeros_like(action))
  saved=torch.load(path,map_location='cpu',weights_only=False);actor_states[which]=saved['actor_state_dict']
  entries=len(saved['optimizer_state_dict']['state']);assert entries==(0 if which=='initial' else 17)
  results[which]={'checkpoint_sha256':sha(path),'strict_actor_critic_normalizer_optimizer_readback':receipt['actor_critic_normalizer_and_optimizer_exact'],'optimizer_entries':entries,'deterministic_action_abs_max_on_recorded_final_smoke_observation':float(action.abs().max()),'actor_width':846,'critic_width':849}
 changed=[k for k in actor_states['initial'] if not torch.equal(actor_states['initial'][k],actor_states['final'][k])]
 assert any(k.startswith('mlp.') for k in changed)
 out=ROOT/'checkpoint_review.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps({'scope':'Actual completed002 checkpoint strict CPU readback on recorded observations; no learning or physical step','rsl_rl_version':importlib.metadata.version('rsl-rl-lib'),'installed_RSL_source_hashes_match_Spark':True,'checkpoints':results,'changed_actor_state_keys':changed,'physical_training_updates_recorded':2,'CPU_training_updates':0,'GPU_launches':0,'Stage2_complete':False},indent=2)+'\n')
if __name__=='__main__':main()
