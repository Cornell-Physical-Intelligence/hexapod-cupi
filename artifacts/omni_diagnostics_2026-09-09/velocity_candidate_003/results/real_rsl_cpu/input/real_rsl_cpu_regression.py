"""Run the actual installed RSL-RL 5.0.1 PPO/logger/checkpoint lifecycle on CPU.

The vector environment is an explicitly synthetic CPU fixture using the actual
candidate action state. It cannot qualify contacts, torque or walking.
"""
from pathlib import Path
import copy,hashlib,importlib.metadata,json,os
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from candidate_runner import initialize_scratch,save_candidate,load_candidate
from velocity_action import JointTargetVelocity,VelocityActionConfig

HERE=Path(__file__).parent

def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:assert_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b

class CPUVecFixture:
    def __init__(self):
        self.num_envs=32;self.num_actions=18;self.device='cpu';self.max_episode_length=1000
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long)
        self.cfg={'kind':'synthetic_CPU_runner_fixture_not_robot_admission'}
        names=tuple('joint_'+str(i) for i in range(18))
        self.controller=JointTargetVelocity(names,{n:-1. for n in names},{n:1. for n in names},32,
                                            VelocityActionConfig('formal_004'),dtype=torch.float32)
        self.controller.reset(torch.zeros(32,18))
        self.previous_action=torch.zeros(32,18);self.history=torch.zeros(32,5,99);self.steps=0
        self._refresh()
    def _refresh(self):
        frame=torch.zeros(32,99)
        frame[:,5]=-1.;frame[:,6]=.5
        frame[:,9:27]=self.controller.position
        frame[:,27:45]=self.controller.velocity*.05
        frame[:,45:63]=self.previous_action
        frame[:,63:81]=self.controller.position
        frame[:,81:99]=self.controller.velocity/2.
        self.history[:,:-1]=self.history[:,1:].clone();self.history[:,-1]=frame
        if self.steps==0:self.history[:]=frame[:,None]
    def get_observations(self):
        actor=self.history.flatten(1).clone()
        return TensorDict({'policy':actor,'critic':torch.cat((actor,torch.zeros(32,3)),-1)},batch_size=[32])
    def step(self,actions):
        self.controller.step(actions);self.previous_action=actions.detach().clone()
        self.episode_length_buf+=1;self.steps+=1;self._refresh()
        reward=-self.controller.position.square().sum(-1)-.01*actions.square().sum(-1)
        reward+=.01*self.controller.position.mean(-1)
        return self.get_observations(),reward,torch.zeros(32,dtype=torch.long),{}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh CPU regression output')
    args.output.mkdir(parents=True)
    assert importlib.metadata.version('rsl-rl-lib')=='5.0.1'
    assert not torch.cuda.is_available(),'CPU regression must not receive a GPU'
    torch.set_num_threads(1);torch.manual_seed(7057)
    cfg=yaml.safe_load((HERE/'installed_agent.yaml').read_text());cfg['device']='cpu'
    config=VelocityActionConfig('formal_004')
    identity={'variant':'CPU_fixture_only','urdf_sha256':'0'*64,'plan_sha256':hashlib.sha256((HERE/'installed_agent.yaml').read_bytes()).hexdigest(),
              'stance_index':0,'source_sha256':hashlib.sha256((HERE/'candidate_runner.py').read_bytes()).hexdigest()}
    # Reproduce the actual installed failure before exercising the correction.
    broken=OnPolicyRunner(CPUVecFixture(),copy.deepcopy(cfg),log_dir=str(args.output/'reproduction'),device='cpu')
    assert not hasattr(broken.logger,'writer')
    try:broken.save(str(args.output/'old_early_save.pt'))
    except AttributeError as exc:
        assert 'writer' in str(exc);reproduced=repr(exc)
    else:raise AssertionError('Expected installed RSL early-save lifecycle failure was not reproduced')
    env=CPUVecFixture();runner=OnPolicyRunner(env,copy.deepcopy(cfg),log_dir=str(args.output/'policy'),device='cpu')
    initialization=initialize_scratch(runner,config)
    fixed_initial=env.get_observations()
    with torch.inference_mode():assert torch.equal(runner.alg.actor(fixed_initial,stochastic_output=False),torch.zeros(32,18))
    initial=save_candidate(runner,args.output/'initial.pt',config,identity)
    assert runner.logger.writer is None
    runner.learn(num_learning_iterations=2,init_at_random_ep_len=False)
    assert env.steps==48
    assert runner.logger.writer is not None and runner.logger.logger_type=='tensorboard'
    assert len(runner.alg.optimizer.state)>0
    final=save_candidate(runner,args.output/'final.pt',config,identity)
    saved=torch.load(args.output/'final.pt',weights_only=False,map_location='cpu')
    first=torch.load(args.output/'initial.pt',weights_only=False,map_location='cpu')
    changed=[key for key in saved['actor_state_dict'] if not torch.equal(saved['actor_state_dict'][key],first['actor_state_dict'][key])]
    assert any(key.startswith('mlp.') for key in changed)
    runner.alg.eval_mode()
    with torch.inference_mode():
        fixed=env.get_observations();before=runner.alg.actor(fixed,stochastic_output=False).detach().clone()
    # Prove strict reload restores actual changed model and Adam state.
    with torch.no_grad():next(runner.alg.actor.parameters()).add_(.1)
    next(iter(runner.alg.optimizer.state.values()))['exp_avg'].add_(.1)
    loaded=load_candidate(runner,args.output/'final.pt',config,identity)
    assert_equal(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
    with torch.inference_mode():after=runner.alg.actor(fixed,stochastic_output=False)
    assert torch.equal(before,after)
    assert loaded['inference_buffers_made_writable']
    # Wrong identity/hash are refused before model/optimizer transfer.
    try:load_candidate(runner,args.output/'final.pt',config,{**identity,'source_sha256':'f'*64})
    except ValueError:wrong_source_rejected=True
    else:raise AssertionError('Wrong source was accepted')
    (args.output/'corrupt.pt').write_bytes((args.output/'final.pt').read_bytes()+b'changed')
    (args.output/'corrupt.pt.json').write_bytes((args.output/'final.pt.json').read_bytes())
    try:load_candidate(runner,args.output/'corrupt.pt',config,identity)
    except ValueError:wrong_hash_rejected=True
    else:raise AssertionError('Wrong checkpoint hash was accepted')
    runner.logger.writer.flush()
    report={'complete':True,'passed':True,'kind':'real_installed_RSL_CPU_integration_not_physical_admission','stage2_complete':False,
        'rsl_rl_version':importlib.metadata.version('rsl-rl-lib'),'torch_version':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'device':'cpu','synthetic_environments':32,
        'reproduced_original_exception':reproduced,'initialization':initialization,'actual_PPO_updates':2,
        'actual_rollout_steps':env.steps,'optimizer_state_entries':len(runner.alg.optimizer.state),
        'optimizer_readback_exact':True,'deterministic_action_readback_max_difference':float((before-after).abs().max()),
        'actor_tensors_changed':changed,'inference_buffers_made_writable':loaded['inference_buffers_made_writable'],
        'wrong_source_rejected':wrong_source_rejected,'wrong_hash_rejected':wrong_hash_rejected,
        'initial_checkpoint_sha256':initial['checkpoint_sha256'],'final_checkpoint_sha256':final['checkpoint_sha256'],
        'input_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['candidate_runner.py','velocity_action.py','installed_agent.yaml','real_rsl_cpu_regression.py']}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('REAL_RSL_CPU_REGRESSION '+json.dumps(report),flush=True)

if __name__=='__main__':main()
