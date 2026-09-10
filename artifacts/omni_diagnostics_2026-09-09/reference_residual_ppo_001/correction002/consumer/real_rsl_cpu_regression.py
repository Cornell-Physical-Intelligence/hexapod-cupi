"""Actual RSL5.0.1 lifecycle, real846/849 encoder/residual core, synthetic physics.

Measured joints/contacts stay in an explicitly artificial standing fixture;
requested targets evolve through the actual bounded residual servo. This tests
runner/state/checkpoint interoperability, never physical quiet or motor safety.
"""
from pathlib import Path
import argparse,copy,importlib.metadata,json,sys
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from residual_runner import *

class CPUVecFixture:
    def __init__(self):
        from fixtures import Rig
        self.rig=Rig(32);self.num_envs=32;self.num_actions=18;self.device='cpu'
        self.max_episode_length=1000;self.episode_length_buf=torch.zeros(32,dtype=torch.long)
        self.cfg={'kind':'synthetic_fixed_measurements_actual_residual_and846849_encoder'}
        self.steps=0;self._encode()
    def _encode(self):
        packet=self.rig.packet()
        # Deliberate raw-rate/angle mismatch remains independently observable.
        packet['measurement']['joint_velocity_rad_s'].fill_(.017)
        self.encoded=self.rig.builder.build(packet)
        assert self.encoded['valid'].all() and self.encoded['policy_training_allowed'] is False
    def get_observations(self):
        return TensorDict({k:self.encoded[k].clone() for k in ('policy','critic')},batch_size=[32])
    def step(self,actions):
        if self.steps>=48:raise RuntimeError('Two24-control updates only')
        from fixtures import pack
        out=self.rig.wave.step(pack([f.snapshot() for f in self.rig.fs]),torch.zeros(32,3,dtype=torch.float64))
        self.rig.c=self.rig.controller.step(out['q_ref'],actions,reference_valid=out['valid'])
        for i,f in enumerate(self.rig.fs):
            f.qtarget=self.rig.c['target_position_rad'][i].detach().numpy().copy()
            f.time+=.02
        self.rig.steps+=1;self.steps+=1;self.episode_length_buf+=1;self._encode()
        # Synthetic nonconstant reward solely gives PPO a gradient for lifecycle
        # verification. The actual physical entrypoint retains frozen rewards.
        r=self.rig.c['residual_position_rad']
        reward=(-1000*r.square().sum(-1)-.01*actions.square().sum(-1)+r[:,0]).float()
        return self.get_observations(),reward,torch.zeros(32,dtype=torch.long),{}

def main():
    p=argparse.ArgumentParser();p.add_argument('--observation-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh CPU output required')
    args.output.mkdir(parents=True);sys.path.insert(0,str(args.observation_bundle.resolve()))
    assert importlib.metadata.version('rsl-rl-lib')=='5.0.1';torch.set_num_threads(1);torch.manual_seed(157)
    env=CPUVecFixture();runner=OnPolicyRunner(env,runner_config(),log_dir=str(args.output/'logs'),device='cpu')
    initial=initialize_scratch(runner);identity=contract('synthetic_CPU_runner_integration',{k:(env.rig.builder.spec()['schema_sha256'] if k=='observation_schema_sha256' else '0'*64) for k in BINDING_KEYS})
    with torch.inference_mode():assert torch.equal(runner.alg.actor(env.get_observations(),stochastic_output=False),torch.zeros(32,18))
    first=save_checkpoint(runner,args.output/'initial.pt',identity)
    assert runner.logger.writer is None
    runner.learn(num_learning_iterations=2,init_at_random_ep_len=False)
    assert env.steps==48 and len(runner.alg.optimizer.state)>0
    final=save_checkpoint(runner,args.output/'final.pt',identity)
    runner.alg.eval_mode()
    with torch.inference_mode():fixed=env.get_observations();before=runner.alg.actor(fixed,stochastic_output=False).clone()
    with torch.no_grad():next(runner.alg.actor.parameters()).add_(.1)
    next(iter(runner.alg.optimizer.state.values()))['exp_avg'].add_(.1)
    loaded=reload_checkpoint(runner,args.output/'final.pt',identity)
    with torch.inference_mode():after=runner.alg.actor(fixed,stochastic_output=False)
    assert torch.equal(before,after)
    rejected=[]
    for changed in ['physical_source_sha256','observation_schema_sha256']:
        bad=copy.deepcopy(identity);bad['bindings'][changed]='f'*64
        try:reload_checkpoint(runner,args.output/'final.pt',bad)
        except ValueError:rejected.append(changed)
        else:raise AssertionError('Wrong checkpoint accepted')
    try:save_checkpoint(runner,args.output/'final.pt',identity)
    except FileExistsError:rejected.append('existing_checkpoint')
    else:raise AssertionError('Immutable checkpoint overwritten')
    (args.output/'orphan.pt.json').write_text('{}')
    try:save_checkpoint(runner,args.output/'orphan.pt',identity)
    except FileExistsError:rejected.append('orphan_sidecar')
    else:raise AssertionError('Existing sidecar ignored')
    assert not (args.output/'orphan.pt').exists()
    initial_state=torch.load(args.output/'initial.pt',weights_only=False,map_location='cpu')
    final_state=torch.load(args.output/'final.pt',weights_only=False,map_location='cpu')
    changed=[k for k,v in final_state['actor_state_dict'].items() if not torch.equal(v,initial_state['actor_state_dict'][k])]
    assert any(k.startswith('mlp.') for k in changed)
    assert env.encoded['interval_rate_valid'].all()
    assert (env.encoded['interval_joint_rate_rad_s']==0).all() and (env.encoded['raw_sdk_joint_velocity_rad_s']==.017).all()
    report={'passed':True,'scope':'real_RSL501_CPU_integration_with_synthetic_fixed_measurements_not_physical_admission',
        'rsl_rl_version':importlib.metadata.version('rsl-rl-lib'),'torch_version':torch.__version__,'device':'cpu',
        'PPO_updates':2,'controls':48,'replicas':32,'actor_width':846,'critic_width':849,
        'initialization':initial,'initial_checkpoint':first,'final_checkpoint':final,'reload':loaded,
        'actor_changed_tensors':changed,'optimizer_state_entries':len(runner.alg.optimizer.state),
        'deterministic_reload_max_difference':float((before-after).abs().max()),'rejected':rejected,
        'rawSDK_and_interval_angle_preserved':True,'Stage2_complete':False,
        'sources':{name:digest(HERE/name) for name in ('real_rsl_cpu_regression.py','residual_runner.py','plan.json','checkpoint_compatibility.py')},
        'observation_freeze_sha256':digest(args.observation_bundle/'FREEZE_SHA256.json')}
    runner.logger.writer.flush();runner.logger.writer.close()
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('REAL_RSL_RESIDUAL_CPU_PASS '+json.dumps({'controls':48,'optimizer_entries':len(runner.alg.optimizer.state),'final_checkpoint':final['checkpoint_sha256']}))
if __name__=='__main__':main()
