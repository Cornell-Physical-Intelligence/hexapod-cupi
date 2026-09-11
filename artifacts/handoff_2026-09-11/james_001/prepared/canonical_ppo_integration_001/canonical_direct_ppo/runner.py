"""Fresh-model bounded PPO collection/update; no native startup or legacy actor import."""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import torch
from .smoke_config import NUM_ENVS,ACTOR_WIDTH,CRITIC_WIDTH,ACTION_WIDTH,UPDATES,CONTROLS_PER_UPDATE,SEED,protocol,runner_config


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def clone_tree(x):
    if torch.is_tensor(x):return x.detach().clone()
    if isinstance(x,np.ndarray):return x.copy()
    if isinstance(x,dict):return {k:clone_tree(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(clone_tree(v)for v in x)
    return copy.deepcopy(x)


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(equal_tree(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(equal_tree(x,y)for x,y in zip(a,b))
    return a==b


def finite_tensor(value,shape,name):
    if not torch.is_tensor(value)or tuple(value.shape)!=tuple(shape)or not bool(torch.isfinite(value).all()):raise ValueError('Malformed/nonfinite '+name)
    return value


def checked_observation(observation):
    # Clone at the boundary: RSL keeps the pre-step observation until process_env_step.
    from tensordict import TensorDict
    if set(observation.keys())!={'policy','critic'}:raise ValueError('Wrong observation groups')
    actor=finite_tensor(observation['policy'],(NUM_ENVS,ACTOR_WIDTH),'actor observation')
    critic=finite_tensor(observation['critic'],(NUM_ENVS,CRITIC_WIDTH),'critic observation')
    if not torch.equal(actor,critic[:,:ACTOR_WIDTH]):raise ValueError('Critic must append only the privileged velocity')
    return TensorDict({'policy':actor.detach().clone(),'critic':critic.detach().clone()},batch_size=[NUM_ENVS])


def construct(env,observation,device):
    from .rsl_binding import verify
    verify()
    from rsl_rl.algorithms import PPO
    if env.num_envs!=NUM_ENVS or env.num_actions!=ACTION_WIDTH:raise ValueError('Wrong bounded environment size')
    with torch.inference_mode(False):
        algorithm=PPO.construct_algorithm(observation,env,runner_config(),device)
        actor_linears=[m for m in algorithm.actor.mlp.modules()if isinstance(m,torch.nn.Linear)]
        if not actor_linears or actor_linears[-1].out_features!=ACTION_WIDTH:raise ValueError('Unexpected actor output layer')
        with torch.no_grad():actor_linears[-1].weight.zero_();actor_linears[-1].bias.zero_()
    return algorithm


def strict_reload(algorithm,env,observation,path,expected_lineage,completed_updates):
    saved=torch.load(path,map_location=algorithm.device,weights_only=False)
    if saved.get('schema')!=protocol()['schema']or saved.get('lineage')!=expected_lineage or saved.get('completed_updates')!=completed_updates:raise ValueError('Checkpoint lineage/iteration mismatch')
    if saved.get('controls_completed')!=completed_updates*CONTROLS_PER_UPDATE:raise ValueError('Checkpoint control count mismatch')
    before=clone_tree(algorithm.save());rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()if torch.cuda.is_available()else None
    try:
        with torch.inference_mode():expected=algorithm.actor(observation).clone()
        # Construct ordinary destination buffers. Never in-place-copy into the
        # inference tensors created by RSL normalizer updates, and never alter originals.
        with torch.inference_mode(False),torch.no_grad():
            restored=construct(env,observation,algorithm.device)
            if any(torch.is_inference(v)for m in (restored.actor,restored.critic)for v in m.buffers()):raise ValueError('Reload destination contains inference buffers')
            restored.load(saved['algorithm'],None,strict=True)
            restored.learning_rate=float(saved['learning_rate'])
        if not equal_tree(before,restored.save())or not equal_tree(saved['algorithm'],restored.save()):raise ValueError('Strict model/normalizer/Adam reload differs')
        if any(float(g['lr'])!=restored.learning_rate for g in restored.optimizer.param_groups):raise ValueError('Adaptive learning rate differs from saved optimizer')
        with torch.inference_mode():actual=restored.actor(observation)
        if not torch.equal(expected,actual):raise ValueError('Deterministic reloaded action differs')
        if not equal_tree(saved['runtime_state'],env.export_runtime_state()):raise ValueError('Saved adapter/session state differs')
        return {'passed':True,'strict_model_normalizer_optimizer':True,'deterministic_action_exact':True,
                'runtime_state_exact':True,'learning_rate_exact':True,'optimizer_entries':len(restored.optimizer.state),
                'checkpoint_sha256':sha(path),'reload_method':'new ordinary-buffer algorithm; original inference buffers untouched'}
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)


def learn_smoke(env,lineage,output,*,device):
    """Caller validates native admission/bindings first; env.step preserves raw failures."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':protocol()['schema'],'lineage':lineage,'protocol':protocol(),'status':'running','updates_completed':0,
             'controls_completed':0,'controls_attempted':0,'optimizer_steps_completed':0,'updates':[],'errors':[],
             'Stage2_complete':False,'quality_admitted':False,'no_input_checkpoint':True}
    receipt['torch_version']=torch.__version__
    save(output/'state.json',receipt);algorithm=None;original_step=None;started=time.monotonic()
    try:
        torch.manual_seed(SEED);np.random.seed(SEED)
        observation=checked_observation(env.get_observations()).to(device)
        algorithm=construct(env,observation,device)
        if len(algorithm.optimizer.state)!=0:raise ValueError('Fresh optimizer already contains state')
        with torch.inference_mode():
            if not bool((algorithm.actor(observation)==0).all()):raise ValueError('Fresh mean action is not exactly zero')
        initial=clone_tree(dict(algorithm.actor.named_parameters()));algorithm.train_mode()
        original_step=algorithm.optimizer.step
        def observed_step(*args,**kwargs):
            value=original_step(*args,**kwargs);receipt['optimizer_steps_completed']+=1;return value
        algorithm.optimizer.step=observed_step
        for update in range(1,UPDATES+1):
            before_steps=receipt['optimizer_steps_completed']
            with torch.inference_mode():
                for _ in range(CONTROLS_PER_UPDATE):
                    action=finite_tensor(algorithm.act(observation),(NUM_ENVS,ACTION_WIDTH),'sampled action')
                    # Native bridge stores command/target/readback before any failure.
                    receipt['controls_attempted']+=1
                    next_observation,reward,done,extras=env.step(action.to(env.device))
                    receipt['controls_completed']+=1
                    reward=finite_tensor(reward,(NUM_ENVS,),'reward').to(device)
                    done=finite_tensor(done,(NUM_ENVS,),'done').to(device)
                    if bool(done.any())or bool(extras.get('time_outs',torch.zeros_like(done)).any()):raise RuntimeError('No-auto-reset smoke observed termination or timeout')
                    observation=checked_observation(next_observation).to(device)
                    algorithm.process_env_step(observation,reward,done,{})
                algorithm.compute_returns(observation)
            losses=algorithm.update()
            if receipt['optimizer_steps_completed']-before_steps!=20:raise ValueError('Expected exactly20 optimizer minibatches/update')
            losses={k:float(v)for k,v in losses.items()}
            if not all(np.isfinite(list(losses.values()))):raise ValueError('Nonfinite PPO losses')
            receipt['updates_completed']=update
            receipt['updates'].append({'completed_update':update,'optimizer_minibatches':20,'losses':losses,'learning_rate':float(algorithm.learning_rate)})
            checkpoint=output/f'decision_{update:03d}.pt'
            if checkpoint.exists():raise FileExistsError(checkpoint)
            saved={'schema':protocol()['schema'],'lineage':lineage,'protocol':protocol(),
                   'completed_updates':update,'controls_completed':receipt['controls_completed'],'algorithm':algorithm.save(),
                   'learning_rate':float(algorithm.learning_rate),'runtime_state':env.export_runtime_state(),
                   'torch_rng_cpu':torch.get_rng_state(),'torch_rng_cuda':torch.cuda.get_rng_state_all()if torch.cuda.is_available()else []}
            torch.save(saved,checkpoint)
            receipt['updates'][-1]['checkpoint_sha256']=sha(checkpoint)
            receipt['updates'][-1]['reload']=strict_reload(algorithm,env,observation,checkpoint,lineage,update)
            save(output/'state.json',receipt)
        if receipt['controls_completed']!=48 or receipt['optimizer_steps_completed']!=40:raise ValueError('Wrong bounded collection/update counts')
        if equal_tree(initial,dict(algorithm.actor.named_parameters())):raise ValueError('No actor parameter changed')
        receipt['status']='completed';receipt['transitions']=48*NUM_ENVS
    except BaseException as error:
        receipt['status']='failed';receipt['errors'].append(repr(error))
        raise
    finally:
        if original_step is not None:algorithm.optimizer.step=original_step
        receipt['wall_s']=time.monotonic()-started
        try:receipt['native_audit']=env.export_audit(output)
        except BaseException as error:
            receipt['status']='failed';receipt['errors'].append('export_audit: '+repr(error));save(output/'state.json',receipt);raise
        receipt['outputs']={p.name:sha(p)for p in output.iterdir()if p.is_file()and p.name!='state.json'and not p.name.endswith('.part')}
        save(output/'state.json',receipt)
    return receipt
