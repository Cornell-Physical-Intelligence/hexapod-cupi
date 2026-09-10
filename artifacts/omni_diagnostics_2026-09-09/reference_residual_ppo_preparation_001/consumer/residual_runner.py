"""New finite-position-residual RSL lineage, with immutable source-bound checkpoints.

The frozen observation prototype remains unmodified and training-disallowed.
Only this separately identified consumer may run the declared CPU integration
or an explicitly admitted two-update physical smoke. It admits no continuation.
"""
from pathlib import Path
import hashlib
import json
import torch
from checkpoint_compatibility import prepare_algorithm_buffers_for_load,prepare_local_logger_for_early_checkpoint

HERE=Path(__file__).resolve().parent
LINEAGE='c_wave005_finite_position_residual_PPO_smoke_v1'
BINDING_KEYS={'physical_source_sha256','observation_freeze_sha256','device_admission_sha256',
              'directional_admission_sha256','standing_admission_sha256','calibration_sha256','consumer_source_sha256',
              'observation_schema_sha256'}
SCOPES={'synthetic_CPU_runner_integration','admitted_two_update_physical_smoke'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():return json.loads((HERE/'plan.json').read_text())
def contract(scope,bindings):
    if scope not in SCOPES or set(bindings)!=BINDING_KEYS:raise ValueError('Explicit exact new consumer scope/bindings required')
    for key,value in bindings.items():
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact SHA256 binding required: '+key)
    return {'lineage':LINEAGE,'scope':scope,'bindings':dict(bindings),'plan_sha256':digest(HERE/'plan.json'),
        'actor_width':846,'critic_width':849,'actions':18,'action_semantics':'finite_position_residual_goal_0p02_tanh',
        'maximum_updates':2,'old_checkpoint_transfer_allowed':False,'prototype_bundle_training_flags_unchanged':True,
        'native_velocity_fidelity_qualified':False,'stage2_complete':False}

def runner_config(device='cpu'):
    p=plan()
    return {'seed':157,'device':device,'num_steps_per_env':24,'max_iterations':2,'empirical_normalization':{},
        'obs_groups':{'actor':['policy'],'critic':['critic']},'clip_actions':None,'check_for_nan':True,
        'save_interval':1000,'experiment_name':LINEAGE,'run_name':'','logger':'tensorboard',
        'resume':False,'class_name':'OnPolicyRunner',
        'actor':{'class_name':'MLPModel','hidden_dims':[256,256,128],'activation':'elu','obs_normalization':True,
            'distribution_cfg':{'class_name':'GaussianDistribution','init_std':p['initial_exploration_std'],'std_type':'scalar'}},
        'critic':{'class_name':'MLPModel','hidden_dims':[256,256,128],'activation':'elu','obs_normalization':True,'distribution_cfg':None},
        'algorithm':{'class_name':'PPO','num_learning_epochs':5,'num_mini_batches':4,'learning_rate':p['learning_rate'],
            'schedule':'fixed','gamma':.99,'lam':.95,'entropy_coef':0.,'desired_kl':.01,'max_grad_norm':1.,
            'optimizer':'adam','value_loss_coef':1.,'use_clipped_value_loss':True,'clip_param':.2,
            'normalize_advantage_per_mini_batch':False,'share_cnn_encoders':False,'rnd_cfg':None,'symmetry_cfg':None}}

def initialize_scratch(runner):
    layers=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)]
    critic=[m for m in runner.alg.critic.mlp.modules() if isinstance(m,torch.nn.Linear)]
    if layers[0].in_features!=846 or layers[-1].out_features!=18 or critic[0].in_features!=849:
        raise ValueError('New 846/849/18 architecture required')
    if runner.current_learning_iteration!=0 or runner.alg.optimizer.state:raise ValueError('Fresh optimizer and iteration zero required')
    distribution=runner.alg.actor.distribution
    if distribution.std_type!='scalar' or distribution.std_param.shape!=(18,):raise ValueError('Explicit18 Gaussian standard deviations required')
    with torch.no_grad():
        layers[-1].weight.zero_();layers[-1].bias.zero_();distribution.std_param.fill_(plan()['initial_exploration_std'])
    return {'mean_head':'exact_zero','initial_std_per_joint':distribution.std_param.detach().cpu().tolist(),
        'optimizer':'fresh_adam','optimizer_state_entries':0,'learning_rate':runner.alg.learning_rate,
        'schedule':runner.alg.schedule,'entropy_coef':runner.alg.entropy_coef,'physical_exploration_admitted':False}

def save_checkpoint(runner,path,identity):
    path=Path(path);sidecar=path.with_suffix(path.suffix+'.json')
    if path.exists() or sidecar.exists():raise FileExistsError('Checkpoint and sidecar are immutable')
    if identity!=contract(identity.get('scope'),identity.get('bindings',{})):raise ValueError('Malformed new checkpoint identity')
    path.parent.mkdir(parents=True,exist_ok=True);prepare_local_logger_for_early_checkpoint(runner)
    runner.save(str(path),infos={'reference_residual_contract':identity})
    metadata={**identity,'checkpoint_sha256':digest(path)}
    sidecar.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata

def _equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(_equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def reload_checkpoint(runner,path,identity):
    """Only strict same-scope readback/evaluation, never entry into longer training."""
    path=Path(path);sidecar=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if sidecar!={**identity,'checkpoint_sha256':digest(path)}:raise ValueError('Wrong checkpoint bytes, source, scope or schema')
    if identity!=contract(identity.get('scope'),identity.get('bindings',{})):raise ValueError('Malformed consumer identity')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('infos',{}).get('reference_residual_contract')!=identity:raise ValueError('Embedded identity differs from sidecar')
    for name,width in [('actor',846),('critic',849)]:
        state=saved.get(name+'_state_dict',{})
        if state.get('mlp.0.weight',torch.empty(0,0)).shape[1:]!=(width,):raise ValueError('Wrong actor/critic tensor width')
        if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite model or normalizer checkpoint')
    repaired=prepare_algorithm_buffers_for_load(runner.alg)
    runner.load(str(path),strict=True,map_location=runner.device)
    for name in ('actor','critic'):
        if not _equal_tree(getattr(runner.alg,name).state_dict(),saved[name+'_state_dict']):raise RuntimeError('Model/normalizer readback differed')
    if not _equal_tree(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict']):raise RuntimeError('Adam readback differed')
    return {'checkpoint_sha256':sidecar['checkpoint_sha256'],'actor_critic_normalizer_and_optimizer_exact':True,
        'inference_buffers_made_writable':repaired,'longer_training_allowed':False}
