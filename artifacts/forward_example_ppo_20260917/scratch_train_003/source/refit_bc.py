"""Explicit CPU BC continuation contrast; no simulator or PPO steps.

The default CLI validates the immutable protocol and inputs. Execution requires
--execute and a fresh output directory. This maintained helper imports only the
maintained learner; a protocol binds the exact helper bytes it authorizes.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import json
import math
import sys
import time
import numpy as np
import torch

SCHEMA = 'canonical_bc_onset_loss_refit_protocol_v1'
LEARNER_SHA = '867e0859b6a66ae9226f5838fd71c553795bfce44e75832d45b2c79504c0518a'
PARENT_SHA = 'dfb6d3ecc6ff5fae30c77bbac9056875eec6f19c9cb129917f97542d3b68d52b'
DATA_SHA = 'f016af1acd6278f37ce3c3c35343554fc59fc258ee9dd39913be4a185a70606a'
AMP_SHA = '22f7b04b0c7540f7a27cbe4d13e23c09dbd3a2363ee0792f292e6293b16b9515'
ARMS = ('uniform', 'onset_weight20')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def weights_for(data, arm):
    require(arm in ARMS,'Unknown contrast arm')
    mask = ((data['control_index']==200) & (data['source_kind']==1)
            & np.any(data['commands']!=0,axis=1))
    require(len(mask)==3760 and int(mask.sum())==23,'Onset selection differs')
    weights = np.ones(len(mask),np.float32)
    if arm=='onset_weight20':
        weights[mask]=20
    return mask,weights


def objective(mean, target, velocity, velocity_target, sampled_weights, global_weight_mean, arm):
    """Uniform draws; only action loss is weighted, by a fixed denominator.

    The uniform arm retains the original two mean-square expressions and
    addition order exactly. The velocity term has coefficient one in both arms.
    """
    require(arm in ARMS,'Unknown contrast arm')
    if arm=='uniform':
        action_loss=(mean-target).square().mean()
    else:
        action_loss=((mean-target).square().mean(-1)*sampled_weights).mean()/global_weight_mean
    velocity_loss=(velocity-velocity_target).square().mean()
    return action_loss+velocity_loss,action_loss,velocity_loss


def prepare(repo, protocol_path):
    repo=Path(repo).resolve()
    protocol=json.loads(Path(protocol_path).read_text())
    require(protocol.get('schema')==SCHEMA,'Protocol schema differs')
    require(protocol['arms']==list(ARMS),'Protocol arms differ')
    expected={'additional_bc_steps':1000,'batch_size':512,'bc_learning_rate':.0003,
              'onset_action_weight':20,'other_action_weight':1,
              'velocity_coefficient':1,'detach_velocity_in_bc_actor':True,
              'normalizers_frozen':True,'sampling':'identical_uniform_with_replacement',
              'global_weight_mean_weighted':4197/3760,'torch_threads':2}
    require(protocol['fit']==expected,'Declared fitting recipe differs')
    require(protocol['helper_sha256']==sha(__file__),'Helper identity differs')
    paths={key:repo/item['path'] for key,item in protocol['inputs'].items()}
    for key,path in paths.items():
        require(sha(path)==protocol['inputs'][key]['sha256'],'Input identity differs: '+key)
    require(sha(paths['learner'])==LEARNER_SHA and sha(paths['parent'])==PARENT_SHA
            and sha(paths['dataset'])==DATA_SHA and sha(paths['amp_prior'])==AMP_SHA,'Fixed lineage differs')
    require(paths['learner'].resolve()==repo/'experiments/paper_walk/learner.py','Maintained learner path required')
    sys.path.insert(0,str(repo))
    module=importlib.import_module('experiments.paper_walk.learner')
    require(Path(module.__file__).resolve()==paths['learner'].resolve(),'Imported learner origin differs')
    parent=torch.load(paths['parent'],map_location='cpu',weights_only=False)
    require(parent['schema']==module.SCHEMA and parent['learner_sha256']==LEARNER_SHA
            and parent['prior_sha256']==AMP_SHA,'Parent strict schema/source/prior differs')
    config=module.Config(**parent['config'])
    config.validate()
    require(config.num_envs==128 and config.bc_optimizer=='separate'
            and config.bc_learning_rate==.0003 and config.bc_velocity_coefficient==1
            and config.bc_detach_velocity is True and config.freeze_actor_obs_normalizer is True
            and config.max_grad_norm==1,'Parent BC configuration differs')
    expected_counters={key:0 for key in ('updates','transitions','optimizer_steps','discriminator_steps','episodes','consecutive_zero_accepted_updates')}
    expected_counters['bc_steps']=1000
    require(parent['counters']==expected_counters,'Only this pre-PPO BC parent is allowed')
    require(not parent['optimizer']['state'] and not parent['discriminator_optimizer']['state'],
            'Expected empty existing PPO/D optimizers')
    require(parent['rng']['cuda']==[],'CPU parent CUDA RNG must remain explicitly empty')
    with np.load(paths['dataset'],allow_pickle=False) as source:
        data={key:source[key].copy() for key in source.files}
    for key,width in (('observations',231),('actions',18),('states',61),('velocity_targets_navigation_mps',3)):
        require(data[key].shape==(3760,width) and data[key].dtype==np.float32
                and np.isfinite(data[key]).all(),'Invalid finite aligned dataset: '+key)
    require(np.array_equal(data['velocity_targets_navigation_mps'],
            data['states'][:,[37,36,38]]*np.array([-1,1,1],np.float32)),'Native pre-hold velocity mapping differs')
    require(np.array_equal(data['observations'][:,210:213],data['commands']),'Command alignment differs')
    require(not np.any(data['cycle_index']==2),'Held-out cycle3 must remain excluded')
    for arm in ARMS:
        mask,weights=weights_for(data,arm)
        require(np.flatnonzero(mask).tolist()==protocol['onset_dataset_rows'],'Declared onset rows differ')
        require(hashlib.sha256(weights.tobytes()).hexdigest()==protocol['weights_sha256'][arm],
                'Declared row weights differ: '+arm)
    return repo,protocol,paths,module,parent,config,data


def preservation(module,parent,learner):
    current=learner.model.state_dict()
    frozen_names=[k for k in parent['model'] if k.startswith(('obs_normalizer.','critic_normalizer.','critic.')) or k=='log_std']
    require(all(torch.equal(current[k],parent['model'][k]) for k in frozen_names),'Frozen model field changed')
    for key in ('amp','optimizer','discriminator_optimizer'):
        require(module.exact_equal(getattr(learner,key).state_dict(),parent[key]),'Preserved state differs: '+key)
    require(module.exact_equal(learner.last_metrics,parent['metrics']),'Existing PPO metrics changed')
    for key,value in parent['counters'].items():
        if key!='bc_steps':
            require(getattr(learner,key)==value,'Non-BC counter changed: '+key)
    return frozen_names


@torch.inference_mode()
def measure(learner,data):
    obs=torch.from_numpy(data['observations'])
    target=torch.from_numpy(data['actions'])
    mean,velocity=learner.model.actor(obs)
    error=(mean-target).double()*.35
    mask,_=weights_for(data,'uniform')
    groups={'all':np.ones(len(mask),bool),'steady':data['source_kind']==0,'first_moving_onset':mask,
            'first_five_moving':(data['control_index']>=200)&(data['control_index']<205)&np.any(data['commands']!=0,axis=1),
            'zero_prefix':data['source_kind']==2}
    return {name:{'rows':int(ids.sum()),'requested_target_rmse_rad':float(error[ids].square().mean().sqrt()),
                  'velocity_rmse_mps':float((velocity[ids]-torch.from_numpy(data['velocity_targets_navigation_mps'][ids])).double().square().mean().sqrt()),
                  'raw_action_clip_fraction':float((mean[ids].abs()>1).float().mean())}
            for name,ids in groups.items()}


def run_pair(prepared, output):
    repo,protocol,paths,module,parent,config,data=prepared
    require(not torch.cuda.is_available(),'This protocol is a CPU-only fit on the local runtime')
    output=Path(output).resolve()
    require(not output.exists(),'Fresh output directory required')
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'INPUT_PROTOCOL.json',protocol)
    results={}
    try:
        for arm in ARMS:
            arm_dir=output/arm
            learner=module.PPOLearner(None,paths['amp_prior'],arm_dir,config,'cpu')
            restored=learner.load(paths['parent'],restore_rng=True)
            require(module.exact_equal(restored,parent),'Strict parent readback differs')
            rng_before=copy.deepcopy(learner._rng())
            require(module.exact_equal(rng_before,parent['rng']),'Parent RNG continuation differs')
            frozen_names=preservation(module,parent,learner)
            before=measure(learner,data)
            _,weights_np=weights_for(data,arm)
            weights=torch.from_numpy(weights_np)
            global_mean=1. if arm=='uniform' else 4197/3760
            obs=torch.from_numpy(data['observations'])
            target=torch.from_numpy(data['actions'])
            velocity_target=torch.from_numpy(data['velocity_targets_navigation_mps'])
            optimizer=torch.optim.Adam(learner.model.parameters(),lr=.0003)
            sampler=hashlib.sha256()
            sampled_counts=np.zeros(3760,np.int64)
            start=time.monotonic()
            with (arm_dir/'refit_metrics.jsonl').open('x') as log:
                for step in range(1000):
                    indices=torch.randint(3760,(512,),device='cpu')
                    sampler.update(indices.numpy().tobytes())
                    np.add.at(sampled_counts,indices.numpy(),1)
                    mean,velocity=learner.model.actor(obs[indices],detach_velocity=True)
                    loss,action_loss,velocity_loss=objective(mean,target[indices],velocity,
                        velocity_target[indices],weights[indices],global_mean,arm)
                    require(bool(torch.isfinite(loss)),'Nonfinite refit loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(learner.model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    learner.bc_steps+=1
                    if (step+1)%50==0:
                        log.write(json.dumps({'additional_bc_step':step+1,'total_bc_steps':learner.bc_steps,
                            'action_objective':float(action_loss.detach()),'velocity_mse_m2_s2':float(velocity_loss.detach()),
                            'total_objective':float(loss.detach()),'unweighted_sample_action_mse':float((mean.detach()-target[indices]).square().mean()),
                            'model_grad_norm_preclip':float(norm),'sampled_onset_rows':int((weights[indices]>1).sum()) if arm!='uniform' else int(np.sum(np.isin(indices.numpy(),protocol['onset_dataset_rows']))),
                            'scope':'Current sampled training batch before optimizer update; no qualification metric'},allow_nan=False)+'\n')
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            require(learner.bc_steps==2000,'BC continuation count differs')
            preservation(module,parent,learner)
            require(all(bool(torch.isfinite(v).all()) for v in learner.model.state_dict().values()),'Nonfinite candidate')
            after=measure(learner,data)
            rng_after=copy.deepcopy(learner._rng())
            require(module.exact_equal(rng_after['python'],rng_before['python'])
                    and module.exact_equal(rng_after['numpy'],rng_before['numpy'])
                    and module.exact_equal(rng_after['cuda'],rng_before['cuda']),'Unexpected non-Torch RNG change')
            receipt=learner.save(arm_dir/'candidate_checkpoint_update000000.pt')
            require(module.exact_equal(learner._rng(),rng_after),'Save changed sampling RNG')
            # Ordinary strict loader, no source bypass or checkpoint rewriting.
            reloader=module.PPOLearner(None,paths['amp_prior'],arm_dir/'strict_reload',config,'cpu')
            reloaded=reloader.load(arm_dir/'candidate_checkpoint_update000000.pt')
            require(module.exact_equal(reloaded['rng'],rng_after),'Saved RNG differs')
            require(module.exact_equal(reloaded['config'],parent['config']),'Parent Config changed')
            with torch.inference_mode():
                require(torch.equal(learner.model.actor(obs)[0],reloader.model.actor(obs)[0]),'All-row strict reload action differs')
                require(torch.equal(learner.model.actor(obs)[1],reloader.model.actor(obs)[1]),'All-row strict reload velocity differs')
            preservation(module,parent,reloader)
            np.save(arm_dir/'sampled_row_counts.npy',sampled_counts,allow_pickle=False)
            results[arm]={'checkpoint':receipt,'additional_bc_steps':1000,'total_bc_steps':2000,
                'seconds':time.monotonic()-start,'before':before,'after':after,
                'minibatch_indices_sha256':sampler.hexdigest(),'sampled_row_counts_sha256':sha(arm_dir/'sampled_row_counts.npy'),
                'final_torch_rng_sha256':hashlib.sha256(rng_after['torch'].numpy().tobytes()).hexdigest(),
                'frozen_model_fields':frozen_names,'preservation_checks_passed':True,
                'normalizer_updates':0,'ppo_steps':0,'discriminator_steps':0,
                'bc_optimizer_discarded':True,'strict_full_dataset_reload_action_velocity_exact':True,
                'stage2_complete':False,'physical_admission':False}
            write_json(arm_dir/'RESULT.json',results[arm])
        require(results[ARMS[0]]['minibatch_indices_sha256']==results[ARMS[1]]['minibatch_indices_sha256'],
                'Paired minibatches differ')
        require(results[ARMS[0]]['final_torch_rng_sha256']==results[ARMS[1]]['final_torch_rng_sha256'],
                'Paired final sampling RNG differs')
        require(results[ARMS[0]]['before']==results[ARMS[1]]['before'],'Paired initial inference differs')
        require(all(sha(path)==protocol['inputs'][key]['sha256'] for key,path in paths.items()),'Input changed during refit')
        write_json(output/'RESULT.json',{'schema':'canonical_bc_onset_loss_refit_result_v1','complete':True,
            'protocol_sha256':sha(output/'INPUT_PROTOCOL.json'),'helper_sha256':sha(__file__),
            'arms':results,'paired_minibatches_exact':True,'inputs_unchanged':True,
            'candidate_selection':'Root review of descriptive fit changes only; native evaluation still required.',
            'simulation_resume_supported':False,'stage2_complete':False,'physical_admission':False})
        return results
    except BaseException as error:
        write_json(output/'FAILURE.json',{'failure':repr(error),'completed_arms':list(results),
            'partial_outputs_preserved':True,'stage2_complete':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Explicitly run both fresh CPU candidates; omit for read-only preflight')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    prepared=prepare(args.repo_root,args.protocol)
    if args.execute:
        require(args.output is not None,'Execution needs a fresh output path')
        print(json.dumps(run_pair(prepared,args.output),indent=2))
    else:
        require(args.output is None,'Read-only preflight does not create output')
        print(json.dumps({'prepared':True,'executed':False,'dataset_rows':3760,'weighted_rows':23,
            'parent_sha256':PARENT_SHA,'learner_sha256':LEARNER_SHA,'helper_sha256':sha(__file__)},indent=2))


if __name__=='__main__':
    main()
