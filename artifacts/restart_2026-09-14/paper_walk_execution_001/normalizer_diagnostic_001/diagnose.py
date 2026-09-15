"""Offline same-observation normalizer diagnostics; no production state writes."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
A=HERE.parent
REPO=A.parents[2]
sys.path.insert(0,str(REPO))
from experiments.paper_walk.learner import ActorCritic, Config, diagonal_gaussian_kl


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def outputs(model,obs,clamp=True):
    norm=model.obs_normalizer
    z=(obs-norm.mean)/torch.sqrt(norm.var+1e-6)
    x=z.clamp(-10.,10.) if clamp else z
    history=x[:,:210]
    velocity=model.estimator(history)
    memory=model.memory(history)
    mean=model.policy(torch.cat((history[:,-42:],x[:,210:],velocity,memory),-1))
    std=model.log_std.clamp(-5.,1.).exp().expand_as(mean)
    return mean,std,velocity,z,x


def distribution_change(before,after):
    mean,std,velocity,z,x=before
    other,other_std,other_velocity,other_z,other_x=after
    kl=diagonal_gaussian_kl(mean,std,other,other_std)
    return {'mean_kl_old_to_new':float(kl.mean()),'maximum_sample_kl':float(kl.max()),
        'action_mean_rms_change':float((mean-other).square().mean().sqrt()),
        'action_mean_abs_change_max':float((mean-other).abs().max()),
        'estimated_velocity_rms_change_by_axis':(velocity-other_velocity).square().mean(0).sqrt().tolist(),
        'normalizer_preclamp_rms_change':float((z-other_z).square().mean().sqrt()),
        'normalizer_postclamp_rms_change':float((x-other_x).square().mean().sqrt())}


def index_name(index):
    axes=('forward','left','up')
    legs=('lf','lm','lr','rf','rm','rr')
    joints=[leg+'_'+joint for leg in legs for joint in ('coxa_yaw','femur_pitch','tibia_pitch')]
    if index>=213:return 'previous_held_target.'+joints[index-213]
    if index>=210:return 'command.'+('forward','left','yaw')[index-210]
    age=index//42-4;column=index%42
    if column<3:feature='angular_velocity.'+axes[column]
    elif column<6:feature='projected_gravity.'+axes[column-3]
    elif column<24:feature='joint_offset.'+joints[column-6]
    else:feature='joint_velocity.'+joints[column-24]
    return f'history[{age}].'+feature


GROUPS={'older_history':list(range(168)),'current_proprioception':list(range(168,210)),
        'command':list(range(210,213)),'previous_held_target':list(range(213,231))}
for label,start,end in (('all_gyro_history',0,3),('all_gravity_history',3,6),
                       ('all_joint_offset_history',6,24),('all_joint_velocity_history',24,42)):
    GROUPS[label]=[42*age+j for age in range(5) for j in range(start,end)]


def experiment(name,original,changed,obs):
    # Work on in-memory copies only. Neither saved checkpoint nor production weights change.
    assert all(torch.equal(value,dict(changed.named_parameters())[key]) for key,value in original.named_parameters())
    before,after=outputs(original,obs),outputs(changed,obs)
    result={'name':name,'identical_parameters':True,'observations':len(obs),
        'statistics_count_before':float(original.obs_normalizer.count),
        'statistics_count_after':float(changed.obs_normalizer.count),
        'clamped_actor_change':distribution_change(before,after),
        'unclamped_actor_change_diagnostic_only':distribution_change(outputs(original,obs,False),outputs(changed,obs,False)),
        'clamp_effect_on_baseline_actor':distribution_change(outputs(original,obs,False),before),
        'clamp_effect_on_changed_actor':distribution_change(outputs(changed,obs,False),after),
        'feature_groups':{},'top20_transformed_feature_changes':[]}
    for label,indices in GROUPS.items():
        hybrid=copy.deepcopy(original)
        hybrid.obs_normalizer.mean[indices]=changed.obs_normalizer.mean[indices]
        hybrid.obs_normalizer.var[indices]=changed.obs_normalizer.var[indices]
        z0,z1=before[3][:,indices],after[3][:,indices]
        result['feature_groups'][label]={
            'width':len(indices),'before_clipped_elements':int((z0.abs()>10).sum()),
            'after_clipped_elements':int((z1.abs()>10).sum()),
            'preclamp_rms_change':float((z0-z1).square().mean().sqrt()),
            'postclamp_rms_change':float((z0.clamp(-10,10)-z1.clamp(-10,10)).square().mean().sqrt()),
            'isolated_group_stats_change':distribution_change(before,outputs(hybrid,obs)),
            'scope':'Group replacement diagnostic; effects overlap and are not additive.'}
    shift=(before[4]-after[4]).square().mean(0).sqrt()
    for i in shift.argsort(descending=True)[:20].tolist():
        result['top20_transformed_feature_changes'].append({'index':i,'name':index_name(i),
            'raw_subset_mean':float(obs[:,i].mean()),'raw_subset_std':float(obs[:,i].std(unbiased=False)),
            'mean_before':float(original.obs_normalizer.mean[i]),'mean_after':float(changed.obs_normalizer.mean[i]),
            'variance_before':float(original.obs_normalizer.var[i]),'variance_after':float(changed.obs_normalizer.var[i]),
            'postclamp_rms_change':float(shift[i]),
            'before_clipped_samples':int((before[3][:,i].abs()>10).sum()),
            'after_clipped_samples':int((after[3][:,i].abs()>10).sum())})
    return result


def main():
    torch.set_num_threads(1)
    paths={'bc_checkpoint':REPO/'artifacts/paper_bc_fit_004/candidate_checkpoint_update000000.pt',
        'final_checkpoint':A/'results_train_006/standing/learner/checkpoint_000200.pt',
        'recording':A/'results_evaluate_012/standing/evaluation/batch_000/control_trace.npz',
        'recording_report':A/'results_evaluate_012/standing/evaluation/batch_000/report.json',
        'training_metrics':A/'results_train_006/standing/learner/metrics.jsonl',
        'learner_source':REPO/'experiments/paper_walk/learner.py'}
    hashes={key:sha(value) for key,value in paths.items()}
    assert hashes['learner_source']=='b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a'
    bc_saved=torch.load(paths['bc_checkpoint'],map_location='cpu',weights_only=False)
    final_saved=torch.load(paths['final_checkpoint'],map_location='cpu',weights_only=False)
    assert bc_saved['config']==final_saved['config']
    assert bc_saved['learner_sha256']==final_saved['learner_sha256']==hashes['learner_source']
    with np.load(paths['recording'],allow_pickle=False) as recording:
        obs=torch.from_numpy(recording['policy_observation'][:24,0].copy())
        recorded_actions=torch.from_numpy(recording['policy_action'][:24,0].copy())
    report=json.loads(paths['recording_report'].read_text())
    assert report['files']['control_trace.npz']==hashes['recording']
    assert obs.shape==(24,231) and torch.isfinite(obs).all()
    bc=ActorCritic(Config(**bc_saved['config'])).eval()
    final=ActorCritic(Config(**final_saved['config'])).eval()
    bc.load_state_dict(bc_saved['model'],strict=True)
    final.load_state_dict(final_saved['model'],strict=True)
    assert torch.allclose(outputs(bc,obs)[0],recorded_actions,atol=1e-5,rtol=1e-5)
    bc_updated=copy.deepcopy(bc);bc_updated.obs_normalizer.update(obs)
    final_updated=copy.deepcopy(final);final_updated.obs_normalizer.update(obs)
    bc_with_final=copy.deepcopy(bc);bc_with_final.obs_normalizer.load_state_dict(final.obs_normalizer.state_dict(),strict=True)
    final_with_bc=copy.deepcopy(final);final_with_bc.obs_normalizer.load_state_dict(bc.obs_normalizer.state_dict(),strict=True)
    actual_metrics=[json.loads(line) for line in paths['training_metrics'].read_text().splitlines()]
    result={'schema':'same_recorded_observation_normalizer_diagnostic_v1',
        'input_paths':{key:str(value) for key,value in paths.items()},'input_sha256':hashes,
        'source_observations':{'evaluation':'012','case':'learning:translate_0.05_0deg','indices':[0,24],
            'pre_policy_times_s':[0,.46],'controls':24,'environment_rows':1,'unique_samples':24,
            'duplicated_or_reweighted_samples':False,'is_actual_train006_first_rollout':False},
        'baseline_actor_matches_saved_eval012_actions':True,
        'recorded_actual_update2':{k:actual_metrics[1][k] for k in ('update','transitions','normalizer_policy_kl','normalizer_policy_kl_samples','kl_final','ppo_batches')},
        'actual_update2_not_reproduced':'Training update2 used last rollout3072 observations and probed128 current states; neither raw tensor is available locally.',
        'experiments':[
            experiment('BC weights: update BC statistics once with actual24-row diagnostic subset',bc,bc_updated,obs),
            experiment('BC weights: replace BC statistics with actual final200 statistics',bc,bc_with_final,obs),
            experiment('Final200 weights: update final200 statistics once with same24-row subset',final,final_updated,obs),
            experiment('Final200 weights: replace final200 statistics with actual BC statistics',final,final_with_bc,obs)],
        'algorithm_facts':{'BC_count':float(bc.obs_normalizer.count),'final_count':float(final.obs_normalizer.count),
            'actual_update1_rows':128,'actual_update2_rows':3072,
            'update2_new_batch_count_fraction':3072/(float(bc.obs_normalizer.count)+128+3072),
            'diagnostic24_row_BC_count_fraction':24/(float(bc.obs_normalizer.count)+24),
            'normalization':'clamp((x-mean)/sqrt(var+1e-6),-10,+10)',
            'PPO_target_KL_scope':'Model optimizer steps only; between-rollout normalization changes are currently reported, not rejected.'},
        'limitations':['Diagnostic evaluates identical saved observations and identical weights within each experiment.',
            'The24-row evaluation subset is not representative evidence for the unavailable3072 training rollout.',
            'Final-statistics swaps span200 updates and are not a reconstruction of update2.',
            'Removing the clamp is a mathematical comparison only; no unclamped policy ran in physics.',
            'No production parameters, buffers, checkpoint, optimizer, source or configuration changed; no learning-lineage reset.'],
        'stage2_complete':False,'production_changes':False}
    assert hashes=={key:sha(value) for key,value in paths.items()}
    result['input_files_unchanged']=True
    with (HERE/'RESULT.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for row in result['experiments']:
        print(row['name'],json.dumps(row['clamped_actor_change']))


if __name__=='__main__':main()
