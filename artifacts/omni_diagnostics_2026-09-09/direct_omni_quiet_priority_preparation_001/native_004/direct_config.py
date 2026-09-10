"""Explicit launch selection for one immutable direct315 experiment source."""
import copy

SCHEMA = 'direct315_quiet_priority_native_v3'
SMOKE_BRANCH = 'quiet_priority'
CHECKPOINT = '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
ALLOCATIONS = {'smoke': {'replicas': 32, 'controls_per_update': 24, 'updates': 2},
               'pilot': {'replicas': 1024, 'controls_per_update': 24, 'updates': 50}}

def protocol():
    return {'schema': SCHEMA, 'allocations': copy.deepcopy(ALLOCATIONS),
            'branches': ['curriculum', 'caps', 'quiet_priority'], 'smoke_branch': SMOKE_BRANCH, 'quiet_priority_objective': {'quiet_temporal_weight':1.0,'moving_temporal_weight':0.1,'spatial_weight':0.1,'denominator':'all_valid_pairs'}, 'optimizer_diagnostics': {'minibatches_per_update':20,'sparse_gradient_updates':[1,10,25,50],'sparse_minibatches':[1,20]}, 'checkpoint_sha256': CHECKPOINT,
            'command_schedule': '25percent_quiet_other_rows_8s_motion_8s_stop',
            'evaluations': ['constant', 'stop'], 'automatic_continuation': False}

def selection(mode, allocation, branch, evaluation, iterations):
    if mode == 'train':
        if allocation not in ALLOCATIONS or branch not in ('curriculum', 'caps', 'quiet_priority') or evaluation is not None:
            raise ValueError('Training requires explicit allocation and branch only')
        result = {'schema': SCHEMA, 'allocation': allocation, 'branch': branch, **ALLOCATIONS[allocation]}
        if iterations is not None and (type(iterations) is not int or iterations != result['updates']):
            raise ValueError('Iteration override differs from bounded allocation')
        result['caps'] = {'temporal_weight': .1 if branch != 'curriculum' else 0.,
                          'spatial_weight': .1 if branch != 'curriculum' else 0., 'noise_scale': 1., 'noise_seed': 1157, 'quiet_temporal_weight':1.0 if branch=='quiet_priority' else (.1 if branch=='caps' else 0.)}
        return result
    if allocation is not None or branch is not None or iterations is not None:
        raise ValueError('Nontraining phase cannot request allocation/branch/iterations')
    if mode == 'validate' and evaluation is None:
        return {'schema': SCHEMA, 'evaluation': None, 'replicas': 32}
    if mode == 'evaluate' and evaluation in ('constant', 'stop'):
        return {'schema': SCHEMA, 'evaluation': evaluation, 'replicas': 48}
    raise ValueError('Only explicit standing, train, constant or stop diagnostic phases exist')

def configure(cfg, selected):
    from caps import validated_options
    expected = selection('train', selected.get('allocation'), selected.get('branch'), None, None)
    if selected != expected:
        raise ValueError('Training selection changed')
    result = copy.deepcopy(cfg)
    if result['obs_groups'] != {'actor': ['policy'], 'critic': ['critic']} or result.get('clip_actions') is not None:
        raise ValueError('Exact 315/318 observation groups and unclipped wrapper required')
    result.update(num_steps_per_env=24, save_interval=1, max_iterations=selected['updates'])
    result['algorithm'].update(class_name='caps_ppo:CapsPPO', caps_options=validated_options(selected['caps']), gradient_diagnostics=True)
    if (result['algorithm']['num_learning_epochs'],result['algorithm']['num_mini_batches'],result['algorithm']['schedule'],result['algorithm']['desired_kl'])!=(5,4,'adaptive',.01):
        raise ValueError('Exact optimizer cadence/adaptive KL required')
    return result
