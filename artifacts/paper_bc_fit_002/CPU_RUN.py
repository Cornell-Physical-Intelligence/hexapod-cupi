"""Recorded one-off diagnostic; imports maintained learner, never runs physics."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
import time

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
REQUEST = json.loads((OUT / 'REQUEST.json').read_text())
ROOT = Path(REQUEST['root'])
sys.path.insert(0, str(ROOT))
from experiments.paper_walk.learner import Config, PPOLearner, exact_equal, file_sha


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def tree_hash(value):
    """Digest tensor/array bytes with explicit names, shapes, dtypes and scalars."""
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            digest.update(str((x.dtype.str, x.shape)).encode()); digest.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                digest.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            digest.update(type(x).__name__.encode())
            for item in x: visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(learner, path):
    with np.load(path, allow_pickle=False) as data:
        obs = torch.from_numpy(data['observations'].copy())
        target = torch.from_numpy(data['actions'].copy())
        command = data['commands'].copy()
        requested = data['requested_joint_target_rad'].copy()
        applied = data['applied_joint_target_rad'].copy()
    pred = learner.model.actor(obs)[0]
    previous = obs[:, 213:]
    desired_delta, pred_delta = target-previous, pred-previous
    def measures(mask):
        y, yhat, prev = target[mask], pred[mask], previous[mask]
        delta, fitted_delta = desired_delta[mask], pred_delta[mask]
        active = delta.abs() > 1e-5
        mse = float((yhat-y).square().mean())
        copy_mse = float((prev-y).square().mean())
        dot = (fitted_delta*delta).sum(-1)
        norms = fitted_delta.norm(dim=-1)*delta.norm(dim=-1)
        valid_cosine = norms > 1e-10
        return {
            'samples': int(mask.sum()), 'actor_action_mse': mse,
            'zero_action_mse': float(y.square().mean()), 'copy_previous_action_mse': copy_mse,
            'actor_to_copy_previous_mse_ratio': mse/copy_mse if copy_mse > 0 else None,
            'normalized_action_increment_mse': float((fitted_delta-delta).square().mean()),
            'desired_normalized_increment_rms': float(delta.square().mean().sqrt()),
            'fitted_normalized_increment_rms': float(fitted_delta.square().mean().sqrt()),
            'increment_direction_cosine_mean': float((dot[valid_cosine]/norms[valid_cosine]).mean()) if valid_cosine.any() else None,
            'increment_sign_agreement_active_coordinates': float((fitted_delta.sign()[active] == delta.sign()[active]).float().mean()) if active.any() else None,
            'predicted_action_clip_fraction': float((yhat.abs()>1).float().mean()),
            'predicted_unlimited_step_over_0p04_fraction': float((fitted_delta.abs()*.35>.04).float().mean()),
            'per_joint_action_mse': (yhat-y).square().mean(0).tolist(),
        }
    result = measures(torch.ones(len(obs), dtype=torch.bool))
    result['per_command'] = []
    for c in np.unique(command, axis=0):
        mask = torch.from_numpy((command == c).all(-1))
        result['per_command'].append({'command': c.tolist(), **measures(mask)})
    neutral = np.tile(np.array([0., -.3, .4], np.float32), 6)
    result.update(dataset_sha256=file_sha(path), action_min=float(pred.min()), action_max=float(pred.max()),
                  target_reconstruction_max_error_rad=float(np.abs(neutral+.35*target.numpy()-requested).max()),
                  requested_applied_target_max_error_rad=float(np.abs(requested-applied).max()),
                  increment_metric_note='Subtracting the same recorded previous action makes increment MSE algebraically equivalent to action MSE; direction and amplitude are additional diagnostics, not closed-loop motion.',
                  evaluation_scope='Teacher-forced real recorded observations only; no simulated transitions produced.')
    return result


def main():
    torch.set_num_threads(REQUEST['torch_num_threads'])
    torch.set_num_interop_threads(1)
    for name, expected in REQUEST['input_files'].items():
        if file_sha(ROOT/name) != expected:
            raise ValueError('Input bytes changed: '+name)
    config = Config(num_envs=32, initial_std=.1, seed=20260914)
    learner = PPOLearner(None, REQUEST['prior'], OUT/'learner', config, device='cpu')
    before_model = {k: v.clone() for k, v in learner.model.state_dict().items()}
    before_amp = {k: v.clone() for k, v in learner.amp.state_dict().items()}
    initial_rng = tree_hash(learner._rng())
    initial_receipt = learner.save(OUT/'initial_fresh_checkpoint.pt')
    baseline = {name: evaluate(learner, REQUEST[key]) for name, key in [('train', 'bc_dataset'), ('original_steady', 'prior'), ('same_cycle_holdout', 'heldout')]}
    save_json(OUT/'BEFORE.json', baseline)
    started = time.monotonic()
    steps = 0
    def after_step(*_):
        nonlocal steps
        steps += 1
        if steps % 100 == 0:
            with (OUT/'PROGRESS.jsonl').open('a') as stream:
                stream.write(json.dumps({'actual_bc_adam_steps': steps, 'elapsed_s': time.monotonic()-started})+'\n')
    hook = learner.optimizer.register_step_post_hook(after_step)
    try:
        learner.pretrain_bc(REQUEST['bc_dataset'], REQUEST['bc_steps'], REQUEST['batch_size'])
    finally:
        hook.remove()
    fit_seconds = time.monotonic()-started
    assert steps == learner.bc_steps == 1000
    after = {name: evaluate(learner, REQUEST[key]) for name, key in [('train', 'bc_dataset'), ('original_steady', 'prior'), ('same_cycle_holdout', 'heldout')]}
    names = ['updates', 'transitions', 'optimizer_steps', 'discriminator_steps', 'episodes', 'bc_steps']
    assert all(getattr(learner, name) == 0 for name in names[:-1])
    assert exact_equal(before_amp, learner.amp.state_dict())
    assert all(torch.equal(before_model[k], value) for k, value in learner.model.state_dict().items()
               if k.startswith('critic.') or k.startswith('critic_normalizer.') or k == 'log_std')
    checkpoint = OUT/'bc_checkpoint_update000000.pt'
    before_save_rng = tree_hash(learner._rng())
    receipt = learner.save(checkpoint)
    assert before_save_rng == tree_hash(learner._rng())
    opt_state = learner.optimizer.state_dict()
    adam_steps = {int(s['step']) for s in opt_state['state'].values()}
    assert adam_steps == {1000}
    report = {
        'schema': 'canonical_offline_bc_fit_diagnostic_v1', 'completed': True,
        'scope': 'Offline imitation fit on original steady, screened onset and actual zero reset/settle native rows; no walking or native handoff claim',
        'config': asdict(config), 'cpu_threads': torch.get_num_threads(), 'device': 'cpu',
        'fit_seconds': fit_seconds, 'config_num_envs_is_metadata_no_env_constructed': True,
        'initial_checkpoint': initial_receipt, 'checkpoint': receipt,
        'learner_sha256': file_sha(ROOT/'experiments/paper_walk/learner.py'),
        'counters': {name: getattr(learner, name) for name in names},
        'before': baseline, 'after': after,
        'optimizer_provenance': {
            'fresh_initialization': True, 'shared_ppo_adam_used': True,
            'bc_actor_parameter_state_count': len(opt_state['state']), 'adam_steps_per_updated_parameter': sorted(adam_steps),
            'ppo_optimizer_steps_counter_scope': 'PPO updates only:0; BC Adam steps are separately recorded as1000',
            'discriminator_optimizer_state_count': len(learner.discriminator_optimizer.state_dict()['state']),
            'critic_and_log_std_unchanged': True, 'discriminator_and_amp_normalizer_unchanged': True,
            'updated_modules': ['estimator', 'memory', 'policy'], 'estimator_velocity_supervised_during_bc': False,
            'handoff_requires_explicit_decision_about_inheriting_adam_moments': True,
        },
        'normalizer_counts': {'actor': float(learner.model.obs_normalizer.count),
                              'critic': float(learner.model.critic_normalizer.count), 'amp': float(learner.amp.normalizer.count)},
        'rng': {'seed': 20260914, 'initial_rng_tree_sha256': initial_rng,
                'post_bc_rng_tree_sha256': before_save_rng, 'strict_save_preserves_rng': True,
                'full_python_numpy_torch_states_stored_in_checkpoints': True},
        'same_cycle_holdout_note': 'Not used for fitting; same target trajectory and selection-conditioned temporal neighbor, not independent generalization.',
        'no_native_actions': True, 'onset_or_stop_examples_added': 'Screened onset and genuine zero reset/settle only; no stop transitions', 'bc_dataset_sha256': file_sha(REQUEST['bc_dataset']), 'amp_prior_sha256': file_sha(REQUEST['prior']), 'amp_prior_unchanged': True,
        'physical_admission': False, 'stage2_complete': False, 'native_handoff_adopted': False,
    }
    restored = PPOLearner(None, REQUEST['prior'], OUT/'reload_verification', config, device='cpu')
    restored.load(checkpoint)
    for name in ('model', 'amp', 'optimizer', 'discriminator_optimizer'):
        assert exact_equal(getattr(restored, name).state_dict(), getattr(learner, name).state_dict())
    assert tree_hash(restored._rng()) == before_save_rng
    with np.load(REQUEST['bc_dataset'], allow_pickle=False) as data:
        obs = torch.from_numpy(data['observations'].copy())
    assert torch.equal(learner.act(obs), restored.act(obs))
    report['independent_strict_reload_full_actor_dataset_exact'] = True
    report['independent_strict_reload_rng_exact'] = True
    for name, expected in REQUEST['input_files'].items():
        assert file_sha(ROOT/name) == expected
    save_json(OUT/'REPORT.json', report)
    print(json.dumps({'completed': True, 'checkpoint': receipt, 'fit_seconds': fit_seconds,
                      'train_mse': after['train']['actor_action_mse'],
                      'heldout_mse': after['same_cycle_holdout']['actor_action_mse'],
                      'copy_previous_mse': after['train']['copy_previous_action_mse']}), flush=True)


if __name__ == '__main__':
    main()
