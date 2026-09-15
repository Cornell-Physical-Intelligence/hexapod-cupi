"""Prepared bounded CPU comparison. Requires a separately approved REQUEST.json.

No simulator, synthetic observations, native operations or historical source import.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper_walk.learner import Config, PPOLearner, exact_equal, file_sha


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            digest.update(str((x.dtype.str, x.shape)).encode())
            digest.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                digest.update(repr(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            digest.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    obs, actions, states = result['observations'], result['actions'], result['states']
    target = np.stack((-states[:, 37], states[:, 36], states[:, 38]), axis=-1)
    assert obs.shape == (len(actions), 231) and actions.shape[1:] == (18,)
    assert states.shape == (len(obs), 61)
    assert np.array_equal(obs[:, 210:213], result['commands'])
    assert all(np.isfinite(x).all() for x in result.values())
    if 'velocity_targets_navigation_mps' in result:
        assert np.array_equal(target, result['velocity_targets_navigation_mps'])
    result['velocity'] = target
    return result


def ridge_baseline(train, other):
    """Fixed lambda, train-only statistics and fit; no holdout model selection."""
    x = train['observations'][:, :210].astype(np.float64)
    y = train['velocity'].astype(np.float64)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = (x - mean) / scale
    ymean = y.mean(0)
    regularization = 1e-3
    weights = np.linalg.solve(z.T @ z / len(z) + regularization * np.eye(210),
                              z.T @ (y - ymean) / len(z))
    return {
        'lambda': regularization, 'x_mean': mean, 'x_scale': scale,
        'y_mean': ymean, 'weights': weights,
        'train': z @ weights + ymean,
        'heldout': ((other['observations'][:, :210] - mean) / scale) @ weights + ymean,
    }


def velocity_metrics(pred, target, train_mean, ridge):
    pred, target, ridge = (np.asarray(x, np.float64) for x in (pred, target, ridge))
    mse = np.mean((pred - target) ** 2, axis=0)
    zero_mse = np.mean(target ** 2, axis=0)
    mean_mse = np.mean((target - train_mean) ** 2, axis=0)
    ridge_mse = np.mean((target - ridge) ** 2, axis=0)
    pred_std, target_std = pred.std(0), target.std(0)
    corr, ratios = [], []
    for axis in range(3):
        ratios.append(float(pred_std[axis] / target_std[axis]) if target_std[axis] > 1e-10 else None)
        corr.append(float(np.corrcoef(pred[:, axis], target[:, axis])[0, 1])
                    if min(pred_std[axis], target_std[axis]) > 1e-10 else None)
    def skill(baseline):
        return [float(1 - m / b) if b > 1e-20 else None for m, b in zip(mse, baseline)]
    return {
        'mse_per_axis': mse.tolist(), 'rmse_mps': float(np.sqrt(mse.mean())),
        'rmse_mps_per_axis': np.sqrt(mse).tolist(),
        'zero_velocity_mse_per_axis': zero_mse.tolist(),
        'training_mean_velocity_mse_per_axis': mean_mse.tolist(),
        'ridge_velocity_mse_per_axis': ridge_mse.tolist(),
        'skill_vs_zero_per_axis': skill(zero_mse),
        'skill_vs_training_mean_per_axis': skill(mean_mse),
        'prediction_std_mps_per_axis': pred_std.tolist(),
        'target_std_mps_per_axis': target_std.tolist(),
        'prediction_to_target_std_ratio_per_axis': ratios,
        'pearson_r_per_axis': corr,
        'undefined_note': 'Null when baseline error or variance is numerically zero; no zero-lag or correlation hard gate.',
    }


@torch.inference_mode()
def evaluate(learner, data, ridge, ridge_prediction, joint_names):
    obs = torch.from_numpy(data['observations'])
    mean, velocity = learner.model.actor(obs)
    mean, velocity = mean.numpy(), velocity.numpy()
    target, previous = data['actions'], data['observations'][:, 213:]
    masks = {'all': np.ones(len(obs), bool), 'all_zero_command': (data['commands'] == 0).all(-1)}
    if 'source_kind' in data:
        masks.update(original_steady=data['source_kind'] == 0,
                     screened_onset=data['source_kind'] == 1,
                     zero_reset_settle=data['source_kind'] == 2,
                     zero_first_three_seconds=data['zero_phase'] == 0,
                     zero_final_one_second=data['zero_phase'] == 1)
    def measures(mask):
        error = mean[mask] - target[mask]
        requested_delta = target[mask] - previous[mask]
        predicted_delta = mean[mask] - previous[mask]
        cosine_den = np.linalg.norm(requested_delta, axis=-1) * np.linalg.norm(predicted_delta, axis=-1)
        valid = cosine_den > 1e-10
        return {
            'rows': int(mask.sum()), 'actor_action_mse': float(np.mean(error ** 2)),
            'zero_action_mse': float(np.mean(target[mask] ** 2)),
            'copy_previous_action_mse': float(np.mean(requested_delta ** 2)),
            'normalized_increment_mse': float(np.mean((predicted_delta - requested_delta) ** 2)),
            'increment_direction_cosine': float(np.mean(np.sum(requested_delta[valid] * predicted_delta[valid], axis=-1) / cosine_den[valid])) if valid.any() else None,
            'predicted_action_absmax': float(np.max(np.abs(mean[mask]))),
            'raw_action_abs_over_one_fraction': float(np.mean(np.abs(mean[mask]) > 1)),
            'raw_target_step_over_0p04_fraction': float(np.mean(np.abs(predicted_delta) * .35 > .04)),
            'named_joint_action_mse': dict(zip(joint_names, np.mean(error ** 2, axis=0).tolist())),
            'velocity': velocity_metrics(velocity[mask], data['velocity'][mask], ridge['y_mean'], ridge_prediction[mask]),
        }
    result = {'subsets': {key: measures(mask) for key, mask in masks.items() if mask.any()}, 'per_command': []}
    for command in np.unique(data['commands'], axis=0):
        mask = (data['commands'] == command).all(-1)
        result['per_command'].append({'command': command.tolist(), **measures(mask)})
    result['scope'] = 'Actual recorded observations only. Increment MSE is algebraically action MSE; clipping/slew metrics are action predictions, not actual motor torque saturation. No native or closed-loop claim.'
    return result


def main():
    request = json.loads((OUT / 'REQUEST.json').read_text())
    if request.get('root_authorized_execution') is not True:
        raise RuntimeError('Prepared only; missing explicit root source/test approval')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, expected in request['input_files'].items():
        if file_sha(ROOT / name) != expected:
            raise RuntimeError('Changed input bytes: ' + name)
    train, heldout = load_data(request['bc_dataset']), load_data(request['heldout'])
    assert len(train['observations']) == 3760 and 'velocity_targets_navigation_mps' in train
    selection = json.loads((ROOT / 'artifacts/paper_bc_data_002/SELECTION.json').read_text())
    ridge = ridge_baseline(train, heldout)
    np.savez_compressed(OUT / 'RIDGE_BASELINE.npz', **{k: v for k, v in ridge.items() if k not in ('train', 'heldout')})
    initial_old = torch.load(request['fit002_initial_checkpoint'], map_location='cpu', weights_only=False)
    fitted_old = torch.load(request['fit002_checkpoint'], map_location='cpu', weights_only=False)
    common = dict(num_envs=32, initial_std=.1, seed=20260914,
                  learning_rate=1e-4, target_kl=.02, bc_optimizer='separate', bc_learning_rate=3e-4)
    report = {'schema': 'canonical_velocity_supervised_bc_contrast_v1', 'scope': 'CPU only; no native handoff or walking qualification',
              'bc_dataset_sha256': file_sha(request['bc_dataset']), 'amp_prior_sha256': file_sha(request['prior']),
              'learner_sha256': file_sha(ROOT / 'experiments/paper_walk/learner.py'),
              'comparison': 'Joint velocity objective plus BC-only feature detach; separate PPO optimizer/LR/KL configuration has no effect on deterministic BC probe inference.',
              'ridge_scope': 'Train-only 210-history linear regression, fixed normalized-MSE regularization 1e-3; diagnostic baseline, not a recoverability bound or controller.',
              'same_cycle_holdout_note': 'No cycle3 fitting; same trajectory, selection-conditioned temporal neighbor, not independent validation.',
              'no_native_actions': True, 'stage2_complete': False, 'physical_admission': False}
    for name, coefficient, detach in [('legacy_parity', 0., False), ('candidate', 1., True)]:
        config = Config(**common, bc_velocity_coefficient=coefficient, bc_detach_velocity=detach)
        learner = PPOLearner(None, request['prior'], OUT / name, config, device='cpu')
        assert exact_equal(learner.model.state_dict(), initial_old['model'])
        assert exact_equal(learner.amp.state_dict(), initial_old['amp'])
        assert exact_equal(learner._rng(), initial_old['rng'])
        initial_rng_sha = tree_hash(learner._rng())
        initial_receipt = learner.save(OUT / (name + '_initial_checkpoint.pt'))
        before_model = {k: v.clone() for k, v in learner.model.state_dict().items()}
        before_amp = {k: v.clone() for k, v in learner.amp.state_dict().items()}
        started = time.monotonic()
        learner.pretrain_bc(request['bc_dataset'], 1000, 512)
        elapsed = time.monotonic() - started
        assert learner.bc_steps == 1000
        counters = {key: getattr(learner, key) for key in ('updates', 'transitions', 'optimizer_steps', 'discriminator_steps', 'episodes', 'bc_steps')}
        assert all(value == 0 for key, value in counters.items() if key != 'bc_steps')
        assert exact_equal(before_amp, learner.amp.state_dict())
        assert not learner.optimizer.state and not learner.discriminator_optimizer.state
        assert all(torch.equal(before_model[k], v) for k, v in learner.model.state_dict().items()
                   if k.startswith(('critic.', 'critic_normalizer.')) or k == 'log_std')
        if name == 'legacy_parity':
            assert exact_equal(learner.model.state_dict(), fitted_old['model'])
            assert exact_equal(learner.amp.state_dict(), fitted_old['amp'])
            assert exact_equal(learner._rng(), fitted_old['rng'])
        rng_sha = tree_hash(learner._rng())
        checkpoint = OUT / (name + '_checkpoint_update000000.pt')
        receipt = learner.save(checkpoint)
        assert tree_hash(learner._rng()) == rng_sha
        result = {'config': asdict(config), 'fit_seconds': elapsed, 'initial_checkpoint': initial_receipt,
                  'checkpoint': receipt, 'counters': counters, 'fresh_initial_model_amp_rng_exact': True,
                  'initial_rng_tree_sha256': initial_rng_sha, 'post_fit_rng_tree_sha256': rng_sha,
                  'legacy_fitted_model_amp_rng_exact': name == 'legacy_parity',
                  'optimizer_provenance': 'Dedicated BC Adam3e-4 was discarded; PPO Adam1e-4 and D Adam remain empty. BC cannot be resumed from discarded Adam; fresh native reset only.',
                  'normalizer_counts': {'actor': float(learner.model.obs_normalizer.count),
                                        'critic': float(learner.model.critic_normalizer.count), 'amp': float(learner.amp.normalizer.count)},
                  'train': evaluate(learner, train, ridge, ridge['train'], selection['joint_names']),
                  'related_cycle3': evaluate(learner, heldout, ridge, ridge['heldout'], selection['joint_names'])}
        restored = PPOLearner(None, request['prior'], OUT / (name + '_strict_reload'), config, device='cpu')
        restored.load(checkpoint)
        for module in ('model', 'amp', 'optimizer', 'discriminator_optimizer'):
            assert exact_equal(getattr(learner, module).state_dict(), getattr(restored, module).state_dict())
        assert tree_hash(restored._rng()) == rng_sha
        assert torch.equal(learner.act(torch.from_numpy(train['observations'])), restored.act(torch.from_numpy(train['observations'])))
        result['strict_reload_full_dataset_outputs_optimizer_rng_exact'] = True
        report[name] = result
        write('REPORT.json', report)
        print(json.dumps({'phase': name, 'fit_seconds': elapsed, 'checkpoint': receipt,
                          'train': result['train']['subsets']['all']['actor_action_mse'],
                          'velocity_rmse': result['train']['subsets']['all']['velocity']['rmse_mps']}), flush=True)
    for filename, expected in request['input_files'].items():
        assert file_sha(ROOT / filename) == expected
    report.update(completed=True, root_native_handoff_adopted=False)
    write('REPORT.json', report)


if __name__ == '__main__':
    main()
