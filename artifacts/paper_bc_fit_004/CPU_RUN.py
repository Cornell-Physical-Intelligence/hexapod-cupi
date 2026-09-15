"""Fresh CPU BC fit with future batch128 metadata; native admission is separate."""
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
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tree_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main():
    request = json.loads((OUT / 'REQUEST.json').read_text())
    if request.get('root_authorized_cpu_execution') is not True:
        raise RuntimeError('Missing root CPU authorization')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, sha in request['input_files'].items():
        assert file_sha(ROOT / name) == sha, 'Changed input: ' + name
    initial32 = torch.load(request['fit003_initial_checkpoint'], map_location='cpu', weights_only=False)
    fitted32 = torch.load(request['fit003_candidate_checkpoint'], map_location='cpu', weights_only=False)
    config = Config(num_envs=128, initial_std=.1, seed=20260914,
                    learning_rate=1e-4, target_kl=.02,
                    bc_optimizer='separate', bc_learning_rate=3e-4,
                    bc_velocity_coefficient=1., bc_detach_velocity=True)
    expected = dict(fitted32['config'])
    assert expected['num_envs'] == 32
    expected['num_envs'] = 128
    assert asdict(config) == expected
    learner = PPOLearner(None, request['prior'], OUT / 'learner', config, device='cpu')
    assert learner.env is None
    for name in ('model', 'amp'):
        assert exact_equal(getattr(learner, name).state_dict(), initial32[name]), 'Initial mismatch: ' + name
    assert exact_equal(learner._rng(), initial32['rng']), 'Initial RNG mismatch'
    initial_rng = tree_hash(learner._rng())
    initial_receipt = learner.save(OUT / 'initial_fresh_checkpoint.pt')
    started = time.monotonic()
    learner.pretrain_bc(request['bc_dataset'], 1000, 512)
    seconds = time.monotonic() - started
    assert learner.bc_steps == 1000
    for name in ('model', 'amp'):
        assert exact_equal(getattr(learner, name).state_dict(), fitted32[name]), 'Fitted mismatch: ' + name
    assert exact_equal(learner._rng(), fitted32['rng']), 'Fitted RNG mismatch'
    assert not learner.optimizer.state and not learner.discriminator_optimizer.state
    assert learner.optimizer.param_groups[0]['lr'] == 1e-4
    counters = {key: getattr(learner, key) for key in ('updates', 'transitions', 'optimizer_steps', 'discriminator_steps', 'episodes', 'bc_steps')}
    assert all(v == 0 for k, v in counters.items() if k != 'bc_steps')
    final_rng = tree_hash(learner._rng())
    checkpoint = OUT / 'candidate_checkpoint_update000000.pt'
    receipt = learner.save(checkpoint)
    assert tree_hash(learner._rng()) == final_rng
    restored = PPOLearner(None, request['prior'], OUT / 'strict_reload', config, device='cpu')
    restored.load(checkpoint)
    for name in ('model', 'amp', 'optimizer', 'discriminator_optimizer'):
        assert exact_equal(getattr(learner, name).state_dict(), getattr(restored, name).state_dict())
    assert tree_hash(restored._rng()) == final_rng
    with np.load(request['bc_dataset'], allow_pickle=False) as data:
        obs = torch.from_numpy(data['observations'].copy())
    assert torch.equal(learner.act(obs), restored.act(obs))
    for name, sha in request['input_files'].items():
        assert file_sha(ROOT / name) == sha, 'Input changed while fitting: ' + name
    report = {
        'schema': 'canonical_fresh_bc_batch128_identity_v1', 'completed': True,
        'scope': 'Fresh CPU fit, future rollout batch128 metadata only; no simulation, native admission or walking qualification',
        'config': asdict(config), 'config_diff_from_fit003': {'num_envs': {'before': 32, 'after': 128}},
        'fit_seconds': seconds, 'torch_threads': 2, 'device': 'cpu', 'env': None,
        'initial_checkpoint': initial_receipt, 'checkpoint': receipt,
        'learner_sha256': file_sha(ROOT / 'experiments/paper_walk/learner.py'),
        'bc_dataset_sha256': file_sha(request['bc_dataset']), 'amp_prior_sha256': file_sha(request['prior']),
        'reference_candidate_sha256': file_sha(request['fit003_candidate_checkpoint']),
        'reference_initial_sha256': file_sha(request['fit003_initial_checkpoint']),
        'initial_model_amp_rng_exact_against_fit003': True,
        'fitted_model_amp_rng_exact_against_fit003': True,
        'initial_rng_tree_sha256': initial_rng, 'fitted_rng_tree_sha256': final_rng,
        'independent_strict_reload_model_amp_optimizers_rng_exact': True,
        'independent_strict_reload_full_bc_observation_actor_exact': True,
        'strict_save_preserves_rng': True, 'counters': counters,
        'optimizer_provenance': 'Fresh dedicated BC Adam3e-4 ran1000 steps and was discarded; PPO Adam1e-4 and discriminator Adam are empty.',
        'normalizer_counts': {'actor': float(learner.model.obs_normalizer.count),
                              'critic': float(learner.model.critic_normalizer.count), 'amp': float(learner.amp.normalizer.count)},
        'diagnostic_reuse_scope': 'Fit003 recorded-input action and estimator diagnostics apply exactly because full fitted model/normalizers and AMP are identical; no new native behavior or independent evaluation is claimed.',
        'fit003_report_sha256': file_sha(request['fit003_report']),
        'existing_checkpoint_rewritten': False, 'checkpoint_metadata_rewritten': False,
        'no_native_actions': True, 'native128_admission_established_here': False,
        'simulation_resume_supported': False, 'stage2_complete': False, 'physical_admission': False,
    }
    write('REPORT.json', report)
    print(json.dumps({'completed': True, 'fit_seconds': seconds, 'checkpoint': receipt,
                      'initial_final_model_amp_rng_exact': True, 'ppo_optimizer_empty': True}), flush=True)


if __name__ == '__main__':
    main()
