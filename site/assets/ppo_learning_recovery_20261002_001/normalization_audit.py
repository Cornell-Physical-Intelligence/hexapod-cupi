"""Measure normalization drift on retained native observations with fixed actor weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output path')
    root = args.repository.resolve()
    sys.path.insert(0, str(root))
    from locomotion.ppo import ppo_config
    import rsl_rl
    assert importlib.metadata.version('rsl-rl-lib') == '5.0.1'
    reference = root/'site/assets/ppo_action_audit_20261002_001/evaluation_comparison.json'
    original = json.loads(reference.read_text())
    records = original['results']['bounded_mean']
    traces, hashes = [], {}
    for i, record in enumerate(records):
        path = args.captures/f'batch_{i:03d}'/'control_trace.npz'
        digest = sha(path)
        assert digest == record['input_sha256']['control_trace.npz']
        hashes[f'batch_{i:03d}/control_trace.npz'] = digest
        with np.load(path, allow_pickle=False) as data:
            trace = torch.from_numpy(data['policy_observation'][:1000, 0].copy())
        assert trace.shape == (1000, 231) and torch.isfinite(trace).all()
        traces.append(trace)
    assert len(traces) == 13
    observations = torch.stack([traces[i % 13] for i in range(128)], dim=1)
    rows = []
    for normalization in ('empirical', 'none'):
        for start in (0, 100, 500):
            torch.manual_seed(20260917)
            config = copy.deepcopy(ppo_config(20260917, action_mean='tanh',
                                              observation_normalization=normalization)['actor'])
            config.pop('class_name')
            actor = MLPModel(TensorDict({'policy': observations[0]}, batch_size=[128]),
                             {'actor': ['policy']}, 'actor', 18, **config)
            parameters = {name: value.clone() for name, value in actor.named_parameters()}
            stored = []
            with torch.no_grad():
                for t in range(start + 24):
                    batch = TensorDict({'policy': observations[t]}, batch_size=[128])
                    actor(batch, stochastic_output=True)
                    if t >= start:
                        stored.append((batch, tuple(x.clone() for x in actor.output_distribution_params)))
                    actor.update_normalization(TensorDict({'policy': observations[t + 1]}, batch_size=[128]))
                drift = []
                for batch, old in stored:
                    mean = actor(batch)
                    new = (mean, actor.distribution.log_std_param.exp().expand_as(mean))
                    drift.append(actor.get_kl_divergence(old, new))
                kl = torch.cat(drift)
            assert all(torch.equal(value, parameters[name]) for name, value in actor.named_parameters())
            rows.append({'observation_normalization': normalization, 'window_start_control': start,
                'controls': 24, 'replicas': 128, 'optimizer_steps': 0,
                'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()),
                'rows_above_kl_reduction_threshold_fraction': float((kl > .02).float().mean())})
    upstream = Path(rsl_rl.__file__).parent
    report = {'schema': 'ppo_fixed_weight_normalization_audit_v1', 'analysis_sha256': sha(__file__),
        'reference_sha256': sha(reference), 'capture_sha256': hashes,
        'source_sha256': {'locomotion/ppo.py': sha(root/'locomotion/ppo.py'),
            **{str(path): sha(upstream/path) for path in ('models/mlp_model.py', 'modules/normalization.py',
                                                       'modules/distribution.py')}},
        'seed': 20260917, 'rsl_rl_version': '5.0.1', 'torch_version': torch.__version__,
        'method': 'Replay the first 1000 stored actor observations from each of 13 completed probes. '
            'Replica i uses probe i modulo 13. Initialize a fresh actor for each case, retain its weights, '
            'update statistics from each following observation, and compare the stored distributions '
            'with the final normalizer on the same inputs. Earlier controls supply normalization history.',
        'scope': 'CPU observation replay. Actions do not drive a simulator, so this is no training rollout '
            'or measurement of native training KL. Nonzero divergence establishes normalization drift '
            'with fixed weights. It does not establish why later native optimizer updates stay small.',
        'native_started': False, 'kl_reduction_threshold': .02, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
