"""Offline CPU diagnostic of the AMP discriminator on the admitted demonstration bank.

You fit the Table III discriminator to the bank against a recorded policy trace
under both gradient-penalty forms and score transitions it never trained on.
A third fit adds standing transitions to the negatives. The report binds its
numbers to the bank, the trace and the source files that produced them.

This is not AMP training and supplies no native evidence. The online learner
trains against the current policy's transitions, which this recorded trace
cannot represent.

    uv run python -m locomotion.amp_diagnostic \\
        --policy-trace locomotion/tests/fixtures/control_trace.npz --output <fresh result.json>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from . import amp, amp_discriminator as disc, amp_ppo

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hexapod_amp_offline_diagnostic_v1'
SOURCE_FILES = ('amp.py', 'amp_discriminator.py', 'amp_ppo.py', 'amp_diagnostic.py')
VELOCITY_FIELDS = ('joint_velocity', 'linear_velocity', 'angular_velocity')
SCOPE = ('Offline CPU fit against one recorded policy trace. It measures the discriminator on fixed data; '
         'it does not measure the online learner, a standing policy or any native behavior.')


@dataclass(frozen=True)
class DiagnosticConfig:
    seed: int = 0
    steps: int = 1500
    batch: int = 256
    learning_rate: float = 1e-4
    gradient_penalty: float = 10.
    hidden: tuple = (1024, 512)
    std_floor: float = 1e-3
    holdout_fraction: float = .1
    evaluation_rows: int = 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def standing(states):
    """Hold each pose with zero joint, linear and angular velocity: s_t equals s_t+1."""
    still = torch.as_tensor(states, dtype=torch.float32).clone()
    for field in amp.feature_contract()['fields']:
        if field['name'] in VELOCITY_FIELDS:
            still[:, slice(*field['slice'])] = 0.
    return disc.transitions(still, still)


def feature_spread(states, floor):
    """Per-field spread of the bank's raw 61-value states."""
    std = torch.as_tensor(states).std(0)
    fields = {field['name']: {'unit': field['unit'], 'min_std': float(std[slice(*field['slice'])].min()),
                              'median_std': float(std[slice(*field['slice'])].median()),
                              'max_std': float(std[slice(*field['slice'])].max())}
              for field in amp.feature_contract()['fields']}
    return {'fields': fields, 'min_std': float(std.min()), 'std_floor': floor,
            'features_below_floor': int((std < floor).sum()), 'features': int(len(std))}


def split(rows, fraction, generator):
    order = torch.randperm(len(rows), generator=generator)
    held = max(1, round(len(rows) * fraction))
    return rows[order[held:]], rows[order[:held]]


def fit(prior, negatives, loss, config):
    """Fit one discriminator; return it with its first and last loss terms."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)
    discriminator = disc.Discriminator(prior.mean(0), prior.std(0).clamp_min(config.std_floor), config.hidden)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    first = last = None
    for _ in range(config.steps):
        prior_rows = torch.randint(len(prior), (config.batch,), generator=generator)
        negative_rows = torch.randint(len(negatives), (config.batch,), generator=generator)
        total, last = loss(discriminator, prior[prior_rows], negatives[negative_rows], config.gradient_penalty)
        first = last if first is None else first
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    return discriminator.eval(), first, last


def style(discriminator, rows):
    with torch.no_grad():
        return float(disc.style_reward(discriminator(rows)).mean())


def diagnose(bank, policy, config=DiagnosticConfig()):
    """Fit the three variants and score held-out bank, policy and standing transitions."""
    generator = torch.Generator().manual_seed(config.seed)
    bank_train, bank_held = split(bank, config.holdout_fraction, generator)
    cut = len(policy) - max(1, round(len(policy) * config.holdout_fraction))
    policy_train, policy_held = policy[:cut], policy[cut:]
    standing_train, standing_held = standing(bank_train[:, :disc.AMP_WIDTH]), standing(bank_held[:, :disc.AMP_WIDTH])
    sets = {'bank_training': bank_train[:config.evaluation_rows], 'bank_held_out': bank_held,
            'policy_trace_held_out': policy_held, 'standing_held_out': standing_held}
    variants = {
        'raw_input_penalty': (disc.discriminator_loss, policy_train),
        'standardized_input_penalty': (amp_ppo.discriminator_loss, policy_train),
        'standardized_input_penalty_with_standing_negatives': (
            amp_ppo.discriminator_loss, torch.cat((policy_train, standing_train))),
    }
    results = {}
    for name, (loss, negatives) in variants.items():
        discriminator, first, last = fit(bank_train, negatives, loss, config)
        results[name] = {'negatives': int(len(negatives)), 'first_update_loss_terms': first, 'last_update_loss_terms': last,
                         'mean_style_reward': {key: style(discriminator, rows) for key, rows in sets.items()}}
    return {'rows': {'bank_training': int(len(bank_train)), 'bank_held_out': int(len(bank_held)),
                     'policy_trace_training': int(len(policy_train)), 'policy_trace_held_out': int(len(policy_held)),
                     'standing_held_out': int(len(standing_held))},
            'bank_feature_spread': feature_spread(bank_train[:, :disc.AMP_WIDTH], config.std_floor),
            'variants': results}


def report(policy_trace, config=DiagnosticConfig(), command=None):
    bank, bank_identity = amp_ppo.load_demonstrations()
    with np.load(policy_trace, allow_pickle=False) as archive:
        before, after = archive['amp_state_before'], archive['amp_state_after']
    if before.ndim != 3 or before.shape[1] != 1 or before.shape != after.shape:
        raise ValueError('Expected a one-robot control trace with AMP states')
    policy = disc.transitions(before[:, 0], after[:, 0])
    source = Path(__file__).resolve().parent
    return {'schema': SCHEMA, 'scope': SCOPE, 'native_evidence': False, 'stage2_complete': False,
            'command': command, 'config': json.loads(json.dumps(asdict(config))),
            'inputs': {'demonstration_bank': bank_identity,
                       'policy_trace': {'path': relative(policy_trace), 'sha256': sha(policy_trace),
                                        'transitions': int(len(policy))}},
            'source_files': {'locomotion/'+name: sha(source/name) for name in SOURCE_FILES},
            'platform': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                         'machine': platform.machine(), 'device': 'cpu'},
            'feature_contract_schema': amp.feature_contract()['schema'],
            'results': diagnose(bank, policy, config)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy-trace', type=Path, required=True, help='Recorded one-robot control_trace.npz.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON report to write.')
    parser.add_argument('--steps', type=int, default=DiagnosticConfig.steps)
    parser.add_argument('--seed', type=int, default=DiagnosticConfig.seed)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('Choose a fresh output; existing evidence is never overwritten')
    if args.steps < 1 or args.seed < 0:
        raise ValueError('Steps must be positive and the seed nonnegative')
    command = ('uv run python -m locomotion.amp_diagnostic --policy-trace ' + relative(args.policy_trace)
               + ' --output ' + relative(args.output) + f' --steps {args.steps} --seed {args.seed}')
    result = report(args.policy_trace, DiagnosticConfig(seed=args.seed, steps=args.steps), command)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: variant['mean_style_reward'] for name, variant in result['results']['variants'].items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
