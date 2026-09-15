"""Independent toy math, not an implementation change or native PPO experiment."""
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
from torch.distributions import Normal, Independent, kl_divergence

OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(734)
np.random.seed(734)
random.seed(734)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def formula(m0, s0, m1, s1):
    return (torch.log(s1/s0) + (s0.square()+(m0-m1).square())/(2*s1.square()) - .5).sum(-1).mean()


def reference(m0, s0, m1, s1):
    return kl_divergence(Independent(Normal(m0, s0), 1), Independent(Normal(m1, s1), 1)).mean()


def main():
    m0 = torch.randn(64, 18, dtype=torch.float64)
    m1 = torch.randn(64, 18, dtype=torch.float64)
    s0 = torch.rand(64, 18, dtype=torch.float64)+.1
    s1 = torch.rand(64, 18, dtype=torch.float64)+.1
    error = abs(float(formula(m0, s0, m1, s1)-reference(m0, s0, m1, s1)))
    assert error < 1e-12
    assert float(formula(m0, s0, m0, s0)) == 0.

    # Arbitrary high-sensitivity linear actor. Gain150 is NOT measured on the robot.
    parameter = torch.nn.Parameter(torch.zeros(18, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    optimizer.param_groups[0]['lr'] = 1e-4
    sigma = torch.full((64, 18), .1, dtype=torch.float64)
    mean = lambda: (150*parameter.detach()).expand(64, -1)
    original_reference = mean().clone()
    bound = .02
    attempts = []
    starting_rng = rng()

    def attempt_update(initial_lr, max_attempts=8, force_reject=False):
        snapshot_parameter = parameter.detach().clone()
        snapshot_optimizer = copy.deepcopy(optimizer.state_dict())
        immutable_snapshot = copy.deepcopy(snapshot_optimizer)
        accepted = False
        for retry in range(max_attempts):
            with torch.no_grad():
                parameter.copy_(snapshot_parameter)
            # load_state_dict can share tensor storage with its input. Restore a
            # new deep copy each attempt, preserving the immutable snapshot.
            optimizer.load_state_dict(copy.deepcopy(snapshot_optimizer))
            assert torch.equal(parameter, snapshot_parameter)
            assert equal(optimizer.state_dict(), snapshot_optimizer)
            optimizer.param_groups[0]['lr'] = initial_lr*(.5**retry)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            kl = float(reference(original_reference, sigma, mean(), sigma))
            accepted = kl <= bound and not force_reject
            attempts.append({'retry': retry, 'lr': optimizer.param_groups[0]['lr'],
                             'full_reference_kl': kl, 'accepted': accepted,
                             'adam_step': int(optimizer.state[parameter]['step'])})
            assert equal(snapshot_optimizer, immutable_snapshot)
            assert equal(rng(), starting_rng)
            if accepted:
                # Independently apply exactly one Adam step at accepted LR.
                control = torch.nn.Parameter(snapshot_parameter.clone())
                control_optimizer = torch.optim.Adam([control], lr=initial_lr)
                control_optimizer.load_state_dict(copy.deepcopy(snapshot_optimizer))
                control_optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
                control.grad = torch.ones_like(control)
                control_optimizer.step()
                assert torch.equal(parameter, control)
                assert equal(optimizer.state_dict(), control_optimizer.state_dict())
                break
        if not accepted:
            with torch.no_grad():
                parameter.copy_(snapshot_parameter)
            optimizer.load_state_dict(copy.deepcopy(snapshot_optimizer))
            assert torch.equal(parameter, snapshot_parameter)
            assert equal(optimizer.state_dict(), snapshot_optimizer)
        return accepted

    assert attempt_update(1e-4)
    first_accepted = attempts[-1].copy()
    last_accepted_mean = mean().clone()
    assert attempt_update(first_accepted['lr'])
    second_accepted = attempts[-1].copy()
    assert second_accepted['full_reference_kl'] <= bound
    assert float(reference(last_accepted_mean, sigma, mean(), sigma)) < bound
    before_exhaustion = (parameter.detach().clone(), copy.deepcopy(optimizer.state_dict()))
    assert not attempt_update(1e-4, max_attempts=2, force_reject=True)
    assert torch.equal(parameter, before_exhaustion[0]) and equal(optimizer.state_dict(), before_exhaustion[1])
    assert equal(rng(), starting_rng)
    result = {
        'schema': 'independent_optimizer_math_fixture_v1', 'passed': True,
        'scope': 'Synthetic fixed-gradient Adam and Gaussian fixture only; no actual learner update, native rollout, or source change.',
        'formula_max_absolute_error_against_torch_independent_gaussian_kl': error,
        'arbitrary_toy_actor_gain': 150, 'sigma': .1, 'action_dimensions': 18,
        'target_empirical_full_reference_kl': bound, 'attempts': attempts,
        'first_accepted': first_accepted, 'second_accepted': second_accepted,
        'rejected_parameters_and_all_adam_state_restored': True,
        'accepted_state_exactly_matches_one_clean_adam_step': True,
        'snapshot_not_mutated_through_restore_aliasing': True,
        'retries_preserve_python_numpy_torch_rng': True,
        'retry_exhaustion_restores_last_accepted_state': True,
        'discriminator_not_involved': True,
        'limitations': ['Gain and gradient are synthetic, not calibrated to the robot.',
            'Future production code still needs full-model, critic/estimator, discriminator schedule and zero-step integration tests.',
            'Empirical mean KL on fixed recorded observations does not bound unseen states or physical outcomes.',
            'Reducing LR only in the next update cannot undo a step already accepted above the target.',
            'Reference must remain collection-time policy across all accepted steps; per-step bounds do not imply total bound.',
            'Normalizer changes are independent of this fixed-statistics optimizer fixture.'],
    }
    (OUT/'MATH_CHECK.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'first': first_accepted, 'second': second_accepted, 'attempts': len(attempts)}))


if __name__ == '__main__':
    main()
