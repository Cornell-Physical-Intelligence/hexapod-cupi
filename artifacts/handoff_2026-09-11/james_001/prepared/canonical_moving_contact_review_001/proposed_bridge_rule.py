"""CPU-only provisional moving contact contract; not adopted/native admission.

Count3 is an explicit proposed experiment termination policy, not an existing
Stage2 threshold or static/dynamic stability guarantee. The real bridge must
run every existing non-contact safety predicate independently.
"""
import numpy as np

SCHEMA = 'canonical_moving_contact_floor3_proposal_v1'


class ContactHoldFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__('Proposed moving contact floor violated; attempted hold retained')


def inspect_hold(before, current, valid, counters, *, prior_counter, phase):
    """All8 accepted samples; command value deliberately cannot select phase.

    Return only after complete source-classified evidence is supplied. Caller
    must independently bind classifier/source identity and preserve raw patches.
    This observes an already attempted20ms hold; it cannot promise to stop at
    its first2.5ms sample without a separately reviewed native callback seam.
    """
    before = np.asarray(before)
    current = np.asarray(current)
    valid = np.asarray(valid)
    counters = np.asarray(counters)
    if current.ndim != 3 or current.shape[1:] != (8, 6) or before.shape != (len(current), 6) or valid.shape != (len(current), 8):
        raise ValueError('Exact prior mask and8 per-environment samples required')
    if len(current) == 0 or any(x.dtype != bool for x in (before, current, valid)) or not valid.all():
        raise ValueError('Missing/invalid source contact evidence is fatal')
    if counters.shape != (8,) or counters.dtype.kind not in 'iu' or not np.array_equal(counters, np.arange(prior_counter + 1, prior_counter + 9)):
        raise ValueError('Wrong exact eight-substep chronology')
    # Only the already admitted actor boundary, never initial settling.
    # Original neutral-window scoring remains entirely in its original scorer.
    if phase == 'neutral_actor_boundary':
        minimum = 6
    elif phase == 'moving_or_stopping':
        minimum = 3
    else:
        raise ValueError('Phase must be explicitly declared; formal quiet scoring stays external')
    counts = current.sum(axis=-1)
    per_sample = counts >= minimum
    bad = np.argwhere(~per_sample.T)  # Earliest time first, then environment.
    previous = np.concatenate([before[:, None], current[:, :-1]], axis=1)
    first = None
    if len(bad):
        sub, env = map(int, bad[0])
        first = {'env': env, 'substep': sub, 'explicit_counter': int(counters[sub]),
                 'count': int(counts[env, sub]), 'mask': current[env, sub].copy()}
    return {'schema': SCHEMA, 'phase': phase, 'proposed_minimum_contact_count': minimum,
            'source_validated_masks': current.copy(), 'support_counts': counts.copy(),
            'per_sample_proposed_rule_ok': per_sample.copy(), 'first_violation': first,
            'force_contact_departures': (previous & ~current).copy(),
            'force_contact_returns': (~previous & current).copy(),
            'contact_rule_allows_commit': bool(per_sample.all()),
            'qualified_lift': None, 'qualified_landing': None,
            'physical_admission': False, 'formal_quality_pass': False}


def enforce_hold(*args, **kwargs):
    """Fail before another action/history/command/storage commit, never ignore false."""
    record = inspect_hold(*args, **kwargs)
    if not record['contact_rule_allows_commit']:
        raise ContactHoldFailure(record)
    return record
