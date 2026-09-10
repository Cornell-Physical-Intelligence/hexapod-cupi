"""Explicit reset-only sensor epochs and bounded per-row canonical recovery."""
from contextlib import contextmanager
from types import MethodType
import torch


@contextmanager
def defer_training_resets(env, capture):
    """Call original done predicate once; retain flags before delaying row reset.

    Training only. The caller must classify each captured outcome and invoke
    the original `_reset_idx` before the next action for ended rows. Evaluation
    never uses this adapter and retains immediate failure/admission behavior.
    """
    original = env._get_dones
    def wrapped(instance):
        term, trunc = original()
        term = term.torch if hasattr(term, 'torch') else term
        trunc = trunc.torch if hasattr(trunc, 'torch') else trunc
        if term.dtype != torch.bool or trunc.dtype != torch.bool:
            raise ValueError('Original done flags must be boolean')
        capture(term.clone(), trunc.clone())
        return torch.zeros_like(term), torch.zeros_like(trunc)
    env._get_dones = MethodType(wrapped, env)
    try:
        yield
    finally:
        env._get_dones = original


def rebase_reset_clocks(freshness, selected):
    """Read real reset timestamps; do not fabricate freshness or refresh data.

    The installed SensorBase.reset kernel clears current/last clocks to zero
    and marks only selected rows outdated. On the next ordinary sensor access
    after8 physical updates, the existing reader validates the real recurrence.
    """
    if selected.shape != (freshness.n,) or selected.dtype != torch.bool:
        raise ValueError('Explicit per-row reset mask required')
    current = torch.stack([freshness._tensor(s._timestamp, torch.float32) for s in freshness.sensors], -1)
    last = torch.stack([freshness._tensor(s._timestamp_last_update, torch.float32) for s in freshness.sensors], -1)
    outdated = torch.stack([freshness._tensor(s._is_outdated, torch.bool) for s in freshness.sensors], -1)
    if not torch.isfinite(current).all() or not torch.isfinite(last).all():
        raise ValueError('Nonfinite reset clocks')
    if ((current[selected] != 0) | (last[selected] != 0) | ~outdated[selected]).any():
        raise ValueError('Selected sensor clocks do not match installed reset kernel')
    if not torch.equal(current[~selected], freshness.previous[~selected]) or not freshness.ready[~selected].all():
        raise ValueError('Unselected sensor clocks changed during reset')
    freshness.previous[selected] = current[selected]
    freshness.ready[selected] = True  # Expected new epoch, NOT a valid observation.
    return {'reset_mask': selected.clone(), 'timestamp': current, 'last_update': last,
            'outdated': outdated, 'measurement_admitted': False}


class RecoveryTargets:
    """Device-native exact source009 2s quintic plus2s settling, by row."""
    def __init__(self, initial, nominal, limits):
        if initial.shape != nominal.shape or initial.ndim != 2 or initial.shape[1] != 18 or limits.shape != (*initial.shape,2):
            raise ValueError('Named Nx18 reset, nominal and joint limits required')
        self.start=initial.to(torch.float64).clone();self.nominal=nominal.to(torch.float64).clone()
        self.limits=limits.to(torch.float64).clone()
        self.age=torch.zeros(len(initial),device=initial.device,dtype=torch.long)
        self.warming=torch.ones(len(initial),device=initial.device,dtype=torch.bool)
        self.reset(torch.ones_like(self.warming),initial)

    def reset(self, selected, initial):
        q=initial.to(torch.float64)
        if q.shape != self.start.shape or not torch.isfinite(q).all():raise ValueError('Finite complete reset target')
        for endpoint in (q,self.nominal):
            if ((endpoint[selected] < self.limits[selected,:,0]+.02) | (endpoint[selected] > self.limits[selected,:,1]-.02)).any():
                raise ValueError('Reset/nominal target lacks residual joint margin')
        self.start[selected]=q[selected];self.age[selected]=0;self.warming[selected]=True

    def _position(self, age):
        u=(age.to(torch.float64)/100).clamp(0,1)[:,None]
        q=self.start+(self.nominal-self.start)*(u*u*u*(10+u*(-15+6*u)))
        return torch.where((age>=100)[:,None],self.nominal,q)

    def sample_next(self):
        age=self.age+1;q=self._position(age);prev=self._position(age-1)
        vv=(q-prev)/.02;aa=(vv-(prev-self._position((age-2).clamp_min(0)))/.02)/.02
        u=(age.to(torch.float64)/100).clamp(0,1)[:,None];delta=self.nominal-self.start
        analytic_v=delta*(30*u*u*(1-u)*(1-u))/2
        analytic_a=delta*(60*u-180*u*u+120*u*u*u)/4
        if (vv[self.warming].abs()>1.75).any() or (aa[self.warming].abs()>6).any():
            raise ValueError('Canonical recovery exceeds unchanged reference P/V/A budget')
        return {'q_ref':q,'v_ref':analytic_v,'a_ref':analytic_a,
                'discrete_v_ref':vv,'discrete_a_ref':aa,'valid':torch.ones_like(self.warming)}

    def advanced(self):
        self.age[self.warming]+=1
        return self.warming & (self.age==200)
