"""Diagnostic facts, never a native admission or a scheduled-flight substitute."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProposedMovingRule:
    # No default. A new contract review must select/bind the count before native integration.
    name:str
    minimum_support_count:int
    def __post_init__(self):
        if not self.name or type(self.minimum_support_count)is not int or not 1<=self.minimum_support_count<=6:
            raise ValueError('Explicit named nonempty moving-support proposal required')


def contact_facts(before, current, valid, *, phase, moving_rule=None):
    a=np.asarray(before);b=np.asarray(current);v=np.asarray(valid)
    if b.ndim!=3 or b.shape[1:]!=(8,6) or a.shape!=(len(b),6) or v.shape!=(len(b),8):
        raise ValueError('Exact before + eight contact masks required')
    if any(x.dtype!=bool for x in [a,b,v]) or not v.all():
        raise ValueError('Missing/invalid contact evidence cannot be zero filled')
    if phase=='neutral_standing': minimum=6
    elif phase=='moving_or_stopping':
        if not isinstance(moving_rule,ProposedMovingRule): raise ValueError('Moving rule unresolved')
        minimum=moving_rule.minimum_support_count
    else: raise ValueError('Formal quiet scoring remains external and unchanged')
    prior=np.concatenate([a[:,None],b[:,:-1]],axis=1)
    counts=b.sum(axis=2)
    return {'support_counts':counts,'proposed_count_check':np.all(counts>=minimum,axis=1),
            'force_contact_departures':prior & ~b,'force_contact_returns':~prior & b,
            'qualified_flight':None,'qualified_landing':None,'physical_admission':False}
