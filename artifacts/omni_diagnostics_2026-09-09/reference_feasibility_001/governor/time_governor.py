"""A separate coherent physical-time governor, no individual-joint projection.

This is offline kinematic screening. Sampled bounds are not physical admission
or a continuous-workspace certificate. Existing prototype files stay frozen.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import *

REFERENCE_RATE_BUDGET=1.25  # rad/s, leaves .25 of the1.5rad/s budget for feedback
SAMPLING_MARGIN=.90

def time_scale_from_rate_bound(bound_rad_s):
    if not math.isfinite(bound_rad_s) or bound_rad_s<=0:
        raise ValueError('Positive finite virtual joint-rate bound required')
    return min(1.,SAMPLING_MARGIN*REFERENCE_RATE_BUDGET/bound_rad_s)

def advance_governed(state, requested_virtual_twist, physical_dt, scale):
    """Same virtual curve, slower clock; request/actual admitted twist are explicit.

    requested_virtual_twist is a shape/time request from the original family.
    Admitted command target is scale * requested_virtual_twist. Scale is frozen
    for a declared sequence, never jumped halfway through a step.
    """
    if not 0<scale<=1 or not 0<physical_dt<=.1:
        raise ValueError('A fixed 0<scale<=1 and bounded physical dt are required')
    out=state.step(requested_virtual_twist,scale*physical_dt)
    if not out['valid'].all():
        raise ValueError('Reference leaves admitted point workspace; no target emitted')
    out['admitted_command']=scale*state.command
    out['admitted_command_target']=scale*tensor(requested_virtual_twist)
    out['admitted_acceleration']=scale**2*state.rate
    out['physical_q_velocity']=scale*out['q_velocity']
    out['physical_foot_velocity']=scale*out['foot_velocity']
    out['physical_frequency']=scale*out['frequency']
    return out
