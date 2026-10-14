"""Compute canonical motor forces and held targets."""
import math
import torch

from ..randomization import strength_ceiling


def motor_force(q, dq, target, kd, kp=12.0, strength=None):
    """Canonical float32 PD plus the provisional 48V speed-dependent curve.

    ``kp`` and ``strength`` carry a Step 7 draw (randomization.py); the defaults are the nominal motor.
    """
    requested = kp * (target - q) - kd * dq
    rpm = dq.to(torch.float64).abs() * 60.0 / (2.0 * math.pi)
    xp = torch.tensor([0., 70., 275., 340., 450., 477., 480.], dtype=torch.float64, device=q.device)
    yp = torch.tensor([5.5, 5.5, 4., 3., 1.6, .5, 0.], dtype=torch.float64, device=q.device)
    index = torch.searchsorted(xp, rpm.contiguous(), right=True).clamp(1, 6)
    ceiling = yp[index - 1] + (rpm - xp[index - 1]) * (yp[index] - yp[index - 1]) / (xp[index] - xp[index - 1])
    ceiling = torch.where(rpm >= 480., 0., ceiling).clamp(min=0., max=1.6).to(torch.float32)
    if strength is not None:
        ceiling = strength_ceiling(ceiling, strength)
    return requested, torch.maximum(torch.minimum(requested, ceiling), -ceiling), ceiling


def emitted_target(action, held, neutral, lower, upper, scale=.35, slew=.04):
    requested = (neutral + scale * action.clamp(-1, 1)).clamp(lower, upper)
    lo = torch.maximum(lower.to(torch.float64), held.to(torch.float64) - slew)
    hi = torch.minimum(upper.to(torch.float64), held.to(torch.float64) + slew)
    target = torch.maximum(torch.minimum(requested.to(torch.float64), hi), lo).to(torch.float32)
    target = torch.where(target.to(torch.float64) > hi, torch.nextafter(target, torch.full_like(target, -torch.inf)), target)
    target = torch.where(target.to(torch.float64) < lo, torch.nextafter(target, torch.full_like(target, torch.inf)), target)
    return target


def executed_action_feature(target, neutral, scale):
    """Expose the actual held-target state after joint clamp and slew limiting."""
    return (target - neutral) / scale


