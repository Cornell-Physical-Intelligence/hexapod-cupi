"""Turn policy actions into slew-limited held targets and the executed-action feature."""

import torch


def emitted_target(action, held, neutral, lower, upper, scale=0.35, slew=0.04):
    requested = (neutral + scale * action.clamp(-1, 1)).clamp(lower, upper)
    lo = torch.maximum(lower.to(torch.float64), held.to(torch.float64) - slew)
    hi = torch.minimum(upper.to(torch.float64), held.to(torch.float64) + slew)
    target = torch.maximum(torch.minimum(requested.to(torch.float64), hi), lo).to(torch.float32)
    target = torch.where(
        target.to(torch.float64) > hi,
        torch.nextafter(target, torch.full_like(target, -torch.inf)),
        target,
    )
    target = torch.where(
        target.to(torch.float64) < lo,
        torch.nextafter(target, torch.full_like(target, torch.inf)),
        target,
    )
    return target


def executed_action_feature(target, neutral, scale):
    """Expose the actual held-target state after joint clamp and slew limiting."""
    return (target - neutral) / scale
