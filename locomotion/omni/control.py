"""Use the canonical XYZW quaternion and navigation frames."""
import torch


def rotate(quaternion, vector):
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    t = 2 * torch.cross(q[..., :3], vector, dim=-1)
    return vector + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def inverse_rotate(quaternion, vector):
    q = quaternion.clone()
    q[..., :3] *= -1
    return rotate(q, vector)


def navigation(vector):
    return torch.stack((-vector[..., 1], vector[..., 0], vector[..., 2]), dim=-1)




"""Compute canonical motor forces and held targets."""
import math
import torch


def motor_force(q, dq, target, kd):
    """Canonical float32 PD plus the provisional 48V speed-dependent curve."""
    requested = 12.0 * (target - q) - kd * dq
    rpm = dq.to(torch.float64).abs() * 60.0 / (2.0 * math.pi)
    xp = torch.tensor([0., 70., 275., 340., 450., 477., 480.], dtype=torch.float64, device=q.device)
    yp = torch.tensor([5.5, 5.5, 4., 3., 1.6, .5, 0.], dtype=torch.float64, device=q.device)
    index = torch.searchsorted(xp, rpm.contiguous(), right=True).clamp(1, 6)
    ceiling = yp[index - 1] + (rpm - xp[index - 1]) * (yp[index] - yp[index - 1]) / (xp[index] - xp[index - 1])
    ceiling = torch.where(rpm >= 480., 0., ceiling).clamp(min=0., max=1.6).to(torch.float32)
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


