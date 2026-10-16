"""Compute RS05 motor forces under the speed-dependent torque curve."""
import math
import torch

STIFFNESS_NM_PER_RAD = 12.0
RATED_TORQUE_NM = 1.6


def motor_force(q, dq, target, kd):
    """Canonical float32 PD plus the provisional 48V speed-dependent curve."""
    requested = STIFFNESS_NM_PER_RAD * (target - q) - kd * dq
    rpm = dq.to(torch.float64).abs() * 60.0 / (2.0 * math.pi)
    xp = torch.tensor([0., 70., 275., 340., 450., 477., 480.], dtype=torch.float64, device=q.device)
    yp = torch.tensor([5.5, 5.5, 4., 3., 1.6, .5, 0.], dtype=torch.float64, device=q.device)
    index = torch.searchsorted(xp, rpm.contiguous(), right=True).clamp(1, 6)
    ceiling = yp[index - 1] + (rpm - xp[index - 1]) * (yp[index] - yp[index - 1]) / (xp[index] - xp[index - 1])
    ceiling = torch.where(rpm >= 480., 0., ceiling).clamp(min=0., max=RATED_TORQUE_NM).to(torch.float32)
    return requested, torch.maximum(torch.minimum(requested, ceiling), -ceiling), ceiling
