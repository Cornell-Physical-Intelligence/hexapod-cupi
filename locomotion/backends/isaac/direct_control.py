"""CPU tensor contract for the DirectRLEnv candidate."""
import torch
from .config import JOINT_NAMES, KD
from ...core.motor import motor_force


def named_motor_force(joint_names, q, dq, target):
    """Return Nm tensors in the actuator's joint order, with exact named gains."""
    if len(joint_names) != 18 or set(joint_names) != set(JOINT_NAMES):
        raise ValueError("Direct actuator requires the canonical 18 named joints")
    kd = q.new_tensor([KD[JOINT_NAMES.index(name)] for name in joint_names])
    return motor_force(q, dq, target, kd)


def reset_rows(indices, num_envs, device):
    """Validate selected reset rows before a state write."""
    rows = torch.arange(num_envs, device=device) if indices is None else torch.as_tensor(indices, dtype=torch.long, device=device)
    if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or bool(torch.any(rows < 0)) or bool(torch.any(rows >= num_envs)) or len(rows.unique()) != len(rows):
        raise ValueError("Invalid selected reset rows")
    return rows.to(torch.long)
