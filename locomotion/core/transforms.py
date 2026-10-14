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


