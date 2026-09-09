"""Torch-only position-setpoint interpolation for the 18 active motors.

This changes the delivered position setpoint between policy updates. It neither
sets passive coordinates nor supplies a velocity target or feedforward torque.
The upstream action processor remains responsible for joint and slew limits.
"""
from __future__ import annotations

import torch


class MotorTargetRamp:
    """Deliver fractions 1/N through N/N of each processed target change.

    Targets are floating tensors shaped ``(environments, 18)``. ``begin`` starts
    from the last delivered target and must follow a completed interval or a
    full reset. Exactly ``substeps`` calls to ``step`` complete that interval.
    Returned tensors and all accepted inputs are isolated from internal state.

    A selected reset accepts one row per selected environment. Those rows hold
    their reset positions for the rest of the current interval; other rows keep
    their existing trajectory and phase. A full reset cancels the interval.
    Resets intentionally bypass interpolation and must coincide with the caller's
    physical episode reset. This helper does not implement a hardware transport.
    """

    def __init__(self, initial_targets, substeps=16):
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self._validate_targets(initial_targets)
        self._substeps = substeps
        self._delivered = initial_targets.detach().clone()
        self._start = self._delivered.clone()
        self._endpoint = self._delivered.clone()
        self._phase = substeps

    @staticmethod
    def _validate_targets(value):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 18
                or not value.is_floating_point()):
            raise ValueError("Motor targets must be floating tensors with shape (environments, 18)")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Motor targets must be finite")

    def _validate_compatible(self, value, rows):
        self._validate_targets(value)
        if (value.shape[0] != rows or value.dtype != self._delivered.dtype
                or value.device != self._delivered.device):
            raise ValueError("Motor targets must match the selected rows, dtype and device")

    def begin(self, processed_endpoint):
        """Accept the already limited 50 Hz endpoint, without delivering it yet."""
        if self._phase != self._substeps:
            raise RuntimeError("Previous motor-target interval has not completed")
        self._validate_compatible(processed_endpoint, self._delivered.shape[0])
        self._start = self._delivered.clone()
        self._endpoint = processed_endpoint.detach().clone()
        self._phase = 0

    def step(self):
        """Return the next position target; the last call returns the exact endpoint."""
        if self._phase >= self._substeps:
            raise RuntimeError("A new processed endpoint is required before another substep")
        self._phase += 1
        if self._phase == self._substeps:
            self._delivered = self._endpoint.clone()
        else:
            self._delivered = torch.lerp(self._start, self._endpoint, self._phase / self._substeps)
            # Preserve the convex bound even at a floating-point rounding edge.
            self._delivered.clamp_(min=torch.minimum(self._start, self._endpoint),
                                   max=torch.maximum(self._start, self._endpoint))
        return self._delivered.clone()

    def reset(self, targets, env_ids=None):
        """Reset all rows, or hold selected rows while the others finish their ramp."""
        if env_ids is None:
            self._validate_compatible(targets, self._delivered.shape[0])
            self._delivered = targets.detach().clone()
            self._start = self._delivered.clone()
            self._endpoint = self._delivered.clone()
            self._phase = self._substeps
            return
        indices = torch.as_tensor(env_ids, device=self._delivered.device)
        if indices.ndim != 1 or (indices.numel() and indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Reset environment IDs must be a one-dimensional integer sequence")
        indices = indices.to(dtype=torch.long)
        if (bool(((indices < 0) | (indices >= self._delivered.shape[0])).any())
                or indices.unique().numel() != indices.numel()):
            raise ValueError("Reset environment IDs must be unique and in range")
        self._validate_compatible(targets, indices.numel())
        values = targets.detach().clone()
        self._delivered[indices] = values
        self._start[indices] = values
        self._endpoint[indices] = values
