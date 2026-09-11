"""Hardware-realizable first-order joint-position request filter."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import math
import torch


def filter_joint_targets(requested, previous_filtered, dt, time_constant_s):
    if requested.shape != previous_filtered.shape:
        raise ValueError("Requested and filtered joint targets must have matching shapes")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(time_constant_s) or time_constant_s < 0:
        raise ValueError("Filter dt must be positive and time constant nonnegative")
    if time_constant_s == 0:
        return requested
    # Exact zero-order-hold discretization of tau*dq_target/dt=request-q_target.
    alpha = -math.expm1(-dt / time_constant_s)
    return previous_filtered + alpha * (requested - previous_filtered)
