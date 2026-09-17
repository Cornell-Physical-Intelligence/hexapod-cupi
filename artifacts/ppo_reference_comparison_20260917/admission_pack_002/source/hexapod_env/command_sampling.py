"""Pure-Torch utilities for opt-in velocity-command curricula."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


STANDING = 0
LONGITUDINAL_ONLY = 1
LATERAL_ONLY = 2
YAW_ONLY = 3
COMBINED = 4
NUM_COMMAND_CATEGORIES = 5

RECOVERY_FORWARD = 0
RECOVERY_FORWARD_YAW = 1
RECOVERY_FORWARD_LATERAL = 2
RECOVERY_COMBINED = 3
NUM_RECOVERY_CATEGORIES = 4

# Twenty fixed environment buckets give the exact 55/20/20/5 Stage 1 mix.
# Keeping a category tied to the environment index makes the allocation stable
# across asynchronous episode resets while the command magnitudes are sampled
# afresh for each episode.
RECOVERY_BUCKET_PERIOD = 20
RECOVERY_FORWARD_BUCKETS = 11
RECOVERY_FORWARD_YAW_BUCKETS = 4
RECOVERY_FORWARD_LATERAL_BUCKETS = 4
RECOVERY_COMBINED_BUCKETS = 1

# Stage 2 deliberately removes combined commands until each missing axis has
# acquired a signed response.  Twenty fixed buckets retain a large forward
# anchor while giving lateral and yaw six balanced-sign buckets apiece.
STAGE2_FORWARD = 0
STAGE2_FORWARD_YAW = 1
STAGE2_FORWARD_LATERAL = 2
NUM_STAGE2_CATEGORIES = 3
STAGE2_BUCKET_PERIOD = 20
STAGE2_FORWARD_BUCKETS = 8
STAGE2_FORWARD_YAW_BUCKETS = 6
STAGE2_FORWARD_LATERAL_BUCKETS = 6

# Stage 2B fixes the acquisition failure observed in the first Stage 2 screen:
# every old "lateral" command also requested 0.20--0.30 m/s forward motion, so
# the warm-start policy could earn most of its return by ignoring y.  The new
# fixed buckets devote half the fleet to lateral motion, including a 40% pure
# lateral subset with vx=0, while preserving 40% forward anchors and 10% of the
# already-emerging forward+yaw behavior.
STAGE2B_FORWARD = 0
STAGE2B_LATERAL_ONLY = 1
STAGE2B_FORWARD_LATERAL = 2
STAGE2B_FORWARD_YAW = 3
NUM_STAGE2B_CATEGORIES = 4
STAGE2B_BUCKET_PERIOD = 20
STAGE2B_FORWARD_BUCKETS = 8
STAGE2B_LATERAL_ONLY_BUCKETS = 8
STAGE2B_FORWARD_LATERAL_BUCKETS = 2
STAGE2B_FORWARD_YAW_BUCKETS = 2

# Stage 2D acquires lateral authority through a forward-moving oblique
# homotopy rather than asking the warm-start gait for pure side translation.
# Twenty fixed buckets keep the exact 70/20/10 oblique/forward/forward+yaw
# allocation while the even oblique and yaw bucket counts provide paired
# signs inside every period.
STAGE2D_FORWARD = 0
STAGE2D_OBLIQUE = 1
STAGE2D_FORWARD_YAW = 2
NUM_STAGE2D_CATEGORIES = 3
STAGE2D_BUCKET_PERIOD = 20
STAGE2D_OBLIQUE_BUCKETS = 14
STAGE2D_FORWARD_BUCKETS = 4
STAGE2D_FORWARD_YAW_BUCKETS = 2

# Stage 2E turns the acquired x/y/yaw primitives into a real joystick policy.
# Forty slots make every signed category exactly balanced while retaining
# explicit stand, pure-y, forward, and forward+yaw anchors.  A caller-supplied
# reset-safe epoch rotates the slot assignment by a coprime stride after each
# resample: the fleet mixture stays deterministic and exact at every common
# epoch, but no environment remains tied to one command category forever.
STAGE2E_STAND = 0
STAGE2E_LATERAL = 1
STAGE2E_FORWARD = 2
STAGE2E_REVERSE = 3
STAGE2E_YAW = 4
STAGE2E_FORWARD_JOYSTICK = 5
STAGE2E_REVERSE_JOYSTICK = 6
NUM_STAGE2E_CATEGORIES = 7
STAGE2E_BUCKET_PERIOD = 40


def body_to_navigation_frame(vectors: torch.Tensor) -> torch.Tensor:
    """Rotate body vectors into ``[forward, lateral, up]`` navigation axes.

    The robot's anatomical forward direction is body ``-Y`` and anatomical
    lateral direction is body ``+X``.  Therefore the navigation-frame mapping
    is ``[-body_y, body_x, body_z]``.  The operation applies to both ordinary
    vectors and axial vectors (for example angular velocity).
    """

    if vectors.shape[-1] != 3:
        raise ValueError(f"Expected vectors with final dimension 3, got {vectors.shape!r}")
    return torch.stack((-vectors[..., 1], vectors[..., 0], vectors[..., 2]), dim=-1)


def _validate_range(name: str, value_range: Sequence[float]) -> tuple[float, float]:
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain two values, got {value_range!r}")
    low, high = (float(value_range[0]), float(value_range[1]))
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{name} must be finite and ordered low-to-high, got {value_range!r}")
    return low, high


def sample_velocity_command_mixture(
    num_samples: int,
    *,
    lin_vel_x_range: Sequence[float],
    lin_vel_y_range: Sequence[float],
    ang_vel_z_range: Sequence[float],
    category_probabilities: Sequence[float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample mutually exclusive stand/x-only/y-only/yaw-only/combined commands."""

    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    x_range = _validate_range("lin_vel_x_range", lin_vel_x_range)
    y_range = _validate_range("lin_vel_y_range", lin_vel_y_range)
    yaw_range = _validate_range("ang_vel_z_range", ang_vel_z_range)
    probabilities = tuple(float(value) for value in category_probabilities)
    if len(probabilities) != NUM_COMMAND_CATEGORIES:
        raise ValueError(
            f"Expected {NUM_COMMAND_CATEGORIES} category probabilities, got {probabilities!r}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError(f"Category probabilities must be finite and non-negative: {probabilities!r}")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Category probabilities must sum to 1.0, got {sum(probabilities)}")

    commands = torch.zeros((num_samples, 3), device=device, dtype=dtype)
    if num_samples == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    # One draw selects exactly one category.  Comparing against the first four
    # cumulative thresholds assigns the residual interval to COMBINED.
    cumulative = torch.tensor(probabilities, device=device, dtype=dtype).cumsum(dim=0)
    category_draw = torch.rand(num_samples, device=device, dtype=dtype, generator=generator)
    categories = torch.sum(category_draw[:, None] >= cumulative[None, :-1], dim=1).long()

    raw = torch.empty((num_samples, 3), device=device, dtype=dtype)
    raw[:, 0].uniform_(*x_range, generator=generator)
    raw[:, 1].uniform_(*y_range, generator=generator)
    raw[:, 2].uniform_(*yaw_range, generator=generator)

    longitudinal = categories == LONGITUDINAL_ONLY
    lateral = categories == LATERAL_ONLY
    yaw = categories == YAW_ONLY
    combined = categories == COMBINED
    commands[longitudinal, 0] = raw[longitudinal, 0]
    commands[lateral, 1] = raw[lateral, 1]
    commands[yaw, 2] = raw[yaw, 2]
    commands[combined] = raw[combined]
    return commands, categories


def _positive_range(name: str, value_range: Sequence[float]) -> tuple[float, float]:
    low, high = _validate_range(name, value_range)
    if low <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value_range!r}")
    return low, high


def _nonnegative_range(name: str, value_range: Sequence[float]) -> tuple[float, float]:
    low, high = _validate_range(name, value_range)
    if low < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value_range!r}")
    return low, high


def sample_stage1_recovery_commands(
    bucket_indices: torch.Tensor,
    *,
    forward_range: Sequence[float] = (0.20, 0.35),
    moving_forward_range: Sequence[float] = (0.20, 0.32),
    lateral_abs_range: Sequence[float] = (0.03, 0.07),
    yaw_abs_range: Sequence[float] = (0.06, 0.15),
    combined_forward_range: Sequence[float] = (0.22, 0.30),
    combined_lateral_abs_range: Sequence[float] = (0.03, 0.06),
    combined_yaw_abs_range: Sequence[float] = (0.06, 0.12),
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the fixed-bucket Phase 2 recovery curriculum.

    The category of an environment is a pure function of its global index:
    eleven of every twenty environments retain the accepted forward gait,
    four add yaw, four add lateral velocity, and one adds both.  Signs are
    balanced deterministically within each category, including all four sign
    combinations for the combined bucket.  Only magnitudes change on reset.
    """

    if bucket_indices.ndim != 1:
        raise ValueError(
            f"bucket_indices must be one-dimensional, got {bucket_indices.shape!r}"
        )
    if bucket_indices.dtype == torch.bool or bucket_indices.dtype.is_floating_point:
        raise ValueError(f"bucket_indices must use an integer dtype, got {bucket_indices.dtype}")
    if torch.any(bucket_indices < 0):
        raise ValueError("bucket_indices must be non-negative")

    forward = _positive_range("forward_range", forward_range)
    moving_forward = _positive_range("moving_forward_range", moving_forward_range)
    lateral_abs = _positive_range("lateral_abs_range", lateral_abs_range)
    yaw_abs = _positive_range("yaw_abs_range", yaw_abs_range)
    combined_forward = _positive_range("combined_forward_range", combined_forward_range)
    combined_lateral_abs = _positive_range(
        "combined_lateral_abs_range", combined_lateral_abs_range
    )
    combined_yaw_abs = _positive_range(
        "combined_yaw_abs_range", combined_yaw_abs_range
    )

    device = bucket_indices.device
    count = bucket_indices.numel()
    commands = torch.zeros((count, 3), device=device, dtype=dtype)
    if count == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    cycle = torch.div(bucket_indices, RECOVERY_BUCKET_PERIOD, rounding_mode="floor")
    slot = torch.remainder(bucket_indices, RECOVERY_BUCKET_PERIOD)
    categories = torch.full(
        (count,), RECOVERY_COMBINED, device=device, dtype=torch.long
    )
    categories[slot < RECOVERY_FORWARD_BUCKETS] = RECOVERY_FORWARD
    categories[
        (slot >= RECOVERY_FORWARD_BUCKETS)
        & (slot < RECOVERY_FORWARD_BUCKETS + RECOVERY_FORWARD_YAW_BUCKETS)
    ] = RECOVERY_FORWARD_YAW
    categories[
        (slot >= RECOVERY_FORWARD_BUCKETS + RECOVERY_FORWARD_YAW_BUCKETS)
        & (
            slot
            < RECOVERY_FORWARD_BUCKETS
            + RECOVERY_FORWARD_YAW_BUCKETS
            + RECOVERY_FORWARD_LATERAL_BUCKETS
        )
    ] = RECOVERY_FORWARD_LATERAL

    raw = torch.empty((count, 3), device=device, dtype=dtype)
    raw[:, 0].uniform_(*moving_forward, generator=generator)
    raw[:, 1].uniform_(*lateral_abs, generator=generator)
    raw[:, 2].uniform_(*yaw_abs, generator=generator)

    forward_mask = categories == RECOVERY_FORWARD
    yaw_mask = categories == RECOVERY_FORWARD_YAW
    lateral_mask = categories == RECOVERY_FORWARD_LATERAL
    combined_mask = categories == RECOVERY_COMBINED

    forward_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *forward, generator=generator
    )
    combined_forward_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*combined_forward, generator=generator)
    commands[:, 0] = raw[:, 0]
    commands[forward_mask, 0] = forward_magnitude[forward_mask]
    commands[combined_mask, 0] = combined_forward_magnitude[combined_mask]

    yaw_rank = cycle * RECOVERY_FORWARD_YAW_BUCKETS + (
        slot - RECOVERY_FORWARD_BUCKETS
    )
    yaw_sign = torch.where(
        torch.remainder(yaw_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[yaw_mask, 2] = raw[yaw_mask, 2] * yaw_sign[yaw_mask]

    lateral_rank = cycle * RECOVERY_FORWARD_LATERAL_BUCKETS + (
        slot - RECOVERY_FORWARD_BUCKETS - RECOVERY_FORWARD_YAW_BUCKETS
    )
    lateral_sign = torch.where(
        torch.remainder(lateral_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[lateral_mask, 1] = raw[lateral_mask, 1] * lateral_sign[lateral_mask]

    combined_rank = cycle
    combined_lateral_sign = torch.where(
        torch.remainder(combined_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    combined_yaw_sign = torch.where(
        torch.remainder(torch.div(combined_rank, 2, rounding_mode="floor"), 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    combined_lateral_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *combined_lateral_abs, generator=generator
    )
    combined_yaw_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *combined_yaw_abs, generator=generator
    )
    commands[combined_mask, 1] = (
        combined_lateral_magnitude[combined_mask]
        * combined_lateral_sign[combined_mask]
    )
    commands[combined_mask, 2] = (
        combined_yaw_magnitude[combined_mask] * combined_yaw_sign[combined_mask]
    )
    return commands, categories


def sample_stage2_recovery_commands(
    bucket_indices: torch.Tensor,
    *,
    forward_range: Sequence[float] = (0.22, 0.34),
    moving_forward_range: Sequence[float] = (0.20, 0.30),
    lateral_abs_range: Sequence[float] = (0.08, 0.14),
    yaw_abs_range: Sequence[float] = (0.18, 0.30),
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the sign-acquisition Stage 2 recovery curriculum.

    Eight of every twenty environments preserve the forward-only anchor, six
    add lateral velocity, and six add yaw.  Lateral and yaw are isolated from
    one another in this stage so PPO gets an unambiguous signed learning
    signal before combined joystick commands are reintroduced.  Categories
    and signs are stable functions of the global environment index; command
    magnitudes are sampled afresh on reset.
    """

    if bucket_indices.ndim != 1:
        raise ValueError(
            f"bucket_indices must be one-dimensional, got {bucket_indices.shape!r}"
        )
    if bucket_indices.dtype == torch.bool or bucket_indices.dtype.is_floating_point:
        raise ValueError(f"bucket_indices must use an integer dtype, got {bucket_indices.dtype}")
    if torch.any(bucket_indices < 0):
        raise ValueError("bucket_indices must be non-negative")

    forward = _positive_range("forward_range", forward_range)
    moving_forward = _positive_range("moving_forward_range", moving_forward_range)
    lateral_abs = _positive_range("lateral_abs_range", lateral_abs_range)
    yaw_abs = _positive_range("yaw_abs_range", yaw_abs_range)

    device = bucket_indices.device
    count = bucket_indices.numel()
    commands = torch.zeros((count, 3), device=device, dtype=dtype)
    if count == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    cycle = torch.div(bucket_indices, STAGE2_BUCKET_PERIOD, rounding_mode="floor")
    slot = torch.remainder(bucket_indices, STAGE2_BUCKET_PERIOD)
    categories = torch.full(
        (count,), STAGE2_FORWARD_LATERAL, device=device, dtype=torch.long
    )
    categories[slot < STAGE2_FORWARD_BUCKETS] = STAGE2_FORWARD
    yaw_mask = (
        (slot >= STAGE2_FORWARD_BUCKETS)
        & (slot < STAGE2_FORWARD_BUCKETS + STAGE2_FORWARD_YAW_BUCKETS)
    )
    categories[yaw_mask] = STAGE2_FORWARD_YAW

    raw_forward = torch.empty(count, device=device, dtype=dtype).uniform_(
        *moving_forward, generator=generator
    )
    anchor_forward = torch.empty(count, device=device, dtype=dtype).uniform_(
        *forward, generator=generator
    )
    lateral_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *lateral_abs, generator=generator
    )
    yaw_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *yaw_abs, generator=generator
    )
    commands[:, 0] = raw_forward
    forward_mask = categories == STAGE2_FORWARD
    lateral_mask = categories == STAGE2_FORWARD_LATERAL
    commands[forward_mask, 0] = anchor_forward[forward_mask]

    yaw_rank = cycle * STAGE2_FORWARD_YAW_BUCKETS + (
        slot - STAGE2_FORWARD_BUCKETS
    )
    yaw_sign = torch.where(
        torch.remainder(yaw_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[yaw_mask, 2] = yaw_magnitude[yaw_mask] * yaw_sign[yaw_mask]

    lateral_rank = cycle * STAGE2_FORWARD_LATERAL_BUCKETS + (
        slot - STAGE2_FORWARD_BUCKETS - STAGE2_FORWARD_YAW_BUCKETS
    )
    lateral_sign = torch.where(
        torch.remainder(lateral_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[lateral_mask, 1] = (
        lateral_magnitude[lateral_mask] * lateral_sign[lateral_mask]
    )
    return commands, categories


def sample_stage2b_lateral_acquisition_commands(
    bucket_indices: torch.Tensor,
    *,
    forward_range: Sequence[float] = (0.22, 0.32),
    lateral_only_abs_range: Sequence[float] = (0.08, 0.12),
    combined_forward_range: Sequence[float] = (0.18, 0.24),
    combined_lateral_abs_range: Sequence[float] = (0.06, 0.10),
    yaw_forward_range: Sequence[float] = (0.20, 0.26),
    yaw_abs_range: Sequence[float] = (0.20, 0.28),
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed 40/40/10/10 forward/pure-y/forward+y/forward+yaw mix.

    Pure lateral buckets remove the forward-gait shortcut that prevented y
    acquisition in Stage 2.  Forward+lateral buckets provide a bridge to the
    eventual joystick task, and a small balanced forward+yaw allocation
    consolidates the signed yaw response already present in the safe seed.
    Categories and signs remain deterministic functions of global environment
    index; only command magnitudes change when an episode resets.
    """

    if bucket_indices.ndim != 1:
        raise ValueError(
            f"bucket_indices must be one-dimensional, got {bucket_indices.shape!r}"
        )
    if bucket_indices.dtype == torch.bool or bucket_indices.dtype.is_floating_point:
        raise ValueError(f"bucket_indices must use an integer dtype, got {bucket_indices.dtype}")
    if torch.any(bucket_indices < 0):
        raise ValueError("bucket_indices must be non-negative")

    forward = _positive_range("forward_range", forward_range)
    lateral_only_abs = _positive_range(
        "lateral_only_abs_range", lateral_only_abs_range
    )
    combined_forward = _positive_range(
        "combined_forward_range", combined_forward_range
    )
    combined_lateral_abs = _positive_range(
        "combined_lateral_abs_range", combined_lateral_abs_range
    )
    yaw_forward = _positive_range("yaw_forward_range", yaw_forward_range)
    yaw_abs = _positive_range("yaw_abs_range", yaw_abs_range)

    device = bucket_indices.device
    count = bucket_indices.numel()
    commands = torch.zeros((count, 3), device=device, dtype=dtype)
    if count == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    cycle = torch.div(bucket_indices, STAGE2B_BUCKET_PERIOD, rounding_mode="floor")
    slot = torch.remainder(bucket_indices, STAGE2B_BUCKET_PERIOD)
    categories = torch.full(
        (count,), STAGE2B_FORWARD_YAW, device=device, dtype=torch.long
    )
    categories[slot < STAGE2B_FORWARD_BUCKETS] = STAGE2B_FORWARD
    lateral_only_mask = (
        (slot >= STAGE2B_FORWARD_BUCKETS)
        & (
            slot
            < STAGE2B_FORWARD_BUCKETS + STAGE2B_LATERAL_ONLY_BUCKETS
        )
    )
    categories[lateral_only_mask] = STAGE2B_LATERAL_ONLY
    forward_lateral_mask = (
        (slot >= STAGE2B_FORWARD_BUCKETS + STAGE2B_LATERAL_ONLY_BUCKETS)
        & (
            slot
            < STAGE2B_FORWARD_BUCKETS
            + STAGE2B_LATERAL_ONLY_BUCKETS
            + STAGE2B_FORWARD_LATERAL_BUCKETS
        )
    )
    categories[forward_lateral_mask] = STAGE2B_FORWARD_LATERAL

    forward_mask = categories == STAGE2B_FORWARD
    yaw_mask = categories == STAGE2B_FORWARD_YAW
    commands[forward_mask, 0] = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*forward, generator=generator)[forward_mask]
    commands[forward_lateral_mask, 0] = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*combined_forward, generator=generator)[forward_lateral_mask]
    commands[yaw_mask, 0] = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*yaw_forward, generator=generator)[yaw_mask]

    lateral_only_rank = cycle * STAGE2B_LATERAL_ONLY_BUCKETS + (
        slot - STAGE2B_FORWARD_BUCKETS
    )
    lateral_only_sign = torch.where(
        torch.remainder(lateral_only_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    lateral_only_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*lateral_only_abs, generator=generator)
    commands[lateral_only_mask, 1] = (
        lateral_only_magnitude[lateral_only_mask]
        * lateral_only_sign[lateral_only_mask]
    )

    forward_lateral_rank = cycle * STAGE2B_FORWARD_LATERAL_BUCKETS + (
        slot - STAGE2B_FORWARD_BUCKETS - STAGE2B_LATERAL_ONLY_BUCKETS
    )
    forward_lateral_sign = torch.where(
        torch.remainder(forward_lateral_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    forward_lateral_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*combined_lateral_abs, generator=generator)
    commands[forward_lateral_mask, 1] = (
        forward_lateral_magnitude[forward_lateral_mask]
        * forward_lateral_sign[forward_lateral_mask]
    )

    yaw_rank = cycle * STAGE2B_FORWARD_YAW_BUCKETS + (
        slot
        - STAGE2B_FORWARD_BUCKETS
        - STAGE2B_LATERAL_ONLY_BUCKETS
        - STAGE2B_FORWARD_LATERAL_BUCKETS
    )
    yaw_sign = torch.where(
        torch.remainder(yaw_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    yaw_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *yaw_abs, generator=generator
    )
    commands[yaw_mask, 2] = yaw_magnitude[yaw_mask] * yaw_sign[yaw_mask]
    return commands, categories


def sample_stage2d_oblique_homotopy_commands(
    bucket_indices: torch.Tensor,
    *,
    oblique_forward_range: Sequence[float],
    oblique_lateral_abs_range: Sequence[float],
    forward_anchor_range: Sequence[float],
    yaw_anchor_forward_range: Sequence[float],
    yaw_anchor_abs_range: Sequence[float],
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed 70/20/10 oblique/forward/forward+yaw homotopy.

    The category and non-zero lateral/yaw sign are pure functions of each
    global environment index.  Consequently asynchronous resets may resample
    magnitudes without changing an environment's assigned heading, and a
    caller can retain one command for a full episode simply by choosing a hold
    interval at least as long as that episode.  Every twenty-index period has
    seven positive and seven negative oblique commands plus one yaw command of
    each sign.  All physical magnitudes are supplied explicitly by the caller;
    oblique forward speed may reach exactly zero at the pure-y endpoint while
    lateral, forward-anchor, and yaw magnitudes remain strictly positive.
    """

    if bucket_indices.ndim != 1:
        raise ValueError(
            f"bucket_indices must be one-dimensional, got {bucket_indices.shape!r}"
        )
    if bucket_indices.dtype == torch.bool or bucket_indices.dtype.is_floating_point:
        raise ValueError(f"bucket_indices must use an integer dtype, got {bucket_indices.dtype}")
    if torch.any(bucket_indices < 0):
        raise ValueError("bucket_indices must be non-negative")

    oblique_forward = _nonnegative_range(
        "oblique_forward_range", oblique_forward_range
    )
    oblique_lateral_abs = _positive_range(
        "oblique_lateral_abs_range", oblique_lateral_abs_range
    )
    forward_anchor = _positive_range(
        "forward_anchor_range", forward_anchor_range
    )
    yaw_anchor_forward = _positive_range(
        "yaw_anchor_forward_range", yaw_anchor_forward_range
    )
    yaw_anchor_abs = _positive_range(
        "yaw_anchor_abs_range", yaw_anchor_abs_range
    )

    device = bucket_indices.device
    count = bucket_indices.numel()
    commands = torch.zeros((count, 3), device=device, dtype=dtype)
    if count == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    cycle = torch.div(bucket_indices, STAGE2D_BUCKET_PERIOD, rounding_mode="floor")
    slot = torch.remainder(bucket_indices, STAGE2D_BUCKET_PERIOD)
    oblique_mask = slot < STAGE2D_OBLIQUE_BUCKETS
    forward_mask = (
        (slot >= STAGE2D_OBLIQUE_BUCKETS)
        & (
            slot
            < STAGE2D_OBLIQUE_BUCKETS + STAGE2D_FORWARD_BUCKETS
        )
    )
    yaw_mask = ~(oblique_mask | forward_mask)

    categories = torch.full(
        (count,), STAGE2D_FORWARD_YAW, device=device, dtype=torch.long
    )
    categories[oblique_mask] = STAGE2D_OBLIQUE
    categories[forward_mask] = STAGE2D_FORWARD

    oblique_forward_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*oblique_forward, generator=generator)
    oblique_lateral_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*oblique_lateral_abs, generator=generator)
    forward_anchor_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*forward_anchor, generator=generator)
    yaw_anchor_forward_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*yaw_anchor_forward, generator=generator)
    yaw_anchor_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*yaw_anchor_abs, generator=generator)

    commands[oblique_mask, 0] = oblique_forward_magnitude[oblique_mask]
    commands[forward_mask, 0] = forward_anchor_magnitude[forward_mask]
    commands[yaw_mask, 0] = yaw_anchor_forward_magnitude[yaw_mask]

    # Each period begins its sign pairing anew.  Because both signed category
    # sizes are even, every complete period is balanced exactly rather than
    # only in expectation over a large fleet.
    oblique_rank = cycle * STAGE2D_OBLIQUE_BUCKETS + slot
    oblique_sign = torch.where(
        torch.remainder(oblique_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[oblique_mask, 1] = (
        oblique_lateral_magnitude[oblique_mask] * oblique_sign[oblique_mask]
    )

    yaw_rank = cycle * STAGE2D_FORWARD_YAW_BUCKETS + (
        slot - STAGE2D_OBLIQUE_BUCKETS - STAGE2D_FORWARD_BUCKETS
    )
    yaw_sign = torch.where(
        torch.remainder(yaw_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[yaw_mask, 2] = yaw_anchor_magnitude[yaw_mask] * yaw_sign[yaw_mask]
    return commands, categories


def sample_stage2e_joystick_transition_commands(
    bucket_indices: torch.Tensor,
    *,
    resample_epochs: torch.Tensor,
    bucket_counts: Sequence[int],
    bucket_stride: int,
    lateral_anchor_abs_range: Sequence[float],
    forward_anchor_range: Sequence[float],
    yaw_anchor_forward_range: Sequence[float],
    yaw_anchor_abs_range: Sequence[float],
    joystick_forward_range: Sequence[float],
    joystick_reverse_abs_range: Sequence[float],
    joystick_lateral_abs_range: Sequence[float],
    joystick_yaw_abs_range: Sequence[float],
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rotating deterministic Stage2E joystick buckets.

    ``bucket_counts`` is ordered as stand, pure lateral, forward-only,
    reverse-only, yaw, forward joystick, and reverse joystick.  Lateral/yaw
    bucket counts must be even and both joystick counts must be divisible by
    four, guaranteeing exact sign or sign-quadrant balance for every complete
    40-environment period.  The yaw category's forward bridge may taper to
    exactly zero so the terminal stage trains genuine in-place pivots.
    ``bucket_stride`` must be coprime with 40, so repeated resampling visits
    every slot instead of locking an environment into one category.  Physical
    magnitudes remain stochastic within caller-provided ranges, while category,
    direction signs, and transitions are deterministic.
    """

    if bucket_indices.ndim != 1:
        raise ValueError(
            f"bucket_indices must be one-dimensional, got {bucket_indices.shape!r}"
        )
    if bucket_indices.dtype == torch.bool or bucket_indices.dtype.is_floating_point:
        raise ValueError(
            f"bucket_indices must use an integer dtype, got {bucket_indices.dtype}"
        )
    if torch.any(bucket_indices < 0):
        raise ValueError("bucket_indices must be non-negative")
    if resample_epochs.shape != bucket_indices.shape:
        raise ValueError(
            "resample_epochs must match the one-dimensional bucket_indices shape"
        )
    if resample_epochs.dtype == torch.bool or resample_epochs.dtype.is_floating_point:
        raise ValueError(
            f"resample_epochs must use an integer dtype, got {resample_epochs.dtype}"
        )
    if torch.any(resample_epochs < 0):
        raise ValueError("resample_epochs must be non-negative")

    raw_counts = tuple(bucket_counts)
    if len(raw_counts) != NUM_STAGE2E_CATEGORIES or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_counts
    ):
        raise ValueError(
            f"bucket_counts must contain {NUM_STAGE2E_CATEGORIES} integers, "
            f"got {bucket_counts!r}"
        )
    counts = tuple(int(value) for value in raw_counts)
    if any(value <= 0 for value in counts) or sum(counts) != STAGE2E_BUCKET_PERIOD:
        raise ValueError(
            f"bucket_counts must be positive and sum to {STAGE2E_BUCKET_PERIOD}, "
            f"got {counts!r}"
        )
    if counts[STAGE2E_LATERAL] % 2 or counts[STAGE2E_YAW] % 2:
        raise ValueError("lateral and yaw bucket counts must be even")
    if (
        counts[STAGE2E_FORWARD_JOYSTICK] % 4
        or counts[STAGE2E_REVERSE_JOYSTICK] % 4
    ):
        raise ValueError("forward/reverse joystick bucket counts must be divisible by four")
    if (
        isinstance(bucket_stride, bool)
        or not isinstance(bucket_stride, int)
        or bucket_stride <= 0
        or bucket_stride >= STAGE2E_BUCKET_PERIOD
        or math.gcd(bucket_stride, STAGE2E_BUCKET_PERIOD) != 1
    ):
        raise ValueError(
            f"bucket_stride must be an integer coprime with {STAGE2E_BUCKET_PERIOD}, "
            f"got {bucket_stride!r}"
        )

    lateral_anchor_abs = _positive_range(
        "lateral_anchor_abs_range", lateral_anchor_abs_range
    )
    forward_anchor = _positive_range("forward_anchor_range", forward_anchor_range)
    yaw_anchor_forward = _nonnegative_range(
        "yaw_anchor_forward_range", yaw_anchor_forward_range
    )
    yaw_anchor_abs = _positive_range("yaw_anchor_abs_range", yaw_anchor_abs_range)
    joystick_forward = _positive_range(
        "joystick_forward_range", joystick_forward_range
    )
    joystick_reverse_abs = _positive_range(
        "joystick_reverse_abs_range", joystick_reverse_abs_range
    )
    joystick_lateral_abs = _positive_range(
        "joystick_lateral_abs_range", joystick_lateral_abs_range
    )
    joystick_yaw_abs = _positive_range(
        "joystick_yaw_abs_range", joystick_yaw_abs_range
    )

    device = bucket_indices.device
    count = bucket_indices.numel()
    commands = torch.zeros((count, 3), device=device, dtype=dtype)
    if count == 0:
        return commands, torch.empty((0,), device=device, dtype=torch.long)

    rotated_indices = bucket_indices + resample_epochs * bucket_stride
    cycle = torch.div(
        rotated_indices, STAGE2E_BUCKET_PERIOD, rounding_mode="floor"
    )
    slot = torch.remainder(rotated_indices, STAGE2E_BUCKET_PERIOD)
    boundaries = []
    total = 0
    for bucket_count in counts:
        total += bucket_count
        boundaries.append(total)
    categories = torch.full(
        (count,), STAGE2E_REVERSE_JOYSTICK, device=device, dtype=torch.long
    )
    lower = 0
    masks: list[torch.Tensor] = []
    for category, upper in enumerate(boundaries):
        mask = (slot >= lower) & (slot < upper)
        masks.append(mask)
        categories[mask] = category
        lower = upper

    lateral_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *lateral_anchor_abs, generator=generator
    )
    forward_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *forward_anchor, generator=generator
    )
    yaw_forward_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*yaw_anchor_forward, generator=generator)
    yaw_magnitude = torch.empty(count, device=device, dtype=dtype).uniform_(
        *yaw_anchor_abs, generator=generator
    )
    joystick_forward_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*joystick_forward, generator=generator)
    joystick_reverse_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*joystick_reverse_abs, generator=generator)
    joystick_lateral_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*joystick_lateral_abs, generator=generator)
    joystick_yaw_magnitude = torch.empty(
        count, device=device, dtype=dtype
    ).uniform_(*joystick_yaw_abs, generator=generator)

    (
        stand_mask,
        lateral_mask,
        forward_mask,
        reverse_anchor_mask,
        yaw_mask,
        joystick_forward_mask,
        reverse_joystick_mask,
    ) = masks
    commands[forward_mask, 0] = forward_magnitude[forward_mask]
    commands[reverse_anchor_mask, 0] = -joystick_reverse_magnitude[
        reverse_anchor_mask
    ]
    commands[yaw_mask, 0] = yaw_forward_magnitude[yaw_mask]
    commands[joystick_forward_mask, 0] = (
        joystick_forward_magnitude[joystick_forward_mask]
    )
    commands[reverse_joystick_mask, 0] = -joystick_reverse_magnitude[
        reverse_joystick_mask
    ]

    category_starts = (0, *boundaries[:-1])

    lateral_rank = cycle * counts[STAGE2E_LATERAL] + (
        slot - category_starts[STAGE2E_LATERAL]
    )
    lateral_sign = torch.where(
        torch.remainder(lateral_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[lateral_mask, 1] = lateral_magnitude[lateral_mask] * lateral_sign[lateral_mask]

    yaw_rank = cycle * counts[STAGE2E_YAW] + (
        slot - category_starts[STAGE2E_YAW]
    )
    yaw_sign = torch.where(
        torch.remainder(yaw_rank, 2) == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )
    commands[yaw_mask, 2] = yaw_magnitude[yaw_mask] * yaw_sign[yaw_mask]

    for category, mask in (
        (STAGE2E_FORWARD_JOYSTICK, joystick_forward_mask),
        (STAGE2E_REVERSE_JOYSTICK, reverse_joystick_mask),
    ):
        quadrant_rank = cycle * counts[category] + (
            slot - category_starts[category]
        )
        lateral_sign = torch.where(
            torch.remainder(quadrant_rank, 2) == 0,
            torch.ones(count, device=device, dtype=dtype),
            -torch.ones(count, device=device, dtype=dtype),
        )
        yaw_sign = torch.where(
            torch.remainder(
                torch.div(quadrant_rank, 2, rounding_mode="floor"), 2
            )
            == 0,
            torch.ones(count, device=device, dtype=dtype),
            -torch.ones(count, device=device, dtype=dtype),
        )
        commands[mask, 1] = joystick_lateral_magnitude[mask] * lateral_sign[mask]
        commands[mask, 2] = joystick_yaw_magnitude[mask] * yaw_sign[mask]

    # Keep the explicit variable in the implementation: a zero stand command
    # is a deliberate stable-anchor category, not an uninitialized residual.
    commands[stand_mask] = 0.0
    return commands, categories


def normalized_signed_axis_progress(
    command: torch.Tensor,
    achieved: torch.Tensor,
    *,
    active_threshold: float = 0.01,
) -> torch.Tensor:
    """Return bounded progress in the commanded sign for one velocity axis.

    An inactive command returns zero.  For an active command, zero response is
    zero reward, reaching the target is +1, and matching the target magnitude
    in the wrong direction is -1.  This supplies a dense, antisymmetric signal
    around the command-insensitive Stage 1 policy while clipping outliers.
    """

    if command.shape != achieved.shape:
        raise ValueError(
            f"command and achieved must have matching shapes, got "
            f"{command.shape!r} and {achieved.shape!r}"
        )
    if not math.isfinite(active_threshold) or active_threshold <= 0.0:
        raise ValueError(
            f"active_threshold must be finite and positive, got {active_threshold}"
        )
    active = torch.abs(command) > active_threshold
    denominator = torch.clamp(torch.abs(command), min=active_threshold)
    progress = torch.clamp(torch.sign(command) * achieved / denominator, -1.0, 1.0)
    return torch.where(active, progress, torch.zeros_like(progress))


def active_axis_gaussian_tracking_reward(
    command: torch.Tensor,
    achieved: torch.Tensor,
    *,
    tracking_std: float,
    reward_scale: float,
    active_threshold: float = 0.01,
) -> torch.Tensor:
    """Return Gaussian tracking reward only where an axis is commanded.

    This avoids the Stage 1 dilution failure in which most environments earned
    a near-maximal lateral/yaw reward for retaining exactly zero response.
    """

    if command.shape != achieved.shape:
        raise ValueError(
            f"command and achieved must have matching shapes, got "
            f"{command.shape!r} and {achieved.shape!r}"
        )
    if not math.isfinite(tracking_std) or tracking_std <= 0.0:
        raise ValueError(f"tracking_std must be finite and positive, got {tracking_std}")
    if not math.isfinite(reward_scale):
        raise ValueError(f"reward_scale must be finite, got {reward_scale}")
    if not math.isfinite(active_threshold) or active_threshold <= 0.0:
        raise ValueError(
            f"active_threshold must be finite and positive, got {active_threshold}"
        )
    active = torch.abs(command) > active_threshold
    reward = torch.exp(-torch.square(command - achieved) / tracking_std**2)
    return torch.where(active, reward * reward_scale, torch.zeros_like(reward))


def sample_uniform_intervals(
    num_samples: int,
    interval_range_s: Sequence[float],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample positive command hold durations in seconds."""

    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    low, high = _validate_range("interval_range_s", interval_range_s)
    if low <= 0.0:
        raise ValueError(f"interval_range_s must be strictly positive, got {interval_range_s!r}")
    intervals = torch.empty(num_samples, device=device, dtype=dtype)
    return intervals.uniform_(low, high, generator=generator)


def advance_resampling_timers(
    time_left_s: torch.Tensor, active_mask: torch.Tensor, step_dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance active timers and return ``(new_time_left, due_mask)``."""

    if time_left_s.ndim != 1 or active_mask.shape != time_left_s.shape:
        raise ValueError(
            "time_left_s and active_mask must be one-dimensional tensors with matching shapes"
        )
    if active_mask.dtype != torch.bool:
        raise ValueError(f"active_mask must be boolean, got {active_mask.dtype}")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt}")
    updated = torch.where(active_mask, time_left_s - step_dt, time_left_s)
    return updated, active_mask & (updated <= 0.0)


__all__ = [
    "COMBINED",
    "LATERAL_ONLY",
    "LONGITUDINAL_ONLY",
    "NUM_COMMAND_CATEGORIES",
    "NUM_RECOVERY_CATEGORIES",
    "NUM_STAGE2_CATEGORIES",
    "NUM_STAGE2D_CATEGORIES",
    "NUM_STAGE2E_CATEGORIES",
    "RECOVERY_COMBINED",
    "RECOVERY_FORWARD",
    "RECOVERY_FORWARD_LATERAL",
    "RECOVERY_FORWARD_YAW",
    "STANDING",
    "STAGE2_FORWARD",
    "STAGE2_FORWARD_LATERAL",
    "STAGE2_FORWARD_YAW",
    "STAGE2D_FORWARD",
    "STAGE2D_FORWARD_YAW",
    "STAGE2D_OBLIQUE",
    "STAGE2E_FORWARD",
    "STAGE2E_FORWARD_JOYSTICK",
    "STAGE2E_LATERAL",
    "STAGE2E_REVERSE",
    "STAGE2E_REVERSE_JOYSTICK",
    "STAGE2E_STAND",
    "STAGE2E_YAW",
    "YAW_ONLY",
    "active_axis_gaussian_tracking_reward",
    "advance_resampling_timers",
    "body_to_navigation_frame",
    "normalized_signed_axis_progress",
    "sample_uniform_intervals",
    "sample_stage1_recovery_commands",
    "sample_stage2_recovery_commands",
    "sample_stage2d_oblique_homotopy_commands",
    "sample_stage2e_joystick_transition_commands",
    "sample_velocity_command_mixture",
]
