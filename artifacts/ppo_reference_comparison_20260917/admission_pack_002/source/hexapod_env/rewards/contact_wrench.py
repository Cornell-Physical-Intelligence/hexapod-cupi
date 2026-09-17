"""Ground-contact wrench costs and their reset-batch summaries."""

from __future__ import annotations

import math

import torch


def bounded_inactive_ground_contact_yaw_moment_cost(
    contact_points_w: torch.Tensor,
    ground_forces_w: torch.Tensor,
    root_positions_w: torch.Tensor,
    ground_contact: torch.Tensor,
    moving_command_active: torch.Tensor,
    yaw_command_active: torch.Tensor,
    *,
    reference_nm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded cost and masked absolute ground-contact yaw moment.

    The moment is the world-Z component of the net ground-reaction wrench
    about the root, ``sum((contact_point - root_position) x force)``.  On the
    nearly level flat-ground task this is the moment that directly drives the
    residual command-frame yaw oscillation.  It also captures asymmetric
    left/right tangential loading without penalizing the vertical-load
    alternation inherent to a tripod gait.

    The Cauchy-shaped cost is 0.5 at ``reference_nm`` and approaches one for
    larger absolute moments.  It is applied only to moving, straight-command
    gait samples (planar motion active and yaw uncommanded).  Because it uses
    only the current contact sample, it has no temporal state that can leak
    across an episode reset.  The second return is the masked raw magnitude in
    N*m for normalization calibration and sensor-path validation.
    """

    if contact_points_w.shape != ground_forces_w.shape:
        raise ValueError(
            "contact_points_w and ground_forces_w must have matching shapes"
        )
    if contact_points_w.ndim != 3 or contact_points_w.shape[-1] != 3:
        raise ValueError(
            "contact_points_w and ground_forces_w must have shape "
            "(num_envs, num_contacts, 3)"
        )
    expected_root_shape = (contact_points_w.shape[0], 3)
    if root_positions_w.shape != expected_root_shape:
        raise ValueError(
            f"root_positions_w must have shape {expected_root_shape}, "
            f"got {tuple(root_positions_w.shape)}"
        )
    expected_contact_shape = contact_points_w.shape[:2]
    if ground_contact.shape != expected_contact_shape:
        raise ValueError(
            f"ground_contact must have shape {expected_contact_shape}, "
            f"got {tuple(ground_contact.shape)}"
        )
    expected_command_shape = (contact_points_w.shape[0],)
    if moving_command_active.shape != expected_command_shape:
        raise ValueError(
            f"moving_command_active must have shape {expected_command_shape}, "
            f"got {tuple(moving_command_active.shape)}"
        )
    if yaw_command_active.shape != expected_command_shape:
        raise ValueError(
            f"yaw_command_active must have shape {expected_command_shape}, "
            f"got {tuple(yaw_command_active.shape)}"
        )
    if ground_contact.dtype != torch.bool:
        raise ValueError(
            f"ground_contact must be boolean, got {ground_contact.dtype}"
        )
    if moving_command_active.dtype != torch.bool:
        raise ValueError(
            "moving_command_active must be boolean, "
            f"got {moving_command_active.dtype}"
        )
    if yaw_command_active.dtype != torch.bool:
        raise ValueError(
            f"yaw_command_active must be boolean, got {yaw_command_active.dtype}"
        )
    if not math.isfinite(float(reference_nm)) or float(reference_nm) <= 0.0:
        raise ValueError(
            f"reference_nm must be finite and positive, got {reference_nm!r}"
        )

    safe_contact_points_w = torch.clamp(
        torch.nan_to_num(
            contact_points_w,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    safe_root_positions_w = torch.clamp(
        torch.nan_to_num(
            root_positions_w,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    safe_ground_forces_w = torch.clamp(
        torch.nan_to_num(
            ground_forces_w,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    contact_offsets_xy_w = (
        safe_contact_points_w[:, :, :2]
        - safe_root_positions_w[:, None, :2]
    )
    per_contact_yaw_moment = (
        contact_offsets_xy_w[:, :, 0] * safe_ground_forces_w[:, :, 1]
        - contact_offsets_xy_w[:, :, 1] * safe_ground_forces_w[:, :, 0]
    )
    net_yaw_moment = torch.sum(
        torch.where(
            ground_contact,
            per_contact_yaw_moment,
            torch.zeros_like(per_contact_yaw_moment),
        ),
        dim=1,
    )
    normalized_moment = torch.clamp(
        torch.nan_to_num(
            net_yaw_moment / float(reference_nm),
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    normalized_moment_l2 = torch.square(normalized_moment)
    unmasked_cost = normalized_moment_l2 / (1.0 + normalized_moment_l2)
    straight_gait_active = moving_command_active & ~yaw_command_active
    zeros = torch.zeros_like(unmasked_cost)
    return (
        torch.where(straight_gait_active, unmasked_cost, zeros),
        torch.where(straight_gait_active, torch.abs(net_yaw_moment), zeros),
    )


def bounded_inactive_bilateral_longitudinal_contact_moment_cost(
    contact_offsets_command_frame: torch.Tensor,
    ground_forces_command_frame: torch.Tensor,
    ground_contact: torch.Tensor,
    longitudinal_command_active: torch.Tensor,
    yaw_command_active: torch.Tensor,
    *,
    reference_nm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cost and magnitude for asymmetric longitudinal traction.

    This isolates ``-sum(lateral_offset * forward_force)``, the bilateral
    longitudinal-traction contribution to contact yaw moment. Equal forward
    thrust on geometrically mirrored sides cancels exactly, regardless of its
    common-mode magnitude. Vertical load, lateral corrective force, and the
    alternating contact count of a tripod gait are not penalized.

    The bounded Cauchy cost is 0.5 at ``reference_nm``. It applies only while
    a longitudinal command is active and yaw is uncommanded. The second return
    value is the masked absolute moment in N*m for scale-calibration telemetry.
    """

    if contact_offsets_command_frame.shape != ground_forces_command_frame.shape:
        raise ValueError(
            "contact offsets and ground forces must have matching shapes"
        )
    if (
        contact_offsets_command_frame.ndim != 3
        or contact_offsets_command_frame.shape[-1] != 3
    ):
        raise ValueError(
            "contact offsets and ground forces must have shape "
            "(num_envs, num_contacts, 3)"
        )
    expected_contact_shape = contact_offsets_command_frame.shape[:2]
    if ground_contact.shape != expected_contact_shape:
        raise ValueError(
            f"ground_contact must have shape {expected_contact_shape}, "
            f"got {tuple(ground_contact.shape)}"
        )
    expected_command_shape = (contact_offsets_command_frame.shape[0],)
    for name, command_mask in (
        ("longitudinal_command_active", longitudinal_command_active),
        ("yaw_command_active", yaw_command_active),
    ):
        if command_mask.shape != expected_command_shape:
            raise ValueError(
                f"{name} must have shape {expected_command_shape}, "
                f"got {tuple(command_mask.shape)}"
            )
        if command_mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean, got {command_mask.dtype}")
    if ground_contact.dtype != torch.bool:
        raise ValueError(
            f"ground_contact must be boolean, got {ground_contact.dtype}"
        )
    if not math.isfinite(float(reference_nm)) or float(reference_nm) <= 0.0:
        raise ValueError(
            f"reference_nm must be finite and positive, got {reference_nm!r}"
        )

    safe_offsets = torch.clamp(
        torch.nan_to_num(
            contact_offsets_command_frame,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    safe_forces = torch.clamp(
        torch.nan_to_num(
            ground_forces_command_frame,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    per_contact_moment = -safe_offsets[:, :, 1] * safe_forces[:, :, 0]
    signed_moment_nm = torch.sum(
        torch.where(
            ground_contact,
            per_contact_moment,
            torch.zeros_like(per_contact_moment),
        ),
        dim=1,
    )
    absolute_moment_nm = torch.abs(signed_moment_nm)
    normalized_moment = torch.clamp(
        absolute_moment_nm / float(reference_nm), min=0.0, max=1.0e6
    )
    normalized_l2 = torch.square(normalized_moment)
    unmasked_cost = normalized_l2 / (1.0 + normalized_l2)
    valid_command = longitudinal_command_active & ~yaw_command_active
    zeros = torch.zeros_like(unmasked_cost)
    return (
        torch.where(valid_command, unmasked_cost, zeros),
        torch.where(valid_command, absolute_moment_nm, zeros),
    )


def reset_batch_time_mean_and_p50(
    integrated_values: torch.Tensor,
    episode_elapsed_s: torch.Tensor,
    *,
    min_elapsed_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize per-episode time means over the current reset batch.

    Episode metrics are accumulated as time integrals.  Reduce each selected
    environment to its own time mean first, then report the reset-batch mean
    and interpolated 50th percentile.  This keeps the percentile comparable to
    the existing ``Episode_Metric`` mean instead of weighting long episodes
    more heavily than short ones.
    """

    if (
        integrated_values.ndim != 1
        or episode_elapsed_s.shape != integrated_values.shape
    ):
        raise ValueError(
            "integrated_values and episode_elapsed_s must be matching 1-D tensors"
        )
    if integrated_values.numel() == 0:
        raise ValueError("reset-batch tensors must be non-empty")
    if not math.isfinite(float(min_elapsed_s)) or float(min_elapsed_s) <= 0.0:
        raise ValueError(
            f"min_elapsed_s must be finite and positive, got {min_elapsed_s!r}"
        )

    elapsed = torch.clamp(episode_elapsed_s, min=float(min_elapsed_s))
    per_episode_time_mean = integrated_values / elapsed
    return (
        torch.mean(per_episode_time_mean),
        torch.quantile(per_episode_time_mean, 0.5),
    )
