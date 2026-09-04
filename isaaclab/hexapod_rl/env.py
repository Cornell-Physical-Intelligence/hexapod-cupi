"""Compatibility shim for ``hexapod_rl.env``.

The implementation now lives in ``hexapod_env.env``
(``packages/hexapod_env/hexapod_env/env.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.env import *  # noqa: F401,F403
from hexapod_env.env import (
    HexapodEnv,
    advance_gait_phase,
    applied_torque_slew_l2,
    apply_command_conditioned_stand_action_scale,
    assign_tripod_pairs_from_foot_offsets,
    bounded_inactive_bilateral_longitudinal_contact_moment_cost,
    bounded_inactive_ground_contact_yaw_moment_cost,
    bounded_inactive_yaw_rate_slew_cost,
    capped_normalized_axis_error,
    gait_phase_contact_reward,
    limit_processed_joint_target_slew,
    max_joint_rated_torque_excess_l1,
    max_joint_rated_torque_excess_l2,
    normalized_deck_stability_reward,
    reset_batch_time_mean_and_p50,
    reset_safe_exponential_moving_average,
    select_command_conditioned_nominal_height,
    select_support_contact_target,
    swing_clearance_reward,
    tripod_expected_stance,
)

__all__ = [
    "HexapodEnv",
    "advance_gait_phase",
    "applied_torque_slew_l2",
    "apply_command_conditioned_stand_action_scale",
    "assign_tripod_pairs_from_foot_offsets",
    "bounded_inactive_bilateral_longitudinal_contact_moment_cost",
    "bounded_inactive_ground_contact_yaw_moment_cost",
    "bounded_inactive_yaw_rate_slew_cost",
    "capped_normalized_axis_error",
    "gait_phase_contact_reward",
    "limit_processed_joint_target_slew",
    "max_joint_rated_torque_excess_l1",
    "max_joint_rated_torque_excess_l2",
    "normalized_deck_stability_reward",
    "reset_batch_time_mean_and_p50",
    "reset_safe_exponential_moving_average",
    "select_command_conditioned_nominal_height",
    "select_support_contact_target",
    "swing_clearance_reward",
    "tripod_expected_stance",
]
