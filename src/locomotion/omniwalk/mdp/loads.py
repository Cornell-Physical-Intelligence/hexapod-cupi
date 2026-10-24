"""Accumulate one control's 400 Hz motor torques and floor forces and pack the control telemetry."""

import torch

from locomotion.omniwalk.mdp.frames import navigation
from locomotion.robots.rs05 import RATED_TORQUE_NM


class SubstepLoads:
    """Saturation counts, torque sums and force extrema over the substeps of one control.

    ``toe_index`` selects the six tibias and ``other_index`` the other bodies in the force matrix's body order.
    """

    def __init__(self, target, toe_index, other_index):
        self.toe_index, self.other_index = toe_index, other_index
        self.saturation = torch.zeros_like(target, dtype=torch.long)
        self.torque_square = torch.zeros_like(target)
        self.requested_max = torch.zeros_like(target)
        self.applied_max = torch.zeros_like(target)
        self.other_force_max = torch.zeros(target.shape[0], device=target.device)
        self.tibia_min_force = torch.full((target.shape[0], 6), torch.inf, device=target.device)

    def add(self, requested, applied, forces):
        """Fold in one substep's requested and applied joint torques and its (rows, bodies, 3) floor forces."""
        norms = torch.linalg.vector_norm(forces, dim=-1)
        self.other_force_max = torch.maximum(
            self.other_force_max, norms[:, self.other_index].amax(-1)
        )
        self.tibia_min_force = torch.minimum(self.tibia_min_force, norms[:, self.toe_index])
        self.saturation += requested.abs() > RATED_TORQUE_NM
        self.torque_square += applied.square()
        self.requested_max = torch.maximum(self.requested_max, requested.abs())
        self.applied_max = torch.maximum(self.applied_max, applied.abs())


def control_telemetry(
    state, target, requested, applied, forces, loads, command, action, terminated, truncated
):
    """The telemetry that the task reward, evaluation and training records read after each control."""
    return {
        "root_pose_xyzw": state["root"].clone(),
        "linear_velocity_body": state["linear"].clone(),
        "linear_velocity_nav": navigation(state["linear"]).clone(),
        "angular_velocity_body": state["angular"].clone(),
        "joint_position_rad": state["q"].clone(),
        "joint_velocity_rad_s": state["dq"].clone(),
        "joint_target_rad": target.clone(),
        "computed_torque_nm": requested.clone(),
        "applied_torque_nm": applied.clone(),
        "saturation_count_400hz": loads.saturation,
        "torque_square_sum_400hz": loads.torque_square,
        "requested_torque_abs_max_400hz": loads.requested_max,
        "applied_torque_abs_max_400hz": loads.applied_max,
        "tibia_floor_force_world_n": forces[:, loads.toe_index].clone(),
        "tibia_floor_force_min_norm_400hz": loads.tibia_min_force,
        "other_body_force_max_400hz": loads.other_force_max,
        "toe_xyz_world": state["toe_world"].clone(),
        "toe_xyz_body": state["toe_body"].clone(),
        "command": command.clone(),
        "action": action.clone(),
        "terminated": terminated.clone(),
        "truncated": truncated.clone(),
    }
