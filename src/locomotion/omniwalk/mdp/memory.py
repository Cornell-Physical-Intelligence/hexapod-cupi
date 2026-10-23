"""Hold the command schedule and the reward memory that a manager-based walking task carries between controls.

``CommandSchedule.draw`` makes the same random draws as ``omniwalk.task.TrainingTask._resample`` and
``RewardMemory`` keeps the memory of ``TrainingTask.reward_telemetry``; ``surrogate/tests/test_manager_memory.py``
compares both with the training task bit for bit.
"""

import math

import torch

from locomotion.omniwalk.mdp.commands import TaskConfig, command_bank
from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation
from locomotion.omniwalk.mdp.rewards import (
    CONTROL_DT_S,
    REWARD_CONFIG,
    StrideWindow,
    displacement_velocity,
    measured_reward,
    scheduled_swing,
)

HOLD_CONTROLS = (100, 250)


class CommandSchedule:
    """Draw bank commands, the integer zero-command quota, turn draws and 100 to 250 control holds."""

    def __init__(
        self, config=None, turn_draw_fraction=REWARD_CONFIG.turn_draw_fraction, device="cpu"
    ):
        self.config = TaskConfig() if config is None else config
        self.config.validate()
        if not 0 <= turn_draw_fraction <= 1:
            raise ValueError("turn_draw_fraction is a share from 0 to 1")
        self.turn_draw_fraction, self.device = turn_draw_fraction, device
        self.bank = torch.tensor(command_bank(self.config), dtype=torch.float32, device=device)
        self.rng = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.command_draws = 0
        self.zero_command_draws = 0

    def draw(self, size):
        """Return ``size`` commands and their hold lengths in controls."""
        bank_indices = torch.randint(len(self.bank), (size,), generator=self.rng)
        values = self.bank[bank_indices.to(self.device)].clone()
        # An integer quota avoids a finite pilot accidentally seeing no stops.
        before = math.ceil(self.config.zero_command_fraction * self.command_draws)
        after = math.ceil(self.config.zero_command_fraction * (self.command_draws + size))
        zero_count = after - before
        selected_zero = torch.randperm(size, generator=self.rng)[:zero_count].to(self.device)
        values[selected_zero] = 0
        durations = torch.randint(
            HOLD_CONTROLS[0], HOLD_CONTROLS[1] + 1, (size,), generator=self.rng, dtype=torch.long
        ).to(self.device)
        self.command_draws += size
        self.zero_command_draws += zero_count
        share = self.turn_draw_fraction
        if share and size:
            chosen = (torch.rand(size, generator=self.rng) < share).to(self.device)
            chosen &= (values != 0).any(-1)
            sign = (torch.randint(0, 2, (size,), generator=self.rng) * 2 - 1).to(values)
            turns = torch.zeros(size, 3, device=self.device, dtype=values.dtype)
            turns[:, 2] = sign * self.config.yaw_rate_rad_s
            values[chosen] = turns[chosen]
        return values, durations


class RewardMemory:
    """Tracking windows, liftoff points, air time, command age and the previous control's state."""

    def __init__(
        self,
        num_envs,
        device,
        *,
        neutral,
        lower,
        upper,
        action_scale,
        nominal_height,
        config=REWARD_CONFIG,
    ):
        config.validate()
        self.config, self.nominal_height = config, nominal_height
        self.neutral, self.lower, self.upper, self.action_scale = (
            neutral,
            lower,
            upper,
            action_scale,
        )
        self.gate_window = StrideWindow(num_envs, config.gate_window_controls, 1, device)
        self.stride_window = StrideWindow(num_envs, config.stride_window_controls, 3, device)
        self.window = StrideWindow(num_envs, config.tracking_window_controls, 3, device)
        self.liftoff_world = torch.zeros(num_envs, 6, 3, device=device)
        self.lifted = torch.zeros(num_envs, 6, dtype=torch.bool, device=device)
        self.air_time = torch.zeros(num_envs, 6, device=device)
        self.in_contact = torch.ones(num_envs, 6, dtype=torch.bool, device=device)
        self.command_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.previous_joint_velocity = torch.zeros(num_envs, 18, device=device)
        self.previous_pose = torch.zeros(num_envs, 7, device=device)
        self.previous_toe_world = torch.zeros(num_envs, 6, 3, device=device)
        self.previous_target = neutral.expand(num_envs, -1).clone()

    def restart(self, rows, state, commands, held):
        """Begin new episodes in ``rows`` from their measured reset ``state``; the liftoff points persist."""
        self.previous_target[rows] = held[rows]
        self.previous_joint_velocity[rows] = state["dq"][rows].to(self.previous_joint_velocity)
        self.previous_pose[rows] = state["root"][rows].to(self.previous_pose)
        self.previous_toe_world[rows] = (
            state["toe_world"][rows].detach().to(self.previous_toe_world)
        )
        self.window.restart(rows, commands[rows])
        self.command_age[rows] = 0
        self.air_time[rows] = 0
        self.in_contact[rows] = True
        self.gate_window.restart(rows, commands[rows])
        self.stride_window.restart(rows, commands[rows])
        self.lifted[rows] = False

    def reward(self, telemetry, command, terminated, episode_steps):
        """Score one completed control under the command it held; return the reward, its components and the inputs."""
        values = self.telemetry(telemetry, command, episode_steps)
        reward, components = measured_reward(
            values, command, terminated, self.config, self.nominal_height
        )
        self.previous_joint_velocity = (
            telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        )
        self.previous_pose = telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        self.previous_target = telemetry["joint_target_rad"].detach().clone()
        return reward, components, values

    def telemetry(self, t, command, episode_steps):
        velocity = displacement_velocity(
            t["root_pose_xyzw"].to(self.previous_pose), self.previous_pose
        )
        stride_yaw = self.gate_window.update(velocity[:, 2:], command)[:, 0]
        stride = self.stride_window.update(velocity, command)
        # Liftoff and touchdown by the same contact threshold as v4, read before v4 advances its contact state.
        contact = (
            torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1).to(self.lifted.device)
            > self.config.contact_force_n
        )
        world = t["toe_xyz_world"].detach().to(self.liftoff_world)
        liftoff = ~contact & self.in_contact.to(contact.device)
        landed = contact & ~self.in_contact.to(contact.device) & self.lifted
        pose = t["root_pose_xyzw"].to(self.liftoff_world)
        travel = navigation(
            inverse_rotate(pose[:, None, 3:].expand(-1, 6, -1), world - self.liftoff_world)
        )[..., :2]
        self.liftoff_world = torch.where(liftoff[..., None], world, self.liftoff_world)
        self.lifted = (self.lifted | liftoff) & ~(contact & ~self.in_contact.to(contact.device))

        pose = t["root_pose_xyzw"].to(self.previous_pose)
        changed = (command.to(self.window.command) != self.window.command).any(-1)
        self.command_age = (
            torch.where(
                changed.to(self.command_age.device),
                torch.zeros_like(self.command_age),
                self.command_age,
            )
            + 1
        )
        tracked = self.window.update(displacement_velocity(pose, self.previous_pose), command)
        contact = (
            torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1)
            > self.config.contact_force_n
        )
        touchdown = contact & ~self.in_contact
        completed = self.air_time.clone()
        self.air_time = torch.where(
            contact, torch.zeros_like(self.air_time), self.air_time + CONTROL_DT_S
        )
        self.in_contact = contact
        # Toe velocity over the floor, expressed in the body's navigation axes.
        world = t["toe_xyz_world"].detach().to(self.previous_toe_world)
        body_axes = pose[:, None, 3:].expand(-1, 6, -1)
        toe_velocity = navigation(
            inverse_rotate(body_axes, (world - self.previous_toe_world) / CONTROL_DT_S)
        )[..., :2]
        self.previous_toe_world = world.clone()
        q = t["joint_position_rad"].to(contact.device)
        # The control just completed started at episode step minus one.
        base = {
            **t,
            "tracked_velocity_nav": tracked,
            "previous_joint_velocity_rad_s": self.previous_joint_velocity,
            "previous_joint_target_rad": self.previous_target,
            "touchdown": touchdown,
            "air_time_s": completed,
            "toe_velocity_nav": toe_velocity,
            "toe_xyz_nav": navigation(t["toe_xyz_body"].detach())[..., :2],
            "scheduled_swing": scheduled_swing(episode_steps - 1, self.config).to(contact.device),
            "joint_limit_margin_rad": torch.minimum(q - self.lower.to(q), self.upper.to(q) - q),
            "executed_action": (t["joint_target_rad"].to(q) - self.neutral.to(q))
            / self.action_scale,
            "command_age_controls": self.command_age.clone(),
        }
        return {
            **base,
            "gate_yaw_rate_rad_s": stride_yaw,
            "stride_velocity_nav": stride,
            "stride_touchdown": landed,
            "stride_displacement_nav": torch.where(
                landed[..., None], travel, torch.zeros_like(travel)
            ),
        }
