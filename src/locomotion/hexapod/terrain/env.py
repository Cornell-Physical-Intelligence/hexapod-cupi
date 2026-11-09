# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain v2.2 task: the original ``HexapodEnv`` plus the terrain curriculum, the swing-clearance reward, the
speed-based success metric, an off-course failure and the optional command-speed curriculum. ``HexapodEnv`` itself is
unchanged."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import quat_apply

from ..hexapod_env import HexapodEnv
from .curriculum import (
    EPISODES_KEY,
    SPEED_ERR_KEY,
    SUCCESS_KEY,
    SUCCESSES_KEY,
    CommandCurriculum,
    cleared_episode_log,
    episode_log,
    episode_outcome,
    swing_clearance,
    terrain_moves,
)
from .env_cfg import HexapodTerrainEnvCfg
from .geometry import course_bounds
from .terrains_cfg import terrain_type_names

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
TOE_OFFSET = (0.130, 0.0, 0.0)
"""Toe tip in the tibia frame [m]."""


class HexapodTerrainEnv(HexapodEnv):
    cfg: HexapodTerrainEnvCfg

    def __init__(self, cfg: HexapodTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        # The command curriculum owns the largest *sampled* speed only. cfg.command_lin_vel_max stays fixed, because
        # HexapodEnv also reads it as the tracking-reward sigma, the air-time gate and the success_rate threshold, and
        # this task reads it as the success tolerance floor and the swing-clearance gate.
        self._command_curriculum = None
        if cfg.command_curriculum:
            if cfg.command_curriculum_start < cfg.command_lin_vel_min:
                raise ValueError(
                    f"command_curriculum_start {cfg.command_curriculum_start} is below command_lin_vel_min "
                    f"{cfg.command_lin_vel_min}; the command sampler would draw from an empty range"
                )
            self._command_curriculum = CommandCurriculum(
                cfg.command_curriculum_start,
                cfg.command_curriculum_step,
                cfg.command_curriculum_max,
                cfg.command_curriculum_success,
                cfg.command_curriculum_window_episodes,
            )
        super().__init__(cfg, render_mode, **kwargs)
        self._v_along_cmd_sum = torch.zeros(self.num_envs, device=self.device)
        self._off_course = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_sums["swing_clearance"] = torch.zeros(self.num_envs, device=self.device)
        # tibia bodies and their contact sensors, both in LEGS order, so foot i's height pairs with foot i's contact
        self._tibia_body_ids = [self._robot.body_names.index(f"{leg}_tibia") for leg in LEGS]
        self._tibia_sensor_ids, _ = self._contact_sensor.find_sensors(
            [f"{leg}_tibia" for leg in LEGS], preserve_order=True
        )
        self._toe_offset = torch.tensor(TOE_OFFSET, device=self.device)
        self._has_terrain_levels = getattr(self._terrain, "terrain_origins", None) is not None
        self._course_lo = self._course_hi = None
        self._type_names = None
        if self._has_terrain_levels:
            gen = self.cfg.terrain.terrain_generator
            lo, hi = course_bounds(
                self._terrain.terrain_origins[..., :2].cpu().numpy(), gen.size[0], gen.border_width
            )
            self._course_lo = torch.tensor(lo, dtype=torch.float, device=self.device)
            self._course_hi = torch.tensor(hi, dtype=torch.float, device=self.device)
            column_names = terrain_type_names(gen)
            self._type_names = list(dict.fromkeys(column_names))
            self._column_type = torch.tensor(
                [self._type_names.index(name) for name in column_names], device=self.device
            )
        self._episode_log_keys = [SUCCESS_KEY, SPEED_ERR_KEY, EPISODES_KEY, SUCCESSES_KEY] + [
            f"{SUCCESS_KEY}/{name}" for name in self._type_names or []
        ]

    def _setup_scene(self):
        super()._setup_scene()
        # HexapodEnv filters collisions between envs on CPU only. Measured on Isaac Lab 3.0 / GPU: with a generator
        # terrain, envs that share a tile origin spawned inside each other (10+ kN body forces).
        if self.device != "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def step(self, action):
        # entries of the previous reset are not outcomes of this step; see curriculum.cleared_episode_log
        if "log" in self.extras:
            self.extras["log"] = cleared_episode_log(self.extras["log"], self._episode_log_keys)
        return super().step(action)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()
        if self._course_lo is not None:
            xy = self._robot.data.root_pos_w.torch[:, :2]
            self._off_course = torch.any((xy < self._course_lo) | (xy > self._course_hi), dim=1)
            died = died | self._off_course
        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        moving = torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.cfg.command_lin_vel_max
        if self.cfg.swing_clearance_reward_scale != 0.0:
            pos = self._robot.data.body_link_pos_w.torch[:, self._tibia_body_ids]
            quat = self._robot.data.body_link_quat_w.torch[:, self._tibia_body_ids]
            n = pos.shape[0]
            toe = pos + quat_apply(quat.reshape(-1, 4), self._toe_offset.expand(n * len(LEGS), 3)).reshape(n, -1, 3)
            forces = self._contact_sensor.data.net_forces_w.torch[:, self._tibia_sensor_ids]
            in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
            clear = swing_clearance(toe[..., 2], in_contact, self.cfg.swing_clearance_target) * moving
            term = clear * self.cfg.swing_clearance_reward_scale * self.step_dt
            reward = reward + term
            self._episode_sums["swing_clearance"] += term
        # speed along the command direction, for the success metric and the curriculum
        v_xy = self._robot.data.root_lin_vel_b.torch[:, :2]
        cmd_xy = self._commands[:, :2]
        cmd_dir = cmd_xy / torch.linalg.norm(cmd_xy, dim=-1).clamp_min(1e-6).unsqueeze(1)
        self._v_along_cmd_sum += torch.sum(v_xy * cmd_dir, dim=-1)
        return reward

    def _resample_lin_commands(self, ids: torch.Tensor, lin_max: float):
        """HexapodEnv's linear-command sampling with ``lin_max`` in place of ``cfg.command_lin_vel_max``."""
        if self.cfg.command_forward_only:
            mag = torch.zeros(len(ids), device=self.device).uniform_(self.cfg.command_lin_vel_min, lin_max)
            sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
            self._commands[ids, 0] = 0.0
            self._commands[ids, 1] = mag * sign
        else:
            self._commands[ids, :2] = torch.zeros_like(self._commands[ids, :2]).uniform_(-lin_max, lin_max)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        ids = env_ids.long()
        # outcome of the ending episodes, before HexapodEnv clears the accumulators
        steps = self._step_count[ids]
        ran = steps > 0
        cmd_speed = torch.linalg.norm(self._commands[ids, :2], dim=-1)
        terminated = self.reset_terminated[ids]
        off_course = self._off_course[ids] & terminated
        fell = terminated & ~off_course
        success, slow, speed_err, _ = episode_outcome(
            self._v_along_cmd_sum[ids],
            steps,
            cmd_speed,
            terminated,  # a fall or leaving the course: never a success
            self.reset_time_outs[ids],
            self.cfg.command_lin_vel_max,
            self.cfg.success_speed_tol_ratio,
            self.cfg.success_min_cmd_ratio,
            self.cfg.curriculum_move_down_speed_ratio,
        )
        # terrain curriculum: the new origins are used by HexapodEnv's reset below
        if self.cfg.terrain_curriculum and self._has_terrain_levels:
            move_up, move_down = terrain_moves(success, terminated, slow, ran)
            self._terrain.update_env_origins(ids, move_up, move_down)
        self._v_along_cmd_sum[ids] = 0.0
        self._off_course[ids] = False
        super()._reset_idx(env_ids)
        # command curriculum: replaces the linear commands HexapodEnv just sampled, with the curriculum's top speed
        if self._command_curriculum is not None:
            lin_max = self._command_curriculum.update(cmd_speed, success, ran)
            self._resample_lin_commands(ids, lin_max)
        log = self.extras["log"]
        type_ids = self._column_type[self._terrain.terrain_types[ids]] if self._type_names else None
        log.update(episode_log(success, ran, speed_err, type_ids, self._type_names))
        log["Metrics/falls_per_episode_s"] = (
            (fell & ran).float().sum() / (steps.sum() * self.step_dt).clamp_min(1e-6)
        ).item()
        log["Episode_Termination/off_course"] = torch.count_nonzero(off_course).item()
        if self._has_terrain_levels:
            log["Curriculum/terrain_level"] = self._terrain.terrain_levels.float().mean().item()
        if self._command_curriculum is not None:
            log["Curriculum/command_lin_vel_max"] = self._command_curriculum.value
