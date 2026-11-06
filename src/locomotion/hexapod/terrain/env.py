# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain v2.1 task: the original ``HexapodEnv`` plus the terrain curriculum, the swing-clearance reward, the
speed-based success metric and the optional command-speed curriculum. ``HexapodEnv`` itself is unchanged."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from ..hexapod_env import HexapodEnv
from .curriculum import CommandCurriculum, episode_outcome, swing_clearance, terrain_moves
from .env_cfg import HexapodTerrainEnvCfg
from .terrains_cfg import terrain_type_names

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
TOE_OFFSET = (0.130, 0.0, 0.0)
"""Toe tip in the tibia frame [m]."""


class HexapodTerrainEnv(HexapodEnv):
    cfg: HexapodTerrainEnvCfg

    def __init__(self, cfg: HexapodTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        # The command curriculum owns the largest commanded speed. It is written to cfg.command_lin_vel_max, which
        # HexapodEnv reads for command sampling, the tracking tolerance and the air-time gate.
        self._command_curriculum = None
        if cfg.command_curriculum:
            self._command_curriculum = CommandCurriculum(
                cfg.command_curriculum_start,
                cfg.command_curriculum_step,
                cfg.command_curriculum_max,
                cfg.command_curriculum_success,
                cfg.command_curriculum_window_episodes,
            )
            cfg.command_lin_vel_max = self._command_curriculum.value
        super().__init__(cfg, render_mode, **kwargs)
        self._v_along_cmd_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["swing_clearance"] = torch.zeros(self.num_envs, device=self.device)
        self._tibia_body_ids = [self._robot.body_names.index(f"{leg}_tibia") for leg in LEGS]
        self._toe_offset = torch.tensor(TOE_OFFSET, device=self.device)
        self._has_terrain_levels = getattr(self._terrain, "terrain_origins", None) is not None
        self.terrain_type_names = (
            terrain_type_names(self.cfg.terrain.terrain_generator) if self._has_terrain_levels else ["flat"]
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        # Filter collisions between envs on GPU too (HexapodEnv does it on CPU only). Measured on Isaac Lab 3.0 / GPU:
        # with a generator terrain, envs that share a tile origin spawned inside each other (10+ kN body forces).
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        moving = torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.cfg.command_lin_vel_max
        if self.cfg.swing_clearance_reward_scale != 0.0:
            pos = self._robot.data.body_link_pos_w.torch[:, self._tibia_body_ids]
            quat = self._robot.data.body_link_quat_w.torch[:, self._tibia_body_ids]
            n = pos.shape[0]
            toe = pos + quat_apply(quat.reshape(-1, 4), self._toe_offset.expand(n * len(LEGS), 3)).reshape(n, -1, 3)
            forces = self._contact_sensor.data.net_forces_w.torch[:, self._feet_ids]
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

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        ids = env_ids.long()
        # outcome of the ending episodes, before HexapodEnv clears the accumulators
        steps = self._step_count[ids]
        ran = steps > 0
        cmd_speed = torch.linalg.norm(self._commands[ids, :2], dim=-1)
        fell = self.reset_terminated[ids]
        success, slow, speed_err, _ = episode_outcome(
            self._v_along_cmd_sum[ids],
            steps,
            cmd_speed,
            fell,
            self.reset_time_outs[ids],
            self.cfg.command_lin_vel_max,
            self.cfg.success_speed_tol_ratio,
            self.cfg.success_min_cmd_ratio,
            self.cfg.curriculum_move_down_speed_ratio,
        )
        # terrain curriculum: the new origins are used by HexapodEnv's reset below
        if self.cfg.terrain_curriculum and self._has_terrain_levels:
            move_up, move_down = terrain_moves(success, fell, slow, ran)
            self._terrain.update_env_origins(ids, move_up, move_down)
        # command curriculum: the new max is used by HexapodEnv's command sampling below
        if self._command_curriculum is not None:
            self.cfg.command_lin_vel_max = self._command_curriculum.update(cmd_speed, success, ran)
        self._v_along_cmd_sum[ids] = 0.0
        super()._reset_idx(env_ids)
        log = self.extras["log"]
        log["Metrics/success_eval"] = success.float().mean().item()
        log["Metrics/speed_err_along_cmd"] = speed_err.mean().item()
        log["Metrics/falls_per_episode_s"] = (fell.float().sum() / (steps.sum() * self.step_dt).clamp_min(1e-6)).item()
        if self._has_terrain_levels:
            log["Curriculum/terrain_level"] = self._terrain.terrain_levels.float().mean().item()
        if self._command_curriculum is not None:
            log["Curriculum/command_lin_vel_max"] = self.cfg.command_lin_vel_max
