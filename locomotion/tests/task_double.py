"""CPU telemetry double for active walking reward and learner tests."""
from types import SimpleNamespace
from pathlib import Path
import json
import torch
from locomotion.omniwalk.robot import JOINT_NAMES
ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT/"robot/hexapod_mkii_updated_v1/geometry/geometry.json"
MODEL = json.loads((ROOT/"robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json").read_text())
HEIGHT = .1
LOADED = torch.tensor([[0., 0., 24.]]).expand(6, -1)


class NativeDouble:
    def __init__(self, n=3):
        self.num_envs, self.device = n, "cpu"
        self.cfg = SimpleNamespace(control_dt=.02, decimation=8, target_slew_rad=.04)
        self.joint_names = list(JOINT_NAMES)
        self.lower, self.upper = torch.full((18,), -1.), torch.full((18,), 1.)
        self.position = torch.zeros(n, 18)
        self.gravity_z = torch.full((n,), -1.)
        self.reference_metadata = {"root_height_m": .1}
        self.reset_height = .1
        self.model, self.geometry_path = MODEL, GEOMETRY
        self.origins = torch.zeros(n, 3)
        self.origins[:, 0] = torch.arange(n)*2.
        roots = torch.zeros(n, 7)
        roots[:, :3] = self.origins
        roots[:, 2], roots[:, 6] = .1, 1.
        self.current = {"root": roots, "linear": torch.zeros(n, 3)}
        self.held = torch.zeros(n, 18)
        self.commands = torch.zeros(n, 3)
        self.history = torch.zeros(n, 210)
        self.amp = torch.zeros(n, 61)
        self.steps = 0
        self.reset_calls = []
        self.telemetry = {}
        self.rate = torch.zeros(n, 18)
        self.last_result = None

    def state(self):
        obs = torch.cat((self.history, self.commands, self.held), -1)
        obs[:, 173] = self.gravity_z
        return {"obs": obs.clone(), "critic": torch.cat((obs, torch.zeros(self.num_envs, 3)), -1), "amp": self.amp.clone()}

    def reset(self, indices):
        self.reset_calls.append(indices.clone())
        self.history[indices] = 0
        self.amp[indices] = 0
        self.held[indices] = 0
        self.current["root"][indices, :3] = self.origins[indices]
        self.current["root"][indices, 2] = self.reset_height
        self.current["linear"][indices] = 0
        return self.state()

    def step(self, action):
        self.steps += 1
        self.history += 1
        self.amp += .01
        pose = self.current["root"].clone()
        velocity = self.commands.clone()
        velocity[:, 2] = 0
        angular = torch.zeros(self.num_envs, 3)
        angular[:, 2] = self.commands[:, 2]
        self.telemetry = {"linear_velocity_nav": velocity, "angular_velocity_body": angular,
            "root_pose_xyzw": pose, "joint_velocity_rad_s": self.rate.clone(),
            "joint_position_rad": self.position.clone(),
            "joint_target_rad": self.held.clone(), "torque_square_sum_400hz": torch.zeros(self.num_envs, 18),
            "saturation_count_400hz": torch.zeros(self.num_envs, 18, dtype=torch.long),
            "other_body_force_max_400hz": torch.zeros(self.num_envs), "command": self.commands.clone()}
        output = self.state()
        output.update(reward=torch.full((self.num_envs,), -999.), terminated=torch.zeros(self.num_envs, dtype=torch.bool),
                      truncated=torch.zeros(self.num_envs, dtype=torch.bool))
        self.last_result = {k: v.clone() for k, v in output.items()}
        return output

class RewardDouble(NativeDouble):
    """NativeDouble plus the joint velocity, raw action and requested torque version 2 reads."""

    def __init__(self, n=3):
        super().__init__(n)
        self.current["dq"] = torch.zeros(n, 18)
        self.velocity = None
        self.requested = torch.zeros(n, 18)

    def reset(self, indices):
        output = super().reset(indices)
        self.current["dq"][indices] = self.rate[indices]
        return output

    def step(self, action):
        output = super().step(action)
        if self.velocity is not None:
            self.telemetry["linear_velocity_nav"] = self.velocity[:, :3].clone()
            self.telemetry["angular_velocity_body"][:, 2] = self.velocity[:, 3]
        self.telemetry.update(action=torch.as_tensor(action, dtype=torch.float32).clone(),
                              requested_torque_abs_max_400hz=self.requested.clone())
        self.current["dq"] = self.rate.clone()
        return output

class GaitDouble(RewardDouble):
    """RewardDouble plus the toe forces, toe positions, limits and episode count version 4 reads."""

    def __init__(self, n=2):
        super().__init__(n)
        self.forces = LOADED.expand(n, -1, -1).clone()
        self.toes = torch.zeros(n, 6, 3)
        self.world = torch.zeros(n, 6, 3)
        self.current["toe_body"] = self.toes.clone()
        self.current["toe_world"] = self.world.clone()
        self.episode_steps = torch.zeros(n, dtype=torch.long)
        self.lower, self.upper = torch.full((18,), -1.), torch.full((18,), 1.)
        self.neutral = torch.zeros(18)
        self.cfg.action_scale_rad = .35

    def step(self, action):
        output = super().step(action)
        self.episode_steps += 1
        self.telemetry.update(tibia_floor_force_world_n=self.forces.clone(), toe_xyz_body=self.toes.clone(),
                              toe_xyz_world=self.world.clone(), computed_torque_nm=torch.zeros(self.num_envs, 18))
        self.current["toe_body"] = self.toes.clone()
        self.current["toe_world"] = self.world.clone()
        return output

def telemetry(commands, **overrides):
    """One control of a robot at rest on six loaded feet, far from every limit."""
    n = len(commands)
    zeros = torch.zeros(n, 18)
    pose = torch.zeros(n, 7)
    pose[:, 2], pose[:, 6] = HEIGHT, 1.
    result = {"linear_velocity_nav": torch.zeros(n, 3), "angular_velocity_body": torch.zeros(n, 3),
        "root_pose_xyzw": pose, "joint_velocity_rad_s": zeros, "previous_joint_velocity_rad_s": zeros,
        "joint_target_rad": zeros, "previous_joint_target_rad": zeros, "torque_square_sum_400hz": zeros,
        "requested_torque_abs_max_400hz": zeros, "computed_torque_nm": zeros,
        "other_body_force_max_400hz": torch.zeros(n), "tibia_floor_force_world_n": LOADED.expand(n, -1, -1).clone(),
        "tracked_velocity_nav": torch.zeros(n, 3), "scheduled_swing": torch.zeros(n, 6, dtype=torch.bool),
        "toe_velocity_nav": torch.zeros(n, 6, 2), "toe_xyz_nav": torch.zeros(n, 6, 2),
        "joint_limit_margin_rad": torch.ones(n, 18), "executed_action": zeros, "command": commands.clone(), "action": zeros}
    result.update(overrides)
    return result

def lifted(rows, feet):
    force = LOADED.expand(rows, -1, -1).clone()
    force[:, feet] = 0.
    return force
