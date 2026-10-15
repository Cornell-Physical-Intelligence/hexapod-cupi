"""Run an exported policy in the hexapod env and report what pose it actually settles into.

Run (Isaac Lab root in the container, repo root on PYTHONPATH):
  ./isaaclab.sh -p <repo>/hexapod/tools/diag_policy_rollout.py --headless --policy <run>/exported/policy.pt --cmd_max 0.15
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy",
    required=True,
    help="TorchScript policy exported by play.py",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--settle", type=int, default=100, help="Steps to skip before averaging.")
parser.add_argument("--cmd_max", type=float, default=None, help="Override cfg.command_lin_vel_max (use the training value).")
parser.add_argument("--fixed_cmd", type=float, default=None, help="If set, command every env with this speed along body -y (the club forward).")
parser.add_argument("--set", action="append", default=[], help="cfg override like env.gait_reward_scale=1.0 (env. prefix optional); repeatable")
parser.add_argument("--usd", default=None, help="Override the robot USD path (e.g. the simple-collision asset).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402

cfg = HexapodFlatEnvCfg()
cfg.scene.num_envs = args.num_envs
if args.cmd_max is not None:
    cfg.command_lin_vel_max = args.cmd_max
for item in args.set:
    key, val = item.split("=", 1)
    key = key[4:] if key.startswith("env.") else key
    if key.startswith("agent."):
        continue
    assert hasattr(cfg, key), f"unknown cfg field {key}"
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        setattr(cfg, key, val.lower() in ("true", "1", "yes"))
    elif isinstance(cur, (int, float)):
        setattr(cfg, key, type(cur)(float(val)))
    else:
        setattr(cfg, key, val)
    print(f"[rollout] cfg.{key} = {getattr(cfg, key)}")
if args.usd:
    cfg.robot.spawn.usd_path = args.usd
env = HexapodEnv(cfg)
policy = torch.jit.load(args.policy, map_location=env.device)
policy.eval()
obs, _ = env.reset()
robot, sensor = env._robot, env._contact_sensor
links = {g: [i for i, n in enumerate(robot.body_names) if n == g or n.endswith("_" + g)] for g in ("body", "coxa", "femur", "tibia")}
joints = {g: [i for i, n in enumerate(robot.joint_names) if n.endswith("_" + g)] for g in ("coxa_yaw", "femur_pitch", "tibia_pitch")}

sums, n = {}, 0
min_root_z = 1.0
for i in range(args.steps):
    with torch.no_grad():
        actions = policy(obs["policy"])
    if args.fixed_cmd is not None:
        env._commands[:, 0] = 0.0
        env._commands[:, 1] = -args.fixed_cmd
        env._commands[:, 2] = 0.0
    obs, _, died, _, _ = env.step(actions)
    if args.fixed_cmd is not None:
        env._commands[:, 0] = 0.0
        env._commands[:, 1] = -args.fixed_cmd
        env._commands[:, 2] = 0.0
    if i < args.settle:
        continue
    f = torch.linalg.norm(sensor.data.net_forces_w.torch, dim=-1)  # (envs, links)
    rec = {
        "root z [m]": robot.data.root_pos_w.torch[:, 2].mean(),
        "|torque| mean [N·m]": robot.data.applied_torque.torch.abs().mean(),
        "|cmd xy| [m/s]": torch.linalg.norm(env._commands[:, :2], dim=1).mean(),
        "|actual xy| [m/s]": torch.linalg.norm(robot.data.root_lin_vel_b.torch[:, :2], dim=1).mean(),
        "deaths/step": died.float().mean(),
    }
    cmd_xy, v_xy = env._commands[:, :2], robot.data.root_lin_vel_b.torch[:, :2]
    moving = torch.linalg.norm(cmd_xy, dim=1) > 0.05
    cosang = (cmd_xy * v_xy).sum(1) / (torch.linalg.norm(cmd_xy, dim=1) * torch.linalg.norm(v_xy, dim=1)).clamp_min(1e-6)
    rec["angle(v, cmd) [deg], |cmd|>0.05"] = torch.rad2deg(torch.acos(cosang.clamp(-1, 1)))[moving].mean() if moving.any() else torch.tensor(float("nan"))
    rec["speed along cmd dir [m/s]"] = ((cmd_xy * v_xy).sum(1) / torch.linalg.norm(cmd_xy, dim=1).clamp_min(1e-6))[moving].mean() if moving.any() else torch.tensor(float("nan"))
    rec["|err xy| [m/s]"] = torch.linalg.norm(cmd_xy - v_xy, dim=1).mean()
    rec["feet airborne (of 6)"] = (torch.linalg.norm(sensor.data.net_forces_w.torch[:, env._feet_ids], dim=-1) < 1.0).float().sum(1).mean()
    rec["torque sat frac"] = (robot.data.applied_torque.torch.abs() > 0.95 * 1.6).float().mean()
    for g, ids in links.items():
        rec[f"{g}: frac of links touching (>1N)"] = (f[:, ids] > 1.0).float().mean()
        rec[f"{g}: total force [N]"] = f[:, ids].sum(-1).mean()
    dev = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    for g, ids in joints.items():
        rec[f"{g}: pos - default [rad]"] = dev[:, ids].mean()
    for k, v in rec.items():
        sums[k] = sums.get(k, 0.0) + v.item()
    n += 1
    min_root_z = min(min_root_z, robot.data.root_pos_w.torch[:, 2].min().item())

print(f"\n=== policy rollout: {args.num_envs} envs, averaged over steps {args.settle}-{args.steps} ===")
for k, v in sums.items():
    print(f"{k:38s} {v / n:9.4f}")
print(f"{'lowest root z seen [m]':38s} {min_root_z:9.4f}")
print(f"{'(standing pose root z, measured earlier)':38s} {'0.096':>9s}")
env.close()
app.close()
