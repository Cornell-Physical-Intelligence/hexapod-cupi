# hexapod/ — Isaac Lab RL environment for the club hexapod

A port of Isaac Lab's ANYmal-C direct velocity-tracking environment to the mkII hexapod, plus the method that got a
plain MLP policy walking: behavior-clone the scripted tripod, then fine-tune with PPO. Shaurya Sen, October 2026.

| file | what |
|---|---|
| `hexapod_robot_cfg.py` | `ArticulationCfg`: the repo's USD, standing pose, PD gains (kp 12, kd per joint), 1.6 N·m cap |
| `hexapod_env_cfg.py`, `hexapod_env.py` | the environment (observations, rewards, resets), 66-dim obs + optional gait clock, 18 actions |
| `agents/rsl_rl_ppo_cfg.py` | PPO settings (rsl_rl) |
| `tools/bc_clone.py` | clone the scripted tripod into the PPO actor (BC + DAgger) and splice it into a checkpoint |
| `tools/diag_policy_rollout.py` | evaluate a policy with exploration off from a standing start: speed along the command, heading error, falls |
| `tools/make_simple_collision.py` | one primitive per link instead of 153 SDF meshes: 2.4-4.5x faster training, same walking (`COLLISION_EXPERIMENT.md`) |
| `METHOD.md` | the method, every change vs. the ANYmal port, reproduce commands, measured results |
| `COLLISION_EXPERIMENT.md` | the simplified collision asset: how it is built, fidelity checks, throughput sweep |
| `RL_PROTOCOL.md` | the five questions to ask before/while training locomotion with RL, and why pure PPO stood still here |

Result (measured, 32 robots, noise-free policy, standing start, commands 0.10-0.15 m/s forward/back):
0.123 m/s along the command, 4.6 deg heading error, 0 falls, tripod rhythm. Checkpoints and videos are on the Spark
under `~shaurya/src/isaaclab-learning/logs/`; ask for a copy.

## Running

Needs Isaac Lab 3.0 (the `isaac-lab-base` container on the Spark). This folder must be importable as the package
`hexapod`, i.e. the repo root on `PYTHONPATH`; it registers the task `Hexapod-Flat-Direct-v0` through Isaac Lab's
`--external_callback` hook, so no files in the Isaac Lab install change.

```bash
# from the Isaac Lab root inside the container, with the repo mounted at /workspace/hexapod-cupi
export PYTHONPATH=/workspace/hexapod-cupi:$PYTHONPATH
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Hexapod-Flat-Direct-v0 \
    --external_callback hexapod.register --headless --num_envs 512 --max_iterations 300
```
Pure PPO from this command stands still (see `RL_PROTOCOL.md`). For the walking recipe follow `METHOD.md` section 3.
