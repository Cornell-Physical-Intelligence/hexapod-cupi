# Locomotion

You can run two Isaac Lab Direct experiments on one robot; compare each under
its own settings. [TRAINING](../docs/TRAINING.md) defines the goal and
acceptance, and the
[code map](https://cornell-physical-intelligence.github.io/hexapod-cupi/code/)
shows definitions and imports.

| Package | Purpose | Start here |
| --- | --- | --- |
| `omniwalk/` | Omnidirectional walking task and PPO. | `env.py`, `task.py` |
| `tripod/` | Original tripod task with BC and PPO fine-tuning. | `hexapod_env.py` |
| `priors/` | Motion demonstrations and AMP learning. | [README](priors/README.md) |
| `surrogate/` | CPU MuJoCo training and trace replay. | [README](surrogate/README.md) |
| `evaluation/` | Native capture, checkpoint identity and motion scoring. | `runner.py` |
| `spark/` | Prepare, launch and bind native allocations. | `prepare.py` |
| `tests/` | CPU, parity and native replay tests. | [Verification](#verification) |

## Omni files

- `robot.py`: joint names, body names, asset hashes and damping.
- `motor.py`: PD torque curve and slew limiter.
- `actuator.py`: Isaac actuator for that curve.
- `frames.py`: quaternion and navigation frames.
- `env_cfg.py`: `EnvConfig` and `DirectRLEnvCfg` factory.
- `isaac.py`: `IsaacEnv`, its scene, views, checks, capture and resets.
- `env.py`: `OmniEnv`, the task-facing surface.
- `features.py`: observations and motion features.
- `commands.py`: command bank and gait-clock period.
- `reward.py`: recorded yaw-gated reward.
- `task.py`: `TrainingTask` for command holds, reward memory, resets and metrics.
- `wrapper.py`: `OmniVecEnv` with input scales, gait clock, smoothing and noise.
- `agents/rsl_rl_ppo_cfg.py`: Torch-free learner configuration and the `c48_long` recipe.
- `ppo.py`: PPO subclasses, distributions, deviation schedule and diagnostics.
- `train.py`: entry point, one function per mode.

## Experiment contracts

| Setting | `omniwalk` | `tripod` |
| --- | --- | --- |
| Actor input | 231 values plus two clock values | 66 values, or 68 with its clock |
| Flat actor | 256/256/128 | 128/128/128 |
| Motor | Explicit curve and 0.040 rad/control slew; 1.6 N·m cap | Implicit PD, 1.6 N·m cap |
| Reward | Recorded yaw-gated reward | Original velocity tracking and penalties |
| Learning | PPO, or AMP with demonstrations | Scripted BC with DAgger, then PPO |

Both use 400 Hz physics with eight substeps; keep their checkpoints separate.

## Stock code and custom code

`IsaacEnv` in `isaac.py` is the omni task's one `DirectRLEnv` subclass. It uses
`InteractiveScene` for clones and collision filtering. `OmniEnv` wraps it for
the task, wrapper and evaluation; the MuJoCo surrogate implements the same
surface. Pin RSL-RL 5.0.1 and Isaac Lab `ffff603e`, and follow the
[Direct tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html).

Keep custom code for a named contract, such as the motor curve or per-substep
capture. `IsaacEnv` keeps one raw PhysX contact view because the stock sensor
averages the patches that tell toe from shaft contact. State reads use raw PhysX
views until native parity on Spark admits `robot.data`. Keep the check in
`evaluation/camera.py` that rendering leaves physics state unchanged.

## Import direction

Reviewers enforce this rule: "`evaluation` → `omniwalk`; `spark` → `omniwalk`; `priors`
→ `omniwalk`, and `priors` → `evaluation`/`spark` from its command modules
(`dataset.py`, `prepare.py`, `replay_native.py`); `surrogate` → `omniwalk`,
`evaluation`, `spark`, `priors`; `omniwalk` imports the other packages only from
`train.py`. No package imports `tripod`." A grep for `from locomotion.spark` or
`from locomotion.evaluation` inside `omniwalk/` other than `train.py` is a review
failure.

## Commands

```sh
uv run python -m unittest discover -s locomotion/tests
uv run python -m locomotion.spark.prepare --help
uv run python -m locomotion.scripts.train --help
uv run python -m locomotion.evaluation.admission --help
```

PPO training and evaluation start from `--recipe c48_long`, the admitted
checkpoint's options in `omniwalk/agents/rsl_rl_ppo_cfg.py`; `test_recipe.py`
checks them against its receipt. Option flags replace recipe values, and
`--recipe none` keeps the option defaults. `--learner amp` adds the motion prior;
`--networks paper` selects paper networks without a gait clock.

For tripod, put this repository on `PYTHONPATH` in the pinned Isaac Lab
checkout. Use stock train/play scripts with callback
`locomotion.tripod.register` and task `Hexapod-Flat-Direct-v0` or
`Hexapod-Rough-Direct-v0`. Run BC with
`./isaaclab.sh -p -m locomotion.tripod.tools.bc_clone` and the
[original recipe](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/969ea5bc942df3122b2662650f29acd11644a41c/hexapod/METHOD.md#3-how-to-reproduce-inside-the-container)
overrides. Its speed diagnostic keeps the reset-time direction; use the common
evaluator for acceptance.

## Verification

Keep the numeric oracles in `tests/fixtures/baseline/` and the gates intact.
`test_frozen_baseline_parity.py` and `test_recorded_trace.py` check frozen
arithmetic and a recorded native trace. `test_direct_environment.py` guards the
stock lifecycle; `test_dependency_identity.py` checks the physics closure.
`test_motor_force.py`, `test_wrapper.py`, `test_guarded_paths.py`,
`test_pack_binding.py`, `test_tripod_environment.py` and `test_tripod_bc.py`
cover motors, transitions, packs and tripod sources.

## Native admission

The physics identity hashes the import closure of `omniwalk/adapter.py`,
`omniwalk/env_cfg.py`, `omniwalk/env.py`, `evaluation/capture.py` and
`evaluation/scoring.py`. The refactor changed that closure; no native run has
used it. Any new result needs fresh standing admission at one robot and batch
size. The unchanged `c48_long` checkpoint must then pass its 13 probes with
torque and contact inside three-baseline ranges (`tests/native_parity.py`,
`tests/native_ranges.py`). Stage 2 also requires the 96-case suite and visual
acceptance. Preserve checkpoint and evidence bytes.
