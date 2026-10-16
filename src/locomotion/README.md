# Locomotion

You run two Isaac Lab Direct experiments on one robot; compare each under
its own settings. [TRAINING](../../docs/TRAINING.md) defines the goal and
acceptance, and the
[code map](https://cornell-physical-intelligence.github.io/hexapod-cupi/code/)
shows definitions and imports.

| Package | Purpose | Start here |
| --- | --- | --- |
| `robots/` | Robot identity, RS05 motor model and Isaac actuator. | `hexapod.py` |
| `omniwalk/` | Omnidirectional walking task, Direct environment and learner recipe. | `env.py`, `task.py` |
| `rl/` | RSL-RL wrapper, PPO subclasses and action distributions. | `wrapper.py` |
| `scripts/` | Container entry points: train, play and diagnose. | `train.py` |
| `evaluation/` | Native capture, scene checks, admission, scoring and training diagnostics. | `runner.py` |
| `spark/` | Prepare, launch and bind native allocations. | `prepare.py` |
| `amp/` | Motion demonstrations, motion features and AMP learning. | [README](amp/README.md) |
| `surrogate/` | CPU MuJoCo training and trace replay. | [README](surrogate/README.md) |
| `hexapod/` | Original tripod task with BC and PPO fine-tuning. | `hexapod_env.py` |
| `tests/` | CPU, parity and native replay tests. | [Verification](#verification) |

## Omniwalk files

- `env.py`: `IsaacEnv`: scene, views, layout guards, substep capture, telemetry and stock reset hooks.
- `adapter.py`: `OmniEnv`, the task-facing surface.
- `env_cfg.py`: `EnvConfig` and the `DirectRLEnvCfg` factory.
- `task.py`: `TrainingTask`: command holds, reward memory, selected resets and an observer hook.
- `mdp/actions.py`: slew limiter and executed-action feature.
- `mdp/frames.py`: quaternion and navigation frames.
- `mdp/observations.py`: proprioception frame and 231-value actor input.
- `mdp/commands.py`: command bank and gait-clock period.
- `mdp/rewards.py`: recorded yaw-gated reward.
- `agents/rsl_rl_ppo_cfg.py`: Torch-free `ppo_config` and the default `RECIPES['c48_long']`.

The layout mirrors Isaac Lab: `robots/` is `isaaclab_assets`, `omniwalk/` a
`tasks/direct/<task>` package with `agents/`, `rl/` is `isaaclab_rl` and `scripts/`
is `scripts/reinforcement_learning/rsl_rl/`, with `cli.py` as `cli_args.py`.

## Experiment contracts

| Setting | `omniwalk` | `hexapod` |
| --- | --- | --- |
| Actor input | 231 values plus two clock values | 66 values, or 68 with its clock |
| Flat actor | 256/256/128 | 128/128/128 |
| Motor | Explicit curve and 0.040 rad/control slew; 1.6 N·m cap | Implicit PD, 1.6 N·m cap |
| Reward | Recorded yaw-gated reward | Original velocity tracking and penalties |
| Learning | PPO, or AMP with demonstrations | Scripted BC with DAgger, then PPO |

Both use 400 Hz physics with eight substeps; keep their checkpoints separate.

## Stock code and custom code

On `IsaacEnv`, the one `DirectRLEnv` subclass, `step`, `reset` and `cfg` keep
their stock meaning. Callers need a selected-row reset, a dictionary step and
`EnvConfig` as `cfg`. `OmniEnv` supplies that surface, and so does the MuJoCo
surrogate.

Pin RSL-RL 5.0.1 and Isaac Lab `ffff603e`, and follow the
[Direct tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html).
Keep custom code for a named contract. The stock contact sensor averages the
patches that tell toe from shaft contact, so `IsaacEnv` keeps one raw PhysX
contact view. State reads use raw PhysX views until native parity admits
`robot.data`.

## Import direction

Reviewers enforce the AST import audit: "`omniwalk -> [robots]`;
`evaluation -> [omniwalk, robots]`; `spark -> [omniwalk, robots]`; `rl -> []`;
`amp -> [evaluation, omniwalk, rl, robots, spark]`;
`scripts -> [amp, evaluation, omniwalk, rl, robots, spark]`;
`surrogate -> [amp, evaluation, omniwalk, rl, robots, spark]`;
no cycles." `robots` imports no package, and no package imports `hexapod`. A new
edge is a review failure.

## Commands

```sh
uv run python -m locomotion.spark.prepare --help
uv run python -m locomotion.scripts.train --help
uv run python -m locomotion.scripts.play --help
uv run python -m locomotion.scripts.diagnose --help
uv run python -m locomotion.evaluation.admission --help
uv run python -m locomotion.surrogate.train --help
```

`prepare --mode` picks the script a pack runs (`ENTRY_POINTS` in
`spark/validation.py`). The default command line reproduces the recorded
configuration of the `c48_long` checkpoint. Option flags replace recipe values,
and `--recipe none` keeps the plain defaults. `test_recipe.py` proves the
recipe against the checkpoint's compatibility receipt.

For the hexapod task, put this repository's `src/` on `PYTHONPATH` in the
pinned Isaac Lab checkout. Use stock train/play scripts with callback
`locomotion.hexapod.register` and task `Hexapod-Flat-Direct-v0` or
`Hexapod-Rough-Direct-v0`. Run BC with
`./isaaclab.sh -p -m locomotion.hexapod.tools.bc_clone` and the
[original recipe](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/969ea5bc942df3122b2662650f29acd11644a41c/hexapod/METHOD.md#3-how-to-reproduce-inside-the-container)
overrides. Use the common evaluator for acceptance.

## Verification

Keep the numeric oracles in `tests/fixtures/baseline/` and the gates intact.
`test_frozen_baseline_parity.py` and `test_recorded_trace.py` check frozen
arithmetic and a recorded native trace. `test_direct_environment.py` guards the
stock lifecycle; `test_dependency_identity.py` checks the physics closure and
the shared entry hash. `test_motor_force.py`, `test_wrapper.py`,
`test_pack_binding.py`, `test_launch.py` and `test_hexapod_*.py`
cover motors, transitions, packs and hexapod sources.

## Native admission

The physics identity hashes the import closure of `adapter.py`, `env_cfg.py`
and `env.py` in `omniwalk/`, with `admission.py`, `capture.py` and `scoring.py`
in `evaluation/`. The new layout changed those identities; no native run has
used them. Any result first needs fresh standing admission at one robot and
batch size. The unchanged `c48_long` checkpoint must then pass its 13 probes
through its compatibility receipt, with torque and contact inside
three-baseline ranges (`tests/native_parity.py`, `tests/native_ranges.py`).
Stage 2 adds the 96-case suite and visual acceptance.
