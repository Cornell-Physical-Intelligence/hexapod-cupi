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
| `randomization.py` | Step 7 Table II declaration and per-replica draws; the surrogate applies them. | [TRAINING](../../docs/TRAINING.md) |

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

## Step 7: domain randomization declaration

[`randomization.py`](randomization.py) declares `table2_v1` for
[issue #67](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/67):
one field per Table II row with its status, this project's range, the draw
granularity and the reason. `declaration("table2_v1")` returns the canonical
JSON and its SHA-256, `730cad982ddac064cdd36090fd3d770a1b38a6f81aa03bba0ce792bba84b6a5c`.
Run identities, packs and admission bind that hash; `none` declares the nominal
properties and draws nothing. The first milestone-1 review freezes these ranges
with the push schedule. No row is excluded.

| Table II row | Paper | Project range | Draw | Status |
| --- | --- | --- | --- | --- |
| Joint stiffness | 0.8–1.2 × 100 | 0.8–1.2 × the explicit servo Kp 12 | one factor per replica | kept |
| Joint damping | 0.8–1.2 × 2 | 0.8–1.2 × the 18 named KD values | one factor per replica | kept |
| Joint position | 0.6–1.4 × nominal | additive reset-pose offset: coxa ±0.12, femur ±0.12, tibia ±0.16 rad | per joint | adapted |
| Link mass | 0.9–1.1 × nominal | 0.9–1.1 × each ledger mass, inertia scaled with it | per link | kept |
| Payload mass | 0–5 kg | 0–1.464 kg, the paper's share of a 7.466 kg robot | per replica | adapted |
| Payload position | ±0.15 m | X and Y ±0.15 m; Z fixed 0.04 m above the base origin | per replica | kept |
| Foot friction | 0.1–2.5 | 0.1–2.5 effective coefficient on the six toe caps; shafts and hulls keep 1.0 | per replica | kept |
| Motor strength | 0.8–1.2 | 0.8–1.2 × the speed-dependent ceiling, capped at 1.6 N·m | per joint | kept |

- **Joint position.** The paper scales the default pose at reset. The six coxa
  yaw joints have a 0.0 rad nominal angle, so a factor never moves them. Each
  joint adds an offset instead: 0.4 × |nominal| keeps the paper's extent on the
  femur (0.3 rad) and tibia (0.4 rad), and the coxa takes the femur half-width.
  Every offset stays inside the named travel limits, so nothing is clipped. The
  reset root height places the lowest drawn toe at the nominal 5 mm clearance.
  The actor's pose reference, the held target and the executed-action feature
  keep the nominal pose.
- **Payload.** A point mass on the base link changes its mass, centre of mass
  and parallel-axis inertia; no body is added, so the 19-body topology stays.
  The 0.04 m height is a declared deck assumption above the base centre of mass.
- **Motor strength under the cap.** The ceiling becomes
  `min(factor × curve(speed), 1.6 N·m)`. The nominal curve exceeds 1.6 N·m
  below 450 rpm, so a factor of 1.2 changes nothing below about 456 rpm, and a
  factor of 0.8 lowers the cap to 1.28 N·m everywhere. The row is asymmetric.
  The saturation counter and the over-rating reward terms keep their nominal
  1.6 N·m reference. `randomization.motor_force` applies a draw's gain and
  strength to the unchanged `robots/rs05.py` law.
- **Foot friction.** MuJoCo takes the larger of the two geom coefficients, so
  the surrogate sets a randomized replica's floor coefficient to zero and its
  toe caps carry the draw; tibia shafts and hulls keep 1.0. PhysX multiplies
  the toe material by the floor material 1.0. Each gives the drawn value at the
  toes and 1.0 elsewhere.
- **Gains.** Both factors act through the explicit motor law. The implicit
  PhysX drive and the MuJoCo joint damping stay at zero.
- **Pushes, adapted.** Section V-B applies velocity perturbations at
  one-second intervals for robustness evaluation and gives no training
  schedule. Training applies a force at the base centre of mass, so it adds no
  torque: the first push 1–10 s after a reset, then every 5–10 s, each 0.2 s
  (10 controls) at 0–15 N with a uniform heading in the body XY plane and no
  vertical component. 15 N over 0.2 s changes the velocity of this robot by
  0.40 m/s, half the paper's 0.803 m/s tolerance. The six privileged values are
  the body-frame XYZ force in N, then the XYZ torque in N·m, each the mean over
  the eight physics steps of the control; they read zero outside a push and
  after a reset.
- **Draws.** Uniform. Each replica owns one generator seeded from the run
  seed, the replica index and the declaration hash. A reset redraws that
  replica alone and advances no task or learner generator. The 0.040 rad /
  20 ms limiter, the 0.35 rad action scale and the 1.6 N·m cap stay fixed.

The surrogate applies these draws under `--randomization table2_v1`
([surrogate guide](surrogate/README.md)). The native implementation and the
`none` against `table2_v1` comparison follow under #67. This declaration admits
nothing.

`grid_declaration("table2_v1_grid")` declares the fixed-condition screen of
issue #67, SHA-256
`7d631f12e8abefd8141f0dc515eaa446b90b5563315fe41ea608db0ab2822882`. It crosses
foot friction 0.1, 1.0 and 2.5, payload 0, 0.732 and 1.464 kg, and one motor
strength factor 0.8, 1.0 or 1.2 on all joints: 27 cells. The payload sits
centred at the 0.04 m deck height, pushes stay off and every other property
keeps its nominal value. A cell holds across resets. The nominal cell steps as
the nominal robot bit for bit.

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
in `evaluation/`, and it names the floor. `EnvConfig.floor` selects a stock
Isaac Lab floor: the recorded two-triangle mesh built by the terrain utility
(the default), the grid ground plane or a cuboid. The recorded physics depends
on the simulator's default collider offsets on that mesh, so the default keeps
both offsets None and the recipe readback refuses a floor that carries authored
offsets or a different material. The 16 October 2026 experiments
([evidence](../../site/assets/floor_contact_20261016_001/floor_experiments.json))
show why: with the robot's 153 SDF collision shapes, the stock ground plane
fails the backward probe on one joint-speed substep whatever contact offset it
carries, a cuboid fails another probe, and the mesh with authored offsets fails
two; only the mesh with default offsets reproduces the recorded one-robot
standing report byte for byte.

On that floor source `8cce83cc` passed standing admission at one and 128
robots, and the unchanged `c48_long` checkpoint passed its 13 probes through
the parity harness with all 8,088 measurements equal to the three recorded
baselines, and again through the certified loader
([verification](../../site/assets/floor_contact_20261016_001/verification.json)).
The review fixes to the replay identity and the policy smoothing changed the
runtime sources but not the physics closure, so source `717548d5` repeated the
harness replay with the same 8,088 equal measurements. Two receipts certify
that replay and bind the same runtime bytes:
[pr84_review_20261017_001](../../site/assets/pr84_review_20261017_001/compatibility/receipt.json)
adds raw trace rescoring and recomputed contact and torque reports, and
[floor_contact_20261016_002](../../site/assets/floor_contact_20261016_002/compatibility/receipt.json)
is the one the production loader ran natively, passing the 13 probes again
([verification](../../site/assets/floor_contact_20261016_002/verification.json)).
Both bind the runtime bytes of source `717548d5`. `scripts/play.py
--checkpoint-compatibility` accepts a receipt only when the running runtime
files match the ones it binds, so the 23 October 2026 Ruff reformat of the
runtime (PR 86) needs its own receipt; that certification is pending.
`tests/native_parity_pack.py` writes a candidate replay pack with the harness
and its pinned ranges, and `spark/pack.py` records the root review of any pack
before launch. Stage 2 adds the 96-case suite and visual acceptance.
