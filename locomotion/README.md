# Locomotion

You can run two Isaac Lab Direct experiments. They share the robot asset, but
keep different observations, rewards and actuators. Compare their results under
those settings. [TRAINING](../docs/TRAINING.md) defines the goal and acceptance.

| Package | Start here |
| --- | --- |
| `omni/` | The omnidirectional PPO experiment that produced `c48_long`. Read `config.py`, `env.py` and `task.py`. |
| `tripod/` | Shaurya's original Direct task, scripted BC and PPO fine-tuning. Read its three `hexapod_*.py` files. |
| `priors/` | Generate and review demonstration trajectories; train the AMP discriminator and optional paper networks. |
| `surrogate/` | Run CPU MuJoCo experiments, resume training and replay traces without Spark. |
| `tools/` | Prepare and launch native jobs; capture and score motion. |
| `tests/` | Frozen behavior comparisons and native replay tools. |

The [code map](https://cornell-physical-intelligence.github.io/hexapod-cupi/code/)
shows definitions and imports. Pages builds it from the source revision.

## Experiment contracts

| Setting | Gait clock | Behavior cloning |
| --- | --- | --- |
| Actor input | 231 values plus two clock values | 66 values, or 68 with its clock |
| Flat actor | 256/256/128 | 128/128/128 |
| Motor | Explicit speed/torque curve, 1.6 N·m cap, 0.040 rad/control slew | Implicit PD, 1.6 N·m cap |
| Reward | Recorded yaw-gated omnidirectional reward | Original velocity tracking, air time and penalties |
| Learning | PPO, or AMP with demonstrations | Scripted BC, DAgger and PPO |

Both use 400 Hz physics and eight substeps per action. Keep their checkpoint
formats and configuration separate. `c48_long` names a historical training run.

## Stock code and custom code

Use Isaac Lab's `DirectRLEnv` for stepping and episode counters, `InteractiveScene`
for clones and collision filtering, and stock spawners for ground and assets.
Use RSL-RL 5.0.1 for PPO and its runner. The pinned Isaac Lab revision is
`ffff603e` with Isaac Sim 6. Follow the
[Direct tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html).

Keep custom code for a specific contract: the motor curve and slew limiter,
recorded observation history and reward, terminal observations before reset,
and per-substep torque/contact capture. AMP adds a discriminator and optional
velocity estimator to stock PPO. Its demonstration optimizer uses this robot's
kinematics and dynamics. The CPU surrogate uses MuJoCo and cannot qualify Isaac
results. Keep docstrings short and place procedures in this guide.

The scorer needs individual contact patches to distinguish toe contact from
shaft contact. The stock sensor exposes averaged contact positions, so capture
retains one raw PhysX contact view for those checks and per-body force reads.
The video wrapper checks that rendering leaves physics state unchanged; keep
that check when changing the recorder.

The gait-clock candidate now uses stock collision filtering and an infinite
ground plane with an 80 × 80 m display mesh. The old simulator used a finite
triangle floor and aborted when robots approached. These changes need fresh
native admission; existing results retain their original simulator identity.

## Commands

Run CPU checks with `uv run python -m unittest discover -s locomotion/tests`.
For native job arguments, use:

```sh
uv run python -m locomotion.spark.prepare --help
uv run python -m locomotion.omni.train --help
uv run python -m locomotion.evaluation.admission --help
```

PPO uses the gait-clock reward. AMP adds `--learner amp`; `--networks paper`
selects the paper actor/critic and disables the gait-clock inputs. Read
[priors](priors/README.md) for generation and demonstration commands, and
[surrogate](surrogate/README.md) for CPU training and resume.

For behavior cloning, put this repository on `PYTHONPATH` in the pinned Isaac
Lab checkout. Use stock train/play scripts with callback
`locomotion.tripod.register` and task IDs `Hexapod-Flat-Direct-v0` or
`Hexapod-Rough-Direct-v0`. Use
`./isaaclab.sh -p -m locomotion.tripod.tools.bc_clone` for BC.
The [original recipe](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/969ea5bc942df3122b2662650f29acd11644a41c/hexapod/METHOD.md#3-how-to-reproduce-inside-the-container)
defines its training overrides. Its speed diagnostic keeps the reset-time
command direction; use the common evaluator for motion acceptance.

Read [SPARK](../SPARK.md) and [OPERATIONS](../docs/OPERATIONS.md) before dispatch.
Keep GPU locks and fresh output directories. Native packs retain the review
hold. This branch has no native migration qualification.

## Verification

CPU tests compare arithmetic, observations, rewards, RNG and optimizer updates.
Keep frozen fixtures and numerical gates intact. Native qualification needs
standing admission and unchanged `c48_long` weights passing its 13 probes,
with torque/contact measurements inside ranges from three baseline replays.
Use `tests/native_parity.py` and `tests/native_ranges.py`. The test harness
permits the named stock-ground and collision-filter changes; it rejects other
physics changes and leaves production loading strict. Full Stage 2 remains
the 96-case suite and visual acceptance.

Preserve checkpoint and evidence bytes. Default loading checks the recorded
configuration and source hashes. The historical compatibility receipt cannot
certify this changed simulator. Full PR review and merge remain on hold.
