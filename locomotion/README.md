# Locomotion

You can work with two experiments on Isaac Lab's stock `DirectRLEnv` and
RSL-RL 5.0.1. Keep their configurations and checkpoints separate.
[TRAINING](../docs/TRAINING.md) defines the intended motion and acceptance gates.

## Experiments

| Contract | c48_long | Shaurya BC/PPO |
| --- | --- | --- |
| Reward | Recorded yaw-gated walking reward | Original velocity tracking, air time and penalties |
| Actor inputs | 231 plus two clock values | 66, or 68 with the clock |
| Actor network | 256/256/128 | 128/128/128 for the flat task |
| Actuator | Explicit speed/torque curve and 0.040 rad control limiter | Original implicit PD, 1.6 N·m cap, no control limiter |
| Learning | PPO | Original scripted BC, DAgger and PPO fine-tuning |
| Entry | `locomotion.runs.train` | Stock Isaac Lab scripts with `locomotion.shaurya.register` |

`shaurya/` preserves the implementation from commit `969ea5bc`, including its
randomization, resets and rough-task variant. Its source tests permit namespace
and file-location changes; they compare numerical code against that revision.
We removed the hybrid BC path that adapted his teacher to the c48 policy.

## Code to read

| File or package | Responsibility |
| --- | --- |
| `backends/isaac/config.py` | CPU-safe control settings and the native scene configuration. |
| `backends/isaac/native.py` | Stock Direct environment hooks and Articulation capture hooks. |
| `backends/isaac/env.py` | Named state readback, observations and the current task interface. |
| `backends/isaac/motor.py`, `core/` | Explicit actuator and shared control arithmetic. |
| `tasks/task.py` | The c48_long reward, command sampler and episode handling. |
| `learning/config.py`, `learning/ppo.py` | c48 actor configuration and learner adapter. |
| `shaurya/` | Original environment/configuration, PPO settings and BC/diagnostic tools. |
| `evaluation/` | Rollout capture, video and unchanged numerical gates. |
| `runs/` | Input validation, source packs, checkpoint identity and guarded execution. |

The [developer map](https://cornell-physical-intelligence.github.io/hexapod-cupi/code/)
shows definitions and imports. Pages generates it from the build revision.
`priors/datasets/` and `surrogate/records/` retain historical data; they contain
no maintained optimizer or simulator. Git retains the retired experiment code.

## Implementation rules

Follow [Isaac Lab's Direct tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
and Shaurya's task structure. The runtime pins Isaac Lab 3.0.0 at `ffff603e`
with Isaac Sim 6. Keep one implementation of each behavior. Use stock scene
cloning and physics stepping; add custom code for a named contract difference.

For c48, preserve 400 Hz physics, eight substeps per action, the 0.35 rad action scale
and the 0.040 rad target-change limit per control. Apply the approved explicit
motor curve at each physics step, with its provisional 1.6 N·m cap. Capture
submitted torque before physics and ground contact after physics.

Keep actor observations at 231 values and critic observations at 234, before
optional gait-clock inputs. Preserve frame order and selected-row reset
isolation. The task owns rewards and resets so the learner can read terminal
observations before reset. Preserve numerical gates. Git and the frozen test
fixture retain historical reward implementations; the runtime imports none of them.

Keep each experiment’s actor and observation layout intact. Keep evaluators outside the training
loop and source identity outside the environment. Geometry, collision filtering
and randomization changes require separate experiments and fresh admission.

## c48 commands

From the repository root:

```sh
uv sync --locked
uv run python -m unittest discover -s locomotion/tests
uv run python -m locomotion.runs.prepare --help
uv run python -m locomotion.runs.train --help
uv run python -m locomotion.runs.admission --help
uv run python -m locomotion.runs.dataset
```

Use `prepare` with `--mode diagnostic|train|evaluate|probe`, explicit inputs and
a fresh output directory. Training supports 128 or 512 robots with matching
standing admission. Diagnostics support 1/32/128/512/1024 robots. Evaluation
uses one robot; `--eval-scope probes` runs 13 learning probes and `full` runs
the 96-case Stage 2 suite. Use `--help` for learner options and deadline limits.

Training uses one reward configuration: the recorded `c48_long` yaw-gated
reward. Training defaults to its 60-control gait clock. Set learner options
from the chosen experiment; the reward configuration does not select them.

Read [SPARK](../SPARK.md) and [OPERATIONS](../docs/OPERATIONS.md) before dispatch.
The launcher retains both GPU locks and the existing cleanup rules. New packs
hold `root_review_complete=False` until you authorize full review and the
review clears the binding. This branch has no native migration qualification.

## Shaurya commands

Use the pinned Isaac Lab checkout with this repository on `PYTHONPATH`. The
callback is `locomotion.shaurya.register`; task IDs remain
`Hexapod-Flat-Direct-v0` and `Hexapod-Rough-Direct-v0`. The robot config resolves
the repository USD, or uses your `HEXAPOD_USD` override.

For the original walking recipe, follow [Shaurya’s method at 969ea5bc](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/969ea5bc942df3122b2662650f29acd11644a41c/hexapod/METHOD.md#3-how-to-reproduce-inside-the-container).
Use the callback above in its stock training commands. Replace its BC script
invocation with `./isaaclab.sh -p -m locomotion.shaurya.tools.bc_clone` and pass
`--splice_into` with the initial checkpoint path. Pass `--out` with a fresh output
directory. Resume stock PPO with that recipe’s environment and learner overrides.
The original diagnostic entry is `locomotion.shaurya.tools.diag_policy_rollout`.

These are native entry points. The c48 pack/launcher does not dispatch Shaurya’s
stock scripts. Follow OPERATIONS for shared GPU ownership and locks before any
native run; this PR authorizes no dispatch. Keep his original result records
separate from c48’s acceptance reports. The restored BC speed diagnostic retains
its reset-time command direction after later resets; do not use it as a c48 gate.

## Regression and evidence

CPU tests compare control arithmetic, observations, rewards, RNG sequences and
optimizer updates against the frozen fixtures. c48 native qualification requires
fresh standing admission and unchanged `c48_long` weights passing all 13 probes.
Use `tests/native_parity.py` and `tests/native_ranges.py`: freeze ranges from
three baseline replays before the new runtime replay. Keep the gates and
contact/torque comparisons unchanged. Retain failures and block merge on a
failed comparison. CPU tests cannot establish simulator parity.

Historical checkpoints use their frozen packs. New packs hash nested runtime
sources; production loading requires matching model, configuration and source
identities. The existing compatibility receipt binds its original runtime and
cannot authorize this migration. Create new proof through the test-only replay
harness before issuing a receipt. Native optimizer resume remains out of scope.

c48 evaluation loads the actor without constructing a training task, critic or
optimizer. Checkpoint identity checks still apply.

Each evaluation retains its control trace, 400 Hz capture chunks, report and
force metrics. Selected video cases also save `rollout.mp4`. Keep checkpoint
and evidence bytes intact. Use [the archive guide](../docs/PIPELINE_LINEAGES.md)
to retrieve a historical implementation outside the source checkout.
