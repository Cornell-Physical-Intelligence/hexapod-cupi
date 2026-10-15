# Locomotion

You train and evaluate the approved hexapod through Isaac Lab's stock
`DirectRLEnv` and RSL-RL 5.0.1 PPO. BC trains the same actor before PPO.
[TRAINING](../docs/TRAINING.md) defines the intended motion and acceptance gates.

## Code to read

| File or package | Responsibility |
| --- | --- |
| `backends/isaac/config.py` | CPU-safe control settings and the native scene configuration. |
| `backends/isaac/native.py` | Stock Direct environment hooks and Articulation capture hooks. |
| `backends/isaac/env.py` | Named state readback, observations and the interface used by frozen tasks. |
| `backends/isaac/motor.py`, `core/` | Explicit actuator and shared control arithmetic. |
| `tasks/` | Command sampling and versioned reward formulas. |
| `learning/config.py`, `learning/ppo.py`, `learning/bc.py` | Shared actor configuration, learner adapter and scripted BC. |
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

Preserve 400 Hz physics, eight substeps per action, the 0.35 rad action scale
and the 0.040 rad target-change limit per control. Apply the approved explicit
motor curve at each physics step, with its provisional 1.6 N·m cap. Capture
submitted torque before physics and ground contact after physics.

Keep actor observations at 231 values and critic observations at 234, before
optional gait-clock inputs. Preserve frame order and selected-row reset
isolation. Frozen tasks own episode rewards and resets so the learner can read
terminal observations before reset. Keep numerical gates and frozen reward
bytes unchanged; `tasks/env.py` supports their retained imports.

Use one actor configuration for PPO and BC. Keep evaluators outside the training
loop and source identity outside the environment. Geometry, collision filtering
and randomization changes require separate experiments and fresh admission.

## Commands

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

Add `--bc-warmstart --gait-clock 40` to PPO training with reward version 1, 2 or
3 to clone the scripted forward/backward teacher before PPO. BC restores the
original command bank, resets the optimizer and saves checkpoint update zero.
The source and result identify this experiment. Sideways motion and turning
still require evaluation; BC supplies no acceptance evidence.

Read [SPARK](../SPARK.md) and [OPERATIONS](../docs/OPERATIONS.md) before dispatch.
The launcher retains both GPU locks and the existing cleanup rules. New packs
hold `root_review_complete=False` until you authorize full review and the
review clears the binding. This branch has no native migration qualification.

## Regression and evidence

CPU tests compare control arithmetic, observations, rewards, RNG sequences and
optimizer updates against the frozen fixtures. Native qualification requires
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

Each evaluation retains its control trace, 400 Hz capture chunks, report and
force metrics. Selected video cases also save `rollout.mp4`. Keep checkpoint
and evidence bytes intact. Use [the archive guide](../docs/PIPELINE_LINEAGES.md)
to retrieve a historical implementation outside the source checkout.
