# CPU surrogate

You run the omni task on MuJoCo 3.14.0 for CPU diagnostics.
The surrogate shares control arithmetic, task rewards and stock RSL-RL PPO with
the native experiment. You use native Isaac runs for acceptance.

| Module | Purpose |
| --- | --- |
| `env.py` | Implement `OmniEnv` on the approved robot: apply the limiter and motor curve, and record contacts and telemetry. |
| `train.py` | Train through stock RSL-RL, save source-bound checkpoints and resume matching runs. |
| `evaluate.py` | Run learning probes, render video and compare static results across contact settings. |
| `replay.py` | Replay action traces or noise and score stored telemetry through the current task. |

```sh
uv run python -m locomotion.surrogate.train --output /absolute/fresh/run \
  --updates 1 --num-envs 4 --threads 1
uv run python -m locomotion.surrogate.evaluate \
  --checkpoint /absolute/run/checkpoint_update000001.pt \
  --output /absolute/fresh/evaluation --video none
uv run python -m locomotion.surrogate.replay --help
uv run python -m unittest discover -s src/locomotion/surrogate/tests
```

Keep output directories outside the source checkout. The trainer defaults to
task `locomotion.omniwalk.task:TrainingTask` and wrapper
`locomotion.rl.wrapper:OmniVecEnv` with the 60-control gait clock. Its learner
flags start from the plain `ppo_config` defaults. To match the `c48_long`
checkpoint, pass its recipe values as flags and `--ppo-config-overrides`.
`--resume` checks checkpoint bytes, the paired record and source/configuration
identity. The update count and deviation schedule continue; simulator state and
random generators restart from the recorded seed.

The evaluator reconstructs the recorded actor inputs with no observation noise.
You can test joint-velocity input gains or force a two-control action mean. Static
probe consensus compares the default contact model with two sensitivity models;
agreement grants no native acceptance.

`replay score --telemetry ... --rewards v4_yaw_gate --output ...` runs
`locomotion.omniwalk.task:TrainingTask` against recorded telemetry, including its
reward memory. You can supply a `module:Class` task or a `module:function` reward
to compare. The replay command accepts recorded policy actions or motor
targets.

## Limits and retained evidence

MuJoCo uses convex collision hulls and a flat floor with no self-collision.
The contact solver includes measured position-integration and touchdown
corrections. Joint-rate costs, joint-limit termination, quiet/stop behavior and
per-joint torque can differ from Isaac. You must confirm those outcomes on native
captures. Native source packs exclude this package and its MuJoCo dependency.

`provenance.json` and `records/` retain the historical calibration identities
and results. Those records used their recorded reward versions and sources.
Git revision `077b98b1` retains the calibration and machine self-check commands;
they depend on retired reward implementations and native extract formats.
Current tests retain control gates, exercise stock PPO updates/resume and
compare live versus replay reward values.
