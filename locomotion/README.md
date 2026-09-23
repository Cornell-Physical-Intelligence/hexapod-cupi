# Locomotion kernel

You run canonical locomotion through this package. The package uses the approved
19-body, 18-joint robot and stock RSL-RL 5.0.1 PPO. You can inspect its control
loop without reading old experiment launchers or a second environment port.

| Read in this order | Responsibility |
| --- | --- |
| [`../robot/active_model.json`](../robot/active_model.json), [`env_config.py`](env_config.py) | Select the URDF, masses, joint order, motor coefficients and fixed timing. |
| [`env.py`](env.py) | Load the native robot, form observations, limit joint targets, apply motor torque and advance eight physics steps. The actor receives 231 values and returns 18 offsets. |
| [`task.py`](task.py) | Sample held velocity commands, compute the training reward, detect failed episodes and reset selected robots. This file owns the reward. |
| [`ppo.py`](ppo.py), [`train.py`](train.py) | Adapt the task to stock PPO, record updates and loads, save checkpoints and load them for evaluation. |
| [`evaluate.py`](evaluate.py), [`evaluation.py`](evaluation.py) | Record actual policy rollouts and apply the existing walking, stopping, contact and motor gates. Missing cases remain missing. |
| [`force_metrics.py`](force_metrics.py), [`camera.py`](camera.py) | Report contact-normal force and motor torque, and record an Isaac camera video from the policy rollout. |
| [`admission.py`](admission.py) | Recompute one-robot and batch standing captures and require matching model, source and geometry before training. |
| [`prepare.py`](prepare.py), [`launch.py`](launch.py), [`reservation.py`](reservation.py) | Freeze named package files and explicit inputs, hold the existing Spark locks, supervise one container and verify its cleanup. |

The controller retains 400 Hz physics, 50 Hz policy actions, a 0.35 rad action
scale, the 0.040 rad target-change limit per control and the provisional 1.6 N·m
motor cap. [`task.py`](task.py) owns reward coefficients and command sampling;
[`evaluation.py`](evaluation.py) owns the numerical gates. Read
[TRAINING](../docs/TRAINING.md) for the proposed paper-method extensions.

## Run and verify

From the repository root:

```sh
uv sync --locked
uv run python -m unittest discover -s locomotion/tests
uv run python -m locomotion.prepare --help
uv run python -m locomotion.admission --help
uv run python -m locomotion.train --help
```

Use `prepare` to create a fresh allocation. Diagnostic mode supports one, 32 or
128 robots. Training uses the existing 128-robot configuration. Evaluation uses
one robot, with `--eval-scope focus` for the forward/quiet/stop subset, `probes`
for all 13 learning probes, or `full` for the 96 required Stage 2 cases.

Start the frozen launcher on Spark from its `source` directory with
`python3 -B -m locomotion.launch`, the binding path and its SHA-256. Read
[OPERATIONS](../docs/OPERATIONS.md) before dispatch. Each native result records
its exact source and inputs. Use `admission` on complete one/batch results to
produce the input declaration for a subsequent pack. Existing admissions cannot
admit changed physics source. Checkpoint loads require matching model, physics,
configuration and implementation; use the original frozen entry for historical
checkpoints that carry a different identity.

Each evaluation saves `control_trace.npz`, 400 Hz capture chunks, `report.json`
and `force_metrics.json`. A selected video also saves `rollout.mp4`. The reports
bind the checkpoint and file hashes. Training writes `metrics.jsonl` and load
summaries; mean contact force includes swing zeros. Contact forces exclude
unrecorded tangential friction. A completed recording does not accept walking.

## Larger project boundary

You need this loop to establish walking and stopping before adding terrain.
The model import and geometry tools remain under `robot/` and `tools/assets/`.
Hardware execution needs a measured actuator profile and a canonical runtime
binding. Navigation can then consume velocity commands and stopping status;
survey recording can consume pose and measurement quality. The navigation example consumes shared commands. Historical runtime bindings
remain in Git until a canonical replacement receives its own parity evidence.

The optional trajectory optimizer and native replay live in `priors/`. Package
canonical inputs with `python -m locomotion.inputs pack --help`. Use the
[archive guide](../docs/PIPELINE_LINEAGES.md) for historical checkpoints and
source packs; preserve their original identities.

## Prescribed tripod controller

You can evaluate the Zhang et al. sinusoidal tripod controller with
[`tripod_evaluate.py`](tripod_evaluate.py). Read the source-cited
[reproduction contract](../docs/TRAINING.md#terrain-adaptive-tripod-reproduction-contract)
for the finite sweep and geometry adaptations. [`tripod.py`](tripod.py) owns
the equations and contact state; [`tripod_config.py`](tripod_config.py) owns
parameters. You use `prepare --mode tripod --tripod-adaptation stop_stride`
with matching standing admission. CPU checks and matching one/batch native
standing admission pass. The current variant passes the
[forward and both-turn screens](../site/assets/tripod_speed_lift_native_20260922_001/native_result.json)
and the [three stop cases](../site/assets/tripod_stop_stride_native_20260922_001/native_result.json);
each stop repeat replays its first trace. Full qualification and clearance
remain incomplete. Terrain requires flat qualification and a terrain capture
extension. [STATUS](../STATUS.md) lists each earlier attempt.
