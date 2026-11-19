# Working with bm0: run it, judge it, view it, change it, put it on the robot

A hands-on guide for anyone who checks out `jg2399/distilled-bm0`. [README.md](README.md) is the record of
what bm0 is and how it scored; this file is how to use it. Every command below runs from the root of a
hexapod-cupi checkout of this branch. The judge, run-file, fine-tune, export and runtime commands were
run on 2026-11-18 on a MacBook (CPU only) before this guide was written; the timings are from that
machine.

## 1. What bm0 is, in one paragraph

A walking policy for the hexapod: a 278 -> 256 -> 256 -> 128 -> 18 MLP (ELU, tanh output; 172,434
parameters on the robot). Every 20 ms (50 Hz) it reads the IMU, the 18 joint angles and speeds, the
command (forward m/s, left m/s, turn rad/s) and a leg-odometry "ghost" estimate, and outputs 18 joint
position targets. It was distilled on the CPU MuJoCo surrogate from pfit1 u2000 with a delay predictor in
front, on a body whose centre piece is nylon plus 0.75 kg (7.280 kg robot), under the five actuator
axes fitted on the leg stand (delay, kd, toe friction, joint friction, rotor inertia). It walks and stops
but misses the 96-case gate (19-20/96): it tracks speed and turn rate about 30-90 % past the bounds and
falls in tight arcs. It is a diagnostic, not a lineage candidate.

## 2. Set up

```sh
uv sync                         # Python 3.12 project environment (CPU torch is enough)
cd viewer && npm ci && cd ..    # only for the browser viewer (section 4)
```

Files you will touch:

| path | what |
|---|---|
| `records/distilled_bm0_u1000_20261112/checkpoint_update001000.pt` | the model (actor + critic, PyTorch) |
| `records/distilled_bm0_u1000_20261112/checkpoint_update001000.json` | its declaration: task, wrapper settings, source hashes, seed |
| `records/distilled_bm0_u1000_20261112/scripts/body_mass.py` | runs any surrogate script on the nylon + 0.75 kg body |
| `records/distilled_bm0_u1000_20261112/scripts/fitted_eval.py` | the 96 cases under the fitted actuator dynamics |
| `records/distilled_bm0_u1000_20261112/full96_suite.py` | the 96 cases under canonical (nominal) dynamics |
| `src/locomotion/surrogate/` | the MuJoCo surrogate: `train.py`, `evaluate.py`, `env.py` |
| `src/locomotion/omniwalk/destination_task.py` | the task and reward (`destination_v70`) |
| `src/locomotion/runtime/` | numpy-only robot runtime (section 6) |
| `viewer/` | browser viewer: `run_view.py` and the Runs and Scene pages |

Use `B=records/distilled_bm0_u1000_20261112` below to keep commands short.

## 3. Run and judge it

**One case, with a video** (about 30 s). The case ids are `learning:` plus the evaluator's id with `:` as
`__`; list them with `grep case_id` on any `surrogate_summary.json`, or see the table in section 3.1.

```sh
B=records/distilled_bm0_u1000_20261112
uv run python $B/full96_suite.py --checkpoint $B/checkpoint_update001000.pt --output out/one \
    --cases learning:static__translate_0.10_0deg --video all --no-contact-consensus
```

Output: `out/one/surrogate_summary.json` (verdict and failed checks per case), `out/one/videos/*.mp4`
(offscreen MuJoCo, three-quarter view; `--view side` for a side view) and `out/one/evaluation/batch_*/`
(the control trace the viewer reads). `--cases` takes a comma-separated list; `--video none` skips
rendering, which is about four times faster. `--output` must be a new directory.

**On bm0's own body**: put `uv run python $B/scripts/body_mass.py run` in front of any script:

```sh
uv run python $B/scripts/body_mass.py run $B/full96_suite.py --checkpoint $B/checkpoint_update001000.pt \
    --output out/one_nylon --cases learning:static__stand --no-contact-consensus
```

**The recorded result** (the fitted dynamics, all 96 cases, about 30 min; `--match stand` is a 3-case,
about 70 s smoke test that should print `pass 3/3`):

```sh
uv run python $B/scripts/body_mass.py run $B/scripts/fitted_eval.py out/fitted \
    bm0_u1000=$B/checkpoint_update001000.pt --motor-check drawn --seed 20261110
```

The last line reads `FITTED bm0_u1000: pass N/96 ...`; the recorded runs are 19 (seed 20261110) and 20
(seed 20261111). `--match TEXT ...` keeps the case ids that contain any TEXT. The fitted dynamics are
drawn per episode in run order, so a `--match` subset gives each case a different draw than the full
run: subset verdicts are representative, not a replay of the recorded ones.

### 3.1 The 96 cases

| group | count | commands | bm0 (two seeds) |
|---|---|---|---|
| `static__stand` | 1 | standing still | 2/2 |
| `static__translate_0.10_*`, `_0.20_*` | 32 | 0.10 and 0.20 m/s on 16 bearings (every 22.5 deg) | 6/64 |
| `static__turn_*` | 4 | +-0.20, +-0.40 rad/s in place | 3/8 |
| `static__combined_*` | 8 | 0.10 m/s on 4 bearings with +-0.20 rad/s | 1/16 |
| `static__gentle_arc_*` | 16 | 0.15 m/s on 8 bearings with +-0.05 rad/s | 0/32 |
| `static__tight_arc_*` | 16 | 0.20 m/s with +-0.40 rad/s | 0/32, all fall |
| `stop__*` | 12 | walk or turn, then stop | 22/24 |
| `quiet__*` | 2 | long stands | 3/4 |
| `formal__*` | 4 | 0.00, 0.16, 0.20, 0.30 m/s forward | only 0.00 passes |
| `transition__full_program` | 1 | a 70 s program of changing commands | fails |

The bounds are in `src/locomotion/evaluation/scoring.py`: planar speed error at most
max(0.025, 0.25 |v|) m/s and turn-rate error at most max(0.06, 0.25 |w|) rad/s, plus no fall, torque,
joint-speed and non-foot contact checks.

## 4. See it in the viewer

The viewer replays evaluation runs: legs from the robot model's joint frames, toes filled while on the
floor, the body path in grey, the ghost as a dashed orange line, the heading against the commanded
heading, and the evaluator's verdict and failed checks per case.

1. Judge the cases you want to see (section 3) without video, for example the low-speed set:

   ```sh
   uv run python $B/scripts/body_mass.py run $B/scripts/fitted_eval.py out/view \
       bm0_u1000=$B/checkpoint_update001000.pt --motor-check drawn --seed 20261110 \
       --match static__stand static__translate_0.10_0deg static__translate_0.10_90deg \
               static__translate_0.10_180deg static__turn_+0.20 static__combined_0deg_+0.20 \
               static__gentle_arc_0deg_+0.05 stop__arc_left
   ```

   (about 3 minutes; `fitted_eval.py` writes one folder per checkpoint, here `out/view/bm0_u1000`).
2. Turn the run into a run file:

   ```sh
   uv run python viewer/run_view.py out/view/bm0_u1000 --output bm0.json
   ```

3. Open the viewer and load the file:

   ```sh
   cd viewer && npm run dev
   ```

   Open the address Vite prints with `#runs` on the end, press **Open run files...** and pick
   `bm0.json` (or drop it on the page). Tick cases in the table (at most 6 panels at once; ticking a
   seventh drops the oldest), switch **Top / 3D / Side**, drag to orbit, and scrub or play with the
   slider. To open it straight from a URL, build the viewer and serve the run file beside it:

   ```sh
   cd viewer && npx vite build --outDir ../out/site && cd .. && cp bm0.json out/site/
   python3 -m http.server 8774 --bind 127.0.0.1 --directory out/site
   ```

   then open `http://localhost:8774/?run=bm0.json#runs`.

Any surrogate output folder works with `run_view.py`: section 3's `out/one`, a fine-tune's judge, or
another checkpoint for a side-by-side (load several run files; panels share one clock).

**Your own command sequence on terrain**: the Scene page (`#scene`) builds a scene (terrain, one or more
robots, each with a checkpoint and a list of commands), saves it, and plays its simulation in the CAD look.
Simulate a saved scene with `uv run python -m locomotion.surrogate.scene scene.json --output scene_out`,
then `uv run python viewer/scene_view.py scene_out --output scene_view.json` and open that on the page.
bm0 never saw terrain, so treat terrain scenes as design play. Details: `viewer/README.md`.

## 5. Change it

### 5.1 Fine-tune bm0 with PPO on the surrogate

This is the recipe bm0 was trained under (`scripts/bmass1.sh`), with bm0 itself as the start and the
stock PPO trainer instead of distillation. Any flag can change; the trainer refuses combinations it
cannot load.

```sh
uv run python $B/scripts/body_mass.py run src/locomotion/surrogate/train.py --output out/ft \
    --task "locomotion.omniwalk.destination_task:experiment_task(destination_v70)" \
    --wrapper locomotion.rl.destination_wrapper:DestinationVecEnv \
    --initialize-from $B/checkpoint_update001000.pt --initialize-rate checkpoint \
    --action-mean tanh --observation-normalization none --observation-scaling fixed \
    --command-segments bootstrap --learning-rate-max 1e-4 --action-std 0.05 --action-std-final 0.05 \
    --gait-clock 60 --action-smoothing mean2 --velocity-noise 0.5 --episode-seconds 32.0 \
    --num-envs 128 --updates 200 --seed 20261110 --threads 4 \
    --contact "touchdown_overshoot=False" \
    --randomize "action_delay_controls=(1, 2)" \
    --randomize "kd_scale=(0.13, 0.37)" \
    --randomize "friction_scale=(0.45, 0.6)" \
    --randomize "joint_friction_nm=((0.0335, 0.0622), (0.1387, 0.2576), (0.018, 0.0335))" \
    --randomize "armature_kg_m2=((0.0, 0.0001), (0.0, 0.00058), (0.0, 0.00049))" \
    --ppo-config-overrides '{"num_steps_per_env": 36, "algorithm": {"num_mini_batches": 6}, "environment_wrapper": {"destination_input": "terrain_v2_odom", "gait_clock_scaling": "speed_v1"}}'
```

It writes `checkpoint_update*.pt` with its `.json` declaration every `--checkpoint-interval` updates
(default 50), plus `metrics.jsonl` (one line per update: reward, command speed, timings) and
`body_mass.json`. Judge a new checkpoint exactly as in section 3. A 2-update, 16-robot smoke run of this
command takes about 15 s; drop the `body_mass.py run` prefix to train on the nominal 7.466 kg body.

What to change, and where:

| to change | where |
|---|---|
| actuator randomization (delay, kd, friction, inertia) | the `--randomize` ranges above; the fitted values are in `records/leg_stand_20261109/` |
| reward, command mix, curriculum | `src/locomotion/omniwalk/destination_task.py`: copy the `destination_v70` entry of `EXPERIMENTS` to a new name and pass that name to `--task` |
| observation noise | `--velocity-noise` (joint speeds only, rad/s); other inputs have none |
| body mass | `ADDED` and `NYLON` at the top of `scripts/body_mass.py` |
| network size, PPO settings | `--ppo-config-overrides` (merged into `src/locomotion/omniwalk/agents/rsl_rl_ppo_cfg.py`) |
| the inputs themselves | `src/locomotion/rl/destination_wrapper.py` and `rl/wrapper.py`; a new input set needs a new task name, a new run from scratch, and the runtime's `inputs.py` changed to match |

Rules the project holds to: a changed input set or reward gets a fresh task name, never an edit of an
existing `EXPERIMENTS` entry (checkpoint declarations pin the source files by hash); results on the
surrogate are design evidence, and acceptance needs a native run on Spark.

### 5.2 Re-run the distillation

`scripts/distill_train.py` (DAgger) and `scripts/bmass1.sh` reproduce bm0 itself, but they need the
teacher (pfit1 u2000) and predictor B, which are not in this branch; the paths in `bmass1.sh` are the
training laptop's.

## 6. Run it outside the simulator (the robot runtime)

`src/locomotion/runtime/` runs the policy with numpy alone (no torch, no MuJoCo):

| file | job |
|---|---|
| `export.py` | checkpoint -> `bundle.json` + `weights.npz`; refuses wrapper settings it does not reproduce |
| `bundle.py` | loads the bundle and runs the MLP |
| `inputs.py` | `PolicyLoop`: the 278 inputs from sensor readings (5-frame history, command, gait clock, leg-odometry ghost), then the 2-action mean and the 0.04 rad slew limit |
| `controller.py` | the 50 Hz state machine DISABLED -> STANDING_UP -> READY -> WALKING, FAULT on any failed check, command expiry; returns position, kp, kd and torque limit per joint |
| `hardware.py` | motor id, sign and zero per joint, and the IMU mounting rotation, from a calibration file |

Export and drive it with made-up readings:

```sh
uv run python -m locomotion.runtime.export --checkpoint $B/checkpoint_update001000.pt --output out/bm0_bundle
```

```python
import numpy as np
from contracts.command import CommandEnvelope
from locomotion.runtime.bundle import load
from locomotion.runtime.controller import Controller
from locomotion.runtime.inputs import Reading

bundle = load("out/bm0_bundle")
control = Controller(bundle, CommandEnvelope((-0.15, 0.15), (-0.1, 0.1), (-0.4, 0.4)))
reading = Reading(
    gyro_body=np.zeros(3), gravity_body=np.array([0.0, 0.0, -1.0]),   # body frame, level
    joint_position=bundle.neutral.copy(), joint_velocity=np.zeros(18),
    contact=np.ones(6, dtype=bool),                                    # or None / contact_valid=False
)
t = 0.0
control.enable()                         # STANDING_UP, then READY once at neutral
for _ in range(200):
    out = control.step(reading, t); t += 0.02
control.start()                          # WALKING
control.set_command(0.1, 0.0, 0.0, t)    # forward 0.1 m/s; expires if not refreshed
out = control.step(reading, t)
print(out.mode, out.position[:3], out.kp[:3], out.kd[:3], out.torque_limit[:3])
```

What the robot must supply each control: gyro and gravity direction in the body frame (forward = -body
y, left = body x, up = body z), the 18 joint angles and speeds in `bundle.joint_names` order, and six
foot-contact booleans with a validity flag (an invalid leg falls back to the gait schedule; never feed
"always in contact"). The motor drivers must run the trained law, torque = kp (target - q) - kd dq at
kp 12 Nm/rad, per-joint kd, limited to 1.6 Nm. The command is in the robot's frame when it is given;
the ghost then holds that direction in the odometry frame until the next command.

Not built yet: the loop that reads the RS05 motors over CAN and the IMU and sends the targets back, and
the calibration file (`hardware.load_calibration`): `{"joints": {"lf_coxa_yaw": {"motor_id": 11,
"sign": 1, "zero_rad": ...}, ...}, "imu": {"body_from_imu": [[...], [...], [...]]}}`. Motor ids
follow leg x 10 + joint (legs lf, lm, lr, rf, rm, rr = 1-6; coxa yaw, femur pitch, tibia pitch = 1-3),
so the leg stand's lf leg is 11-13. The leg-stand actuator fit is in `records/leg_stand_20261109/`.

Check the runtime against the evaluator after changing it (replays a v70 checkpoint through both paths):

```sh
uv run python -m unittest src/locomotion/tests/test_runtime.py
```

## 7. Things that trip people up

- `--output` folders must not exist; every script refuses to overwrite.
- `out/` and `bm0.json` in these examples are not git-ignored: keep them out of commits, or write
  outputs outside the checkout.
- `body_mass.py run` takes a script *path*, not a module (`src/locomotion/surrogate/train.py`, not
  `-m locomotion.surrogate.train`).
- The judge with `--motor-check canonical` stops at the first case under kd draws; keep `drawn` for
  fitted runs.
- Run judges one at a time on a laptop: each uses several cores, and parallel runs slow each other more
  than they gain.
- The surrogate is not Isaac Lab: numbers here are design evidence. Spark launches go through
  `locomotion.spark.launch` and are reviewed separately.
