# Hexapod MKII — RL Walking

Long-running reinforcement-learning project to train walking gaits for the Hexapod MKII.
The original URDF in `robot/hexapod_mkii_mock_assy/` is an Onshape export. The
`hexapod_mkii_robstride.urdf` variant and its USD are the physics/training assets.

## Robot model

The training asset is the CAD assembly package `robot/hexapod_mkii_assy/`
(built from the Onshape export by `robot/tools/import_onshape_hexapod.py`;
`robot/hexapod_mkii_assy/assembly_report.md` audits the import):

- `urdf/hexapod_mkii_serial.urdf`: 19 links, 18 actuated revolute joints
  (`<leg>_coxa_yaw`, `<leg>_femur_pitch`, `<leg>_tibia_pitch` for legs
  `lf lm lr rf rm rr`), root link `body`
- Body frame: z up, forward = -y, left = +x; coxa_yaw zero points the leg
  straight out, femur/tibia zero is the CAD pose (see
  `robot/hexapod_mkii_assy/joint_limits.json` for the sign conventions)
- Joint limits: coxa +-0.87 rad, femur -1.745 to +0.55 rad, tibia -0.95 to
  +1.75 rad (femur/tibia derived from the CAD kinematics; replace with
  measured hardware stops)
- Mass 8.261 kg from Onshape's per-part properties with every RS05 hard-set
  to its published 191 g; per-part collision primitives; meshes: binary STL,
  meters, z up
- Reset/validation stance (`robot/hexapod_mkii_assy/stance.json`): femur
  -0.25 rad, tibia -0.55 rad, bottom plate 0.124 m above the ground, static
  hip and knee torque 0.9 N*m each
- RS05 policy limit: 1.6 N*m continuous, 5.5 N*m published short-duration
  peak, 50.27 rad/s nominal no-load speed, and 0.0007 kg*m^2 output armature

`robot/hexapod_mkii_mock_assy/` is the earlier mock (normalised 1.5 kg body /
0.8 kg legs) that the archived Phase 1-2 checkpoints were trained on; its
`revolute_*` joint names and stance values in `phase2_cfg.py` belong to that
asset.

## Isaac Lab training

The direct locomotion task lives in `isaaclab/hexapod_rl`. It runs PhysX at 200 Hz,
the policy at 50 Hz, and trains an 18-position-action PPO policy through a staged
curriculum: stabilized anatomical-forward walking, signed lateral acquisition, then
full navigation-frame `x/y/yaw` joystick commands and command transitions.

The stabilized curriculum uses the lower insect-like reset stance selected by the
controlled stance sweep: a `0.185 m` root, `0.6000 rad` femur pitch, and `2.2335 rad`
tibia angle. Its moving and standing deck targets are `0.181 m` and `0.177 m`.
The reward and admission screens separately constrain deck heave, roll/pitch rate,
tilt, height error, action/torque slew, foot slip, falls, command tracking, and every
RS05 motor's torque demand. Later joystick checkpoints are not accepted merely for
walking in one direction: reverse, both lateral signs, both yaw signs, mixed-axis
commands, and uninterrupted command changes all have command-local gates.

On the configured DGX Spark (`spark-e26c`):

```sh
hexapod-rl status       # persistent training service
hexapod-rl logs         # live PPO metrics
hexapod-rl latest       # newest checkpoint
hexapod-rl stop         # cleanly stop the run
hexapod-rl start        # start a fresh configured run
```

Rebuild the source assets with:

```sh
# URDF package from the onshape-to-robot export (macOS/Linux, numpy + scipy)
python3 robot/tools/import_onshape_hexapod.py --source "<export dir>"
# USD for Isaac Sim (Isaac Sim python, headless), then contact reports:
python tools/import_urdf_to_usd.py \
  robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
python tools/enable_nested_contact_reports.py \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
# Standing validation gate (Isaac Lab):
python isaaclab/validate.py --num_envs 32 --steps 1000
```

The validation gate checks 18 joints, 19 rigid bodies, six foot sensors, finite
observations, 1,000 standing steps, falls/timeouts, raw pre-clip torque demand,
and post-settling saturation.

Varied-terrain sensor-fusion training is a separate approval-gated phase. The current
simulation proposal uses a forward depth camera plus a 360-degree near-hemispherical
LiDAR; do not start a long terrain/sensor-fusion run until the physical sensor model
has been identified and the user has approved the run.

## Viewer

Interactive three.js/React viewer with per-joint sliders:

```sh
cd viewer
npm install
npm run dev   # http://localhost:5173
```

## Layout

```
robot/hexapod_mkii_mock_assy/   URDF package (urdf/, meshes/, launch/)
isaaclab/                       Direct RL task, validator, deployment helpers
tools/                          Reproducible URDF/USD physics preparation
viewer/                         Vite + React + urdf-loader web viewer
```

## Sim-to-real gates still required

- Replace/identify the assumed body/leg mass distribution, housing inertia, and friction.
- Add measured CAN/control delay, encoder/IMU noise, voltage sag, and an RS05 thermal/I²t model.
- Refine link collisions into primitives plus explicit foot pads and validate self-collision.
- Verify observation parity and joint signs on a restrained robot before powered walking.
- Enforce the 1.6 N·m training limit, joint limits, watchdog, and E-stop on hardware.
