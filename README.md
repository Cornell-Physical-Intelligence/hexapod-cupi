# Hexapod MKII — RL Walking

Long-running reinforcement-learning project to train walking gaits for the Hexapod MKII.
The original URDF in `robot/hexapod_mkii_mock_assy/` is an Onshape export. The
`hexapod_mkii_robstride.urdf` variant and its USD are the physics/training assets.

## Robot model

- 26 links, 25 joints (18 actuated revolute: coxa / femur / tibia × 6 legs)
- Joint limits: coxa ±0.87 rad, femur 0 – 1.75 rad, tibia 0 – 2.53 rad
- Meshes: binary STL, meters, Z-up
- Validated: single-rooted kinematic tree (`root`), no cycles, all mesh references resolve,
  all revolute joints have `<limit>` elements
- Physics mass target: 6.3 kg total — a 1.5 kg body plus six 0.8 kg complete legs;
  each leg total already includes its three 191 g RobStride RS05 actuators
- RS05 policy limit: 1.6 N·m continuous, 5.5 N·m published short-duration peak,
  50.27 rad/s nominal no-load speed, and 0.0007 kg·m² output armature

The 1.5 kg body and 0.8 kg complete-leg values are user-specified mass targets, not
measurements of the assembled robot; 191 g is the published per-actuator mass. The
link-level mass distribution and inertias therefore remain estimates. RS05 housing
inertia, joint friction, latency, and thermal parameters are not published; the
generator records the explicit estimates used here. Measure these properties before
treating a policy as hardware-ready.

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
python3 tools/generate_robstride_urdf.py
# Import the generated URDF as a floating-base USD with Isaac Sim's URDF importer.
# Then, using Isaac Sim Python:
python tools/enable_nested_contact_reports.py \
  robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
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
