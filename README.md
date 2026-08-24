# Hexapod MKII — RL Walking

Long-running reinforcement-learning project to train walking gaits for the Hexapod MKII.
The URDF here (`robot/hexapod_mkii_mock_assy/`) is a **mock** of the real robot, exported
from Onshape, and is the model used for simulation training.

## Robot model

- 26 links, 25 joints (18 actuated revolute: coxa / femur / tibia × 6 legs)
- Joint limits: coxa ±0.87 rad, femur 0 – 1.75 rad, tibia 0 – 2.53 rad
- Meshes: binary STL, meters, Z-up
- Validated: single-rooted kinematic tree (`root`), no cycles, all mesh references resolve,
  all revolute joints have `<limit>` elements

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
viewer/                         Vite + React + urdf-loader web viewer
```

## Next steps

- Add collision geometry + realistic mass/inertia (current values are Onshape defaults at mock scale)
- Build MuJoCo/Isaac/PyBullet training environment from this URDF
- Sim-to-real transfer to the physical MKII
