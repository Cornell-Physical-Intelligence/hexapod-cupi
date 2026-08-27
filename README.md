# Hexapod MKII — RL Walking

Long-running reinforcement-learning project to train walking gaits for the
Hexapod MKII, using Isaac Sim 6.0.1 / Isaac Lab `DirectRLEnv` with RSL-RL PPO.
The original URDF in `robot/hexapod_mkii_mock_assy/` is an Onshape export; the
`hexapod_mkii_robstride.urdf` variant and its USD are the physics/training
assets.

**Start with [`STATUS.md`](STATUS.md)** for the current best checkpoint, the
current target, open contradictions, and next actions. This README is a map,
not a status page.

## Repository map

```text
STATUS.md                       Current state. Rewritten in place, never appended
CLAUDE.md                       Agent/contributor guide and project invariants
HANDOFF.md                      Stub pointing at the split documentation
docs/TRAINING.md                Task and training design, curriculum, acceptance gates
docs/OPERATIONS.md              Spark runbook, GPU protocol, launchers, screens
docs/incidents/                 Frozen forensic records of specific failures
docs/archive/                   Superseded documents, kept verbatim
robot/hexapod_mkii_mock_assy/   URDF package (urdf/, meshes/, launch/)
packages/hexapod_core/          Frozen interface contracts: observation, action, command, joints
packages/hexapod_env/           Direct RL task: env, rewards/, stage configs, registration
packages/hexapod_train/         Run composition, run contract, startup supervisor
packages/hexapod_eval/          Acceptance gates (gates.py) and the evaluation entry point
packages/hexapod_runtime/       Deployable policy runtime: observation builder, action pipeline
packages/hexapod_nav/           Command-producer seam and an example waypoint follower
configs/                        Named experiment baselines and intervention deltas
isaaclab/hexapod_rl/            Compatibility shims re-exporting packages/hexapod_env
isaaclab/deploy/                Launchers, services, and the pipeline SHA-256 manifest
ops/                            hexctl operator CLI; attic/ holds retired deploy scripts
isaaclab/tests/                 Unit and contract tests
tools/                          Reproducible URDF/USD physics preparation
viewer/                         Vite + React + urdf-loader web viewer
artifacts/                      Evidence: checkpoints, evaluations, videos, probe ledger
```

## Quickstart

Run the test suite (no simulator required):

```sh
python3 -m unittest discover -s isaaclab/tests
```

Without `torch`, `numpy`, `pyyaml`, and `opencv-python-headless` installed, the
ten torch-dependent modules surface as 10 import errors and only the
dependency-free tests run. The full suite needs those four dependencies; its
total grows with every added test, so read the count off the run rather than
from this page.

Run the interactive viewer:

```sh
cd viewer
npm install
npm run dev   # http://localhost:5173
```

Regenerate the training assets from source:

```sh
python3 tools/generate_robstride_urdf.py
# Import the generated URDF as a floating-base USD with Isaac Sim's URDF importer.
# Then, using Isaac Sim Python:
python tools/enable_nested_contact_reports.py \
  robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
```

Training and evaluation run on the DGX Spark host, not locally. See
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Robot model

- 26 links, 25 joints (18 actuated revolute: coxa / femur / tibia × 6 legs)
- Joint limits: coxa ±0.87 rad, femur 0 – 1.75 rad, tibia 0 – 2.53 rad
- Meshes: binary STL, meters, Z-up
- Validated: single-rooted kinematic tree (`root`), no cycles, all mesh
  references resolve, all revolute joints have `<limit>` elements
- Physics mass target: 6.3 kg total — a 1.5 kg body plus six 0.8 kg complete
  legs; each leg total already includes its three 191 g RobStride RS05 actuators
- RS05 policy limit: 1.6 N·m continuous, 5.5 N·m published short-duration peak,
  50.27 rad/s nominal no-load speed, 0.0007 kg·m² output armature

The 1.5 kg body and 0.8 kg complete-leg values are user-specified mass targets,
not measurements of the assembled robot; 191 g is the published per-actuator
mass. Link-level mass distribution and inertias therefore remain estimates, and
RS05 housing inertia, joint friction, latency, and thermal parameters are not
published. The remaining hardware prerequisites are listed under "Sim-to-real
gates still required" in [`docs/TRAINING.md`](docs/TRAINING.md#sim-to-real-gates-still-required);
no checkpoint here is hardware-ready.

Varied-terrain sensor-fusion training is a separate approval-gated phase. Do
not start a long terrain or sensor-fusion run without explicit user approval.
