# Collision simplification: 153 SDF meshes -> 19 primitives

Written for: the club. All numbers are measured on the Spark (DGX, GB10) unless marked.

## What changed

The club USD gives every CAD part its own SDF (signed-distance-field) mesh collider: 153 per robot, 1 mm contact
offset. SDF is the most expensive collider PhysX offers. `tools/make_simple_collision.py` replaces them with one
primitive per link, fitted to the same CAD parts:

| link | shape | fitted size |
|---|---|---|
| body | box | 0.29 x 0.45 x 0.067 m (bounding box of its parts) |
| coxa (x6) | capsule along +x | r 3.1 cm, x in [-0.025, 0.083] |
| femur (x6) | capsule along +x | r 3.2 cm, x in [-0.016, 0.097] |
| tibia (x6) | capsule along +x | r 1.76 cm, x in [-0.024, 0.130]; far end-cap = the toe tip |

Radii come from the 95th-percentile cross-section of each segment's distal half, so the leg capsules are the motor
housings and the tibia's is its shaft/toe. Visual meshes, masses, inertias, joints and limits are untouched.
Output: `robot/hexapod_mkii_updated_v1/usd/robot_simple.usda`, next to the approved `robot.usda`, which is not modified.
The generated file is gitignored; regenerate it on demand. Train on it with `HEXAPOD_USD=<that path>` (read by
`hexapod_robot_cfg.py`), no code change.

**One gotcha, found by the standing test:** the femur capsules (motor housings) overlap the body box at rest. PhysX
ignores contacts between links joined by a joint, but body-femur is two joints apart, so the overlap produced
spurious forces: every motor pinned at the 1.6 N·m cap and the body registering "contact" while standing still. The
fix is a `FilteredPairsAPI` on the body listing the six femurs. Everything else still self-collides.
`--visible` writes a debug asset with the shapes rendered semi-transparent for a visual check.

## Does the physics still match? (yes)

| test | SDF asset | simple asset |
|---|---|---|
| Standing, zero action: body height / holding torque / deaths in 100 steps | 0.0957 m / 0.31 N·m / 0 | 0.0953 m / 0.34 N·m / 0 |
| Scripted tripod (club gait, T 1.2 s, A 0.12): speed / falls | 0.041 m/s / 0 | 0.041 m/s / 0 |
| Scripted tripod T 0.8 s, A 0.2: speed / femur saturation | 0.101 m/s / 13 % | 0.104 m/s / 13 % |
| Scripted tripod T 0.8 s, A 0.3: speed | 0.120 m/s | 0.131 m/s |
| Fastest scripted gait (T 0.6 s, A 0.3): speed | 0.171 m/s | 0.144 m/s |
| 15 scripted gaits: falls | 0 | 0 |
| **Zero-shot transfer**: the policy trained on the SDF asset (`bcmin/model_299`), run on each asset. Speed along command / heading error / falls | 0.123 m/s / 4.6 deg / 0 | **0.1225 m/s / 5.8 deg / 0** |

The trained policy does not notice the swap. Scripted gaits agree within ~10 % except the most aggressive one
(T 0.6 s, A 0.3), where the capsule feet slip more than the SDF toes at high foot speed (0.171 -> 0.144 m/s). Peak
torques are 0.0-0.2 N·m higher on the simple asset across gaits. Raw logs and videos are on the Spark under `~shaurya/src/isaaclab-learning/logs/collision/`.

## Throughput

Same config for every cell (the clean bc overrides); 20 iterations per run, first 5 dropped as warm-up; median
steps/s over the rest. Variants interleaved so GPU-contention drift hits all alike; GPU utilisation noted before
each run (raw logs on the Spark, see above). Row C adds one more override on top of B
(`env.robot.spawn.articulation_props.solver_position_iteration_count=4`, the club USD authors 32) to separate the
collider cost from the solver-iteration cost.

Measured 2026-10-15 18:32-19:12 EST, GPU otherwise idle (no club run active), every run completed (no hangs).

| envs | A: 153 SDF | B: 19 primitives | speed-up B/A | C: B + solver iters 32 -> 4 | C/A |
|---|---|---|---|---|---|
| 128 | 720 steps/s (4.27 s/iter) | 1,730 (1.77) | **2.4x** | 3,496 (0.88) | 4.9x |
| 512 | 1,788 (6.88) | 5,874 (2.09) | **3.3x** | 11,527 (1.07) | 6.4x |
| 2048 | 3,016 (16.30) | 13,678 (3.59) | **4.5x** | 34,332 (1.43) | 11.4x |

A and B are the median of 2 interleaved repetitions (repetitions agree within 5 %); C ran once.
Peak host memory at 2048 envs: A 73 GB, B 27-29 GB, C 27 GB (unified CPU/GPU pool, 121 GB total).

What it means:
- The simple asset is 2.4-4.5x faster, and the gain grows with env count: SDF contact cost scales with the
  number of shapes x envs, primitives barely register. The 600-iteration BC fine-tune at 512 envs drops from
  ~70 min to ~21 min (computed from s/iter).
- 2048 envs, which was impractical with SDF (16 s/iter, 73 GB), is now 3.6 s/iter. The four-times-larger batch
  per update is the usual lever for PPO quality (ANYmal's reference config uses 4096).
- The club USD's 32 solver position iterations cost another 2-2.5x. **Row C's physics fidelity is NOT tested**:
  4 iterations is Isaac Lab's ANYmal value, but joint drift and contact softness at 4 iterations on this
  light-legged robot need the same standing / open-loop / policy-transfer checks before anyone trains on it.

## Reproduce

```bash
# inside the Isaac Lab container, from the Isaac Lab root, with the repo root on PYTHONPATH
./isaaclab.sh -p <repo>/hexapod/tools/make_simple_collision.py      # writes robot_simple.usda next to robot.usda
export HEXAPOD_USD=<repo>/robot/hexapod_mkii_updated_v1/usd/robot_simple.usda
# zero-shot transfer of a trained policy (the decisive fidelity check), exploration off, standing start:
./isaaclab.sh -p <repo>/hexapod/tools/diag_policy_rollout.py --headless --policy <run>/exported/policy.pt ...
# throughput: train.py for 20 iterations per (asset, num_envs) with the GPU otherwise idle,
# compare 'Steps per second' after the first 5 iterations; unset HEXAPOD_USD for the SDF baseline
```
The standing and scripted-gait diagnostics used above live in the author's Spark workspace and are not in the repo.
