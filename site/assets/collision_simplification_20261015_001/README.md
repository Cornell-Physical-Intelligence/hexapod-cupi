# Collision simplification: 153 SDF meshes -> 19 primitives

Measured 15 October 2026 on branch `sps252/simple-collision` (`68886c29`), before the `src/` layout; paths below name the current layout.

Written for: the club. All numbers are measured on the Spark (DGX, GB10) unless marked.

## What changed

The club USD gives every CAD part its own SDF (signed-distance-field) mesh collider: 153 per robot, 1 mm contact
offset. SDF is the most expensive collider PhysX offers. `src/locomotion/hexapod/tools/make_simple_collision.py` replaces them with one
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
The generated file is gitignored; regenerate it on demand. The variant has no native admission, and `training_policy`
in `robot/active_model.json` forbids new training on it: it serves the throughput and fidelity diagnostics below until
it is admitted. `HEXAPOD_USD=<that path>` (read by `src/locomotion/hexapod/hexapod_robot_cfg.py`) selects it for those.
Each primitive carries the PhysxCollisionAPI offsets of the SDF parts it replaces (contactOffset 0.001 m,
restOffset 0, as the approved recipe authors them), and the tool refuses an `--out` that resolves to `--src`.

**One gotcha, found by the standing test:** the femur capsules (motor housings) overlap the body box at rest. PhysX
ignores contacts between links joined by a joint, but body-femur is two joints apart, so the overlap produced
spurious forces: every motor pinned at the 1.6 N·m cap and the body registering "contact" while standing still. The
fix is a `FilteredPairsAPI` on the body listing the six femurs. Everything else still self-collides.
`--visible` writes a debug asset with the shapes rendered semi-transparent for a visual check.

## Does the physics still match?

### Zero-shot transfer, five repetitions (26 October 2026, review of PR 87)

The policy trained on the SDF asset runs unchanged on each asset. Each repetition is a fresh process with an unset
env seed: 32 robots, forward commands 0.1-0.15 m/s, steps 100-500 averaged (8 s at 50 Hz). The two assets alternate,
so contention drift hits both alike. Values are the mean ± standard deviation over the 5 repetitions, with the range.
Welch t is the difference over its standard error.

| metric | SDF asset | simple asset (with offsets) | difference |
|---|---|---|---|
| speed along command [m/s] | 0.1208 ± 0.0016 (0.119-0.123) | 0.1249 ± 0.0034 (0.121-0.130) | +0.004 (+3 %), t = 2.4 |
| heading error [deg] | 4.68 ± 0.24 (4.32-4.90) | 5.67 ± 0.24 (5.44-6.01) | +0.99, t = 6.6 |
| joints at > 95 % of the 1.6 N·m cap | 1.6 % ± 0.2 | 4.3 % ± 0.4 | 2.6x, t = 12.5 |
| mean \|torque\| [N·m] | 0.387 ± 0.005 | 0.390 ± 0.005 | +0.004, t = 1.1 |
| body height [m] | 0.0948 ± 0.0001 | 0.0951 ± 0.0001 | +0.3 mm |
| total tibia contact force [N] | 24.75 ± 0.05 | 24.88 ± 0.16 | +0.13, t = 1.8 |
| falls (body contact) | 0 in 5 x 32 robots | 0 in 5 x 32 robots | |

What this supports: the policy walks on the variant at the same speed within about 3 %, with no falls. What it
does not support: that the swap is invisible. Heading error is about 1 deg larger and the share of joints at the
torque cap is 2.6 times higher, both well outside the repetition spread. One checkpoint is tested, so this is the
spread of rollouts, not of trained policies. Load comparisons here use the hexapod task's implicit PD actuator, not
the recorded 0.040 rad / 20 ms limiter, so they belong in a separate comparison from omniwalk's.

Bound to:
- policy `bcmin/model_299`: checkpoint `model_299.pt` sha256 `93fbcbfb0a6a3abdbb85b8d085d50a26a5561669e803d51a300c09eca2d229b2`, exported `policy.pt` sha256 `5700391f75a1eaf7174d7d8c885548640c5232c4339a2030aeb07e677ce89cc4`
- approved `robot.usda` sha256 `3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c`
- generated `robot_simple.usda` sha256 `0372cd514bbb5fdf2bdde802d152407cb5c732b2cc6cece6213aa2ca7e300bba`
- tool `src/locomotion/hexapod/tools/make_simple_collision.py` sha256 `10a157625be5a4680ad2bb844ab56dbc218309219d754e8def45f29b918be9dc`
- rollout tool `src/locomotion/hexapod/tools/diag_policy_rollout.py` from `main` at `d05d1f5c`

Raw logs: `~shaurya/src/isaaclab-learning/logs/collision_pr87/` on the Spark.

### First checks, one rollout each (15 October 2026)

Each row below is one rollout (scripted gaits: one run per gait; transfer: 32 robots, one process) on the
variant as the tool wrote it at `68886c29`. That tool was `hexapod/tools/make_simple_collision.py`, sha256
`5afff2ab8b63dd54caa0e2d69d39dc301e8ff1513419d72956d35d5cb79478f4`, and it authored no PhysxCollisionAPI
offsets, so PhysX chose its automatic contact offset. There is no spread for these rows. The transfer row is
superseded by the table above.

| test | SDF asset | simple asset (no offsets) |
|---|---|---|
| Standing, zero action: body height / holding torque / deaths in 100 steps | 0.0957 m / 0.31 N·m / 0 | 0.0953 m / 0.34 N·m / 0 |
| Scripted tripod (club gait, T 1.2 s, A 0.12): speed / falls | 0.041 m/s / 0 | 0.041 m/s / 0 |
| Scripted tripod T 0.8 s, A 0.2: speed / femur saturation | 0.101 m/s / 13 % | 0.104 m/s / 13 % |
| Scripted tripod T 0.8 s, A 0.3: speed | 0.120 m/s | 0.131 m/s |
| Fastest scripted gait (T 0.6 s, A 0.3): speed | 0.171 m/s | 0.144 m/s |
| 15 scripted gaits: falls | 0 | 0 |
| Zero-shot transfer of `bcmin/model_299`: speed along command / heading error / falls | 0.123 m/s / 4.6 deg / 0 | 0.1225 m/s / 5.8 deg / 0 |

In these single runs, scripted gaits agreed within ~10 % except the most aggressive one (T 0.6 s, A 0.3), where
the capsule feet slipped more than the SDF toes at high foot speed (0.171 -> 0.144 m/s). Peak torques were
0.0-0.2 N·m higher on the simple asset. Raw logs and videos are on the Spark under
`~shaurya/src/isaaclab-learning/logs/collision/`.

## Throughput

Measured on the variant written at `68886c29` (no PhysxCollisionAPI offsets); not re-measured with the offsets.
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
# inside the Isaac Lab container, from the Isaac Lab root, with the repo src/ on PYTHONPATH
./isaaclab.sh -p <repo>/src/locomotion/hexapod/tools/make_simple_collision.py      # writes robot_simple.usda next to robot.usda
export HEXAPOD_USD=<repo>/robot/hexapod_mkii_updated_v1/usd/robot_simple.usda
# zero-shot transfer of a trained policy (the decisive fidelity check), exploration off, standing start:
./isaaclab.sh -p <repo>/src/locomotion/hexapod/tools/diag_policy_rollout.py --headless --policy <run>/exported/policy.pt ...
# throughput: train.py for 20 iterations per (asset, num_envs) with the GPU otherwise idle,
# compare 'Steps per second' after the first 5 iterations; unset HEXAPOD_USD for the SDF baseline
```
The standing and scripted-gait diagnostics used above live in the author's Spark workspace and are not in the repo.
