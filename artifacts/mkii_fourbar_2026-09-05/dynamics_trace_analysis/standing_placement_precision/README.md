# Standing noise, solver iterations and world placement

The matched traces show a placement-associated velocity-noise floor and much
larger transients at 128 position iterations. They do **not** establish world
coordinate precision as the cause. Noise also increases at the world origin,
and robots at the same radius differ substantially.

This audit compares exactly physics samples `[640, 3200)`, the 0.8–4.0 s
standing interval, in two eight-environment traces. Both have the same v5 USD,
motor configuration, initial active joint coordinates and world placements.
Every actually delivered position target is identical between runs and constant
throughout standing; velocity targets and feedforward torques are zero. The
runtime-manifest differences are 128 versus 64 position iterations and target
ramp metadata. The changed scheduling implementation remains a source-code
confound, although the traced standing inputs match exactly. Each run uses one
velocity iteration, external forces each iteration and 1.25 ms physics.

| Env | Origin XY, m | Peak torque, 64 / 128, N·m | Peak joint speed, 64 / 128, rad/s |
|---|---|---:|---:|
| 0 | (2, −2) | 0.66573 / 1.39145 | 0.02374 / 1.53611 |
| 1 | (2, 0) | 0.66721 / 0.78959 | 0.01411 / 0.53781 |
| 2 | (2, 2) | 0.66587 / 0.90828 | 0.01956 / 0.72399 |
| 3 | (0, −2) | 0.66651 / 0.84113 | 0.01464 / 0.54367 |
| 4 | (0, 0) | 0.66752 / 0.66946 | 0.00655 / 0.07524 |
| 5 | (0, 2) | 0.66546 / 0.81469 | 0.01296 / 0.55058 |
| 6 | (−2, −2) | 0.66693 / 0.81053 | 0.02069 / 0.53951 |
| 7 | (−2, 0) | 0.66748 / 0.77770 | 0.00941 / 0.53419 |

Raw and applied peaks are equal in this window. Pearson correlation of radius
with joint-speed RMS is 0.762 at 128 iterations and 0.774 at 64. Radius versus
torque RMS after subtracting each joint's mean is 0.740 and 0.792 respectively.
These are descriptive statistics from eight placements with only three distinct
radii, not independent repeats or causal estimates. Environment index and world
placement are confounded. At the origin itself, joint-speed RMS increases from
0.000618 to 0.006432 rad/s; a pure distance-only explanation is insufficient.

## The largest derivative event

At 128 iterations, environment 0, the left-front lever's peak is sample 1,210
(pre-step time 1.5125 s). Demand is **1.39145 N·m = P 0.46978 + D 0.92167**.
The pre-step joint velocity is −1.53611 rad/s, whereas the position increment
during that step corresponds to only −0.002766 rad/s.

The adjacent snapshots show an actual contact/control transient:

| Sample | LF foot force Z, N | Post-step joint velocity, rad/s | Position-difference rate, rad/s | Applied lever torque, N·m |
|---|---:|---:|---:|---:|
| 1,207 | 7.073 | +0.32769 | +0.00753 | 0.40993 |
| 1,208 | 0 | −0.29828 | −0.39616 | 0.26430 |
| 1,209 | 0 | −1.53611 | +0.15960 | 0.65474 |
| 1,210 | 40.897 | −0.000011 | −0.00277 | 1.39145 |

Projected parent/child body angular velocities reproduce the reported joint
velocity, including −1.53611 rad/s. Independently differenced relative body
quaternions reproduce the small position-difference rate. Thus this is not a
joint-array cache mismatch or merely one mislabeled velocity field. It also is
not proof that the velocities are erroneous: PhysX's split position/velocity
constraint solution can produce different reported velocity and position-change
rates. The trace cannot expose the internal iterations that generated them.

## Plausibility and the next discriminating test

NVIDIA documents both loss of world-position precision away from the origin and
TGS's division of a physics step into one internal substep per position iteration.
Here those intervals are 19.53125 µs at 64 and 9.765625 µs at 128. The same
documentation explains the split-impulse distinction between reported velocity
and position changes. These facts make precision/iteration interaction plausible,
but do not identify the cause in this robot. [PhysX simulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html).

Near coordinate magnitude 2 m, recorded float32 positions have roughly
0.119–0.238 µm spacing. One such spacing divided by the 128-iteration interval
is 0.0122–0.0244 m/s, half as large at 64. This is a dimensional comparison, not
an observed internal solver calculation. Quaternion accumulation, contact bias,
constraint ordering and batch effects remain alternatives. The much longer
32-environment runs lack matched per-environment traces and cannot be used here
to separate elapsed time from count/placement.

The simplest next test is two fresh **one-environment, 600-standing-step,
128/1, Kp 30 / Kd 0.30** runs, translating the entire robot/terrain reference
from `(0, 0, 0)` to `(6, 0, 0)` while holding every other input fixed. Record
the same per-physics poses, joint velocities, P/D terms and contact forces, and
compare exactly the same settling/time window. Use the same seed and, if the
difference is small, repeat the origin run to assess run-to-run variability.
A difference that follows translation supports placement sensitivity; a quiet
pair would leave the multi-environment/batch hypothesis unresolved.

Co-locating 32 robots at zero is a stronger batch-controlled follow-up **only
after actual GPU environment-ID collision isolation is verified**. Otherwise
inter-robot contact changes the experiment. Neither test relaxes admission gates,
changes the robot geometry/inertias, or establishes that a different solver
recipe is qualified.

## Reproduction and identities

[analyze.py](analyze.py) verifies both standing NPZ hashes, layouts and finite
values, compares matched inputs and runtime metadata, and reconstructs the peak
derivative event using the hashed hinge frames. It uses NumPy only.
[results.json](results.json) contains every per-environment metric, exact source
paths/hashes, quantization scales, runtime differences and event snapshots.
[SHA256SUMS](SHA256SUMS) pins this new artifact without changing previous evidence.

Inputs, each using its adjacent `trace_000.npz`:

- 128/1: `/home/orionh/HEXAPOD_runs/mkii_fourbar_physical_mimic_v1/diagnostics/hexapod-fourbar-diagnose-20260905T174632Z-36232e95/report.json`; trace SHA-256 `817ee646ace603eca598889d0a400ffe27ac3f65952ed38cd84aeacf387f565b`.
- 64/1: `/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/diagnostics/hexapod-fourbar-diagnose-20260905T181945Z-ec78eb1e/report.json`; trace SHA-256 `0894c2b231eab60f6aa2e5044cb7782b119815779845df29f71e4ada4ecbe977`.
- Kinematics: `configs/mkii_fourbar_v3_kinematics.json`, SHA-256 `0ab3acc43d4c6c77c93a996858b8f928a1f157a533c8967c5ac84b77222f7f61`.

```sh
python3 analyze.py /path/to/128/report.json /path/to/64/report.json \
  /path/to/mkii_fourbar_v3_kinematics.json /new/path/results.json
```

The CPU analysis copy and result remain at
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/analysis/standing_placement_precision/`.
No GPU job, production source, model, acceptance gate or shared note changed.
