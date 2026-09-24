# AMP demonstration preparation

You can generate optimized tripod candidates for issue [#36](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/36),
replay them in Isaac and export reviewed transitions. James (`palerdr`) owns
this work and reviews the motion. CPU feasibility does not admit a dataset.
The discriminator and policy comparison belong to #37; this package does not
require the reward changes in #35.

## Objects and sources

| Object | Code and definition | Source |
| --- | --- | --- |
| Motion command | `commands.motion_cases()`: eight bearings at 0.025 and 0.05 m/s, yaw ±0.2 rad/s, arcs (0.04, 0, ±0.15). | Existing `locomotion.task.command_bank(TaskConfig())`; these 20 commands are project coverage choices. |
| Cycle | `optimize.Config`: 1.2 s, 0.65 stance fraction, 0.018 m swing lift, two alternating tripods. | Retained optimizer defaults; [accepted forward experiment](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/solve_smooth_001/INPUT.json). Use smoothing weight 0.1 and root-velocity weight 1 from that experiment. |
| Mechanical state | `RobotModel`, `q[61,24]`, `v[61,24]`, `a[60,24]`: six floating-root coordinates and 18 joint coordinates. `force[60,18]`, `torque[60,18]`, `target[60,18]` complete the collocation variables. | Existing approved 19-body URDF, inverse dynamics and 50 Hz optimizer. The 61 nodes here count time samples, separate from AMP feature width. |
| Turn geometry | `planar_pose()` and `cycle_state()` integrate a constant planar body command and close a cycle through its rigid transform. | Project derivation below; the paper does not specify this optimizer implementation. |
| Native record | `replay_native.py`: 1,000 controls, 8,000 physics samples, video and force/torque metrics. | Existing `omni_static` screen and `evaluate.run_batch`; unchanged gates. |
| Feature map | `locomotion.amp.extract_features()`: raw 61-value state shared with `LocomotionEnv._observations()`. | Existing kernel layout, retained by James. [Liu §III-A](https://arxiv.org/html/2511.03167v1#S3.SS1) names 61 values but describes foot heights. Six scalar heights would produce 49 values; this project uses six XYZ toe positions. |
| Transition bank | `dataset.export()`, `load()`, `sample()`: native `(s_t, s_next)` pairs, concatenated to 122 values for a learner. | Consecutive-state discriminator input in [Liu §III-B and Table III](https://arxiv.org/html/2511.03167v1); export schema and sampler are project decisions. |

The optimizer retains full-body force balance, midpoint integration and its
inverse PD target relation. It preserves the joint bounds, 1.6 N·m torque cap,
0.35 rad target envelope and 0.040 rad cyclic target slew. `audit()` uses the
existing 2e-5 feasibility tolerance. Native replay applies the approved motor
curve at 400 Hz; CPU point contacts cannot establish mesh-contact behavior.

For command `(forward, left, yaw)`, let `u = (left, -forward, 0)` and
`J u = (-u_y, u_x, 0)`. In native axes, integrate
`R(t) = Rz(yaw*t)` and `p(t) = integral_0^t R(s) u ds`.
`planar_pose()` evaluates this integral with its zero-yaw limit. At period `T`,
close root position through `r(T) = R(T) r(0) + p(T)`, add `yaw*T` to root yaw,
and rotate world linear velocity through `v(T) = R(T) v(0)`. Joint positions,
joint velocities and Euler rates repeat. Stance feet stay at fixed world
anchors. This permits repeating body-relative motor targets through a turn.
Cycle closure also fixes terminal foot positions from the initial contacts.
The solver omits those 18 duplicate terminal equalities. At the tested pose,
the endpoint/contact Jacobian has 60 rows with rank 42 before this reduction,
and 42 rows with rank 42 after it. The tests verify the rigid-transform identity and target closure;
the feasible motions and audit limits stay the same.
For turning commands, the new yaw-rate objective uses
`root_velocity_weight * ((yaw_rate - commanded_yaw) / 0.2)^2`.
The 0.2 rad/s scale comes from the existing yaw command envelope. Root
orientation cost and its retained ±0.15 rad bound use error from the nominal
turn. These objective choices are project adaptations, separate from the
paper's unspecified trajectory-optimization objective.

The paper reports 8.6 s motion examples. This implementation retains the
project's 1.2 s cycle and 20 s native test window. Treat these choices as
adaptations. No current result establishes a benefit from AMP learning.

## Retained feature contract

You use native absolute joint angles and native joint order from
`env_config.JOINT_NAMES`: `lf, lm, lr, rf, rm, rr`, with coxa yaw, femur pitch
and tibia pitch per leg. Body axes are +X left, -Y forward and +Z up.

| Slice | Values | Units and frame |
| --- | --- | --- |
| 0:18 | Joint position | rad, absolute native joint angles |
| 18:36 | Joint velocity | rad/s, native joints |
| 36:39 | Root-origin linear velocity | m/s, body frame; subtract the COM offset velocity |
| 39:42 | Angular velocity | rad/s, body frame |
| 42:43 | Root height | m, world Z above the flat zero plane |
| 43:61 | Six toe XYZ positions | m, body frame relative to the root origin |

The raw contract has no command, phase or normalization. Adjacent states span
0.02 s. The native recorder retains the COM offset so the offline audit can
rebuild root-origin velocity from COM velocity and angular velocity. A parity
error above 2e-5 rejects a clip; this is a project float32 reconstruction check.
Learner normalization belongs downstream and must preserve this raw contract.

## Generate and evaluate

```sh
uv run --locked python -m locomotion.priors.optimize --bank \
  --output /absolute/fresh/optimization \
  --target-curvature-weight 0.1 --root-velocity-weight 1
uv run --locked python -m locomotion.priors.prepare --help
uv run --locked python -m locomotion.priors.dataset --help
```

You give each attempt a fresh directory. `coverage.json` retains solved and
failed CPU cases. `INPUT.json`, `RESULT.json` and `SHA256.json` bind source,
model and trajectory bytes. Preserve failed attempts before changing a candidate.
To reuse a finite failed iterate, pass `--initial-trajectory /absolute/attempt/trajectory.npz`
to a single-command solve with the same model and problem settings. You can change
the iteration budget. The optimizer copies the seed and its input/result records
into the new attempt and records their hashes. It initializes the primal variables
through [CasADi `Opti.set_initial`](https://github.com/casadi/casadi/blob/main/docs/users_guide/source/opti.rst).
The solver tolerances and feasibility gates stay the same; the seed grants no admission.
You can use `--mu-strategy adaptive` with a restart to select
[Ipopt's adaptive barrier update](https://github.com/coin-or/Ipopt/blob/stable/3.14/doc/special.dox).
The default uses its monotone update. Each new input record lists the solver
options; the objective, physical constraints and convergence tolerances stay fixed.

Follow [OPERATIONS](../../docs/OPERATIONS.md) for same-source one-robot and
32-robot standing admission, input packing and the guarded native launcher.
Prepare each solved command with `--start-phase 0` and `--start-phase 0.5`.
The second phase tests another cold start; it supplies no extra training rows.
These two phases are a project stress test. They do not represent independent
random seeds. Match the same physics and gait configuration across the bank.
If a direct start fails, you can specify `--startup-ramp-controls 50` for both
phases of that command. This project startup procedure blends from the admitted
neutral target for one second through `(1-cos(pi*t/1s))/2`. It uses the cosine
blend form from the [classical startup procedure](../README.md), while the
optimized target cycle keeps its phase clock. The native kernel still applies
the 0.040 rad limiter. The recorder stores the ramp length; the exporter requires
matching startup settings within each command's phase pair. The fixed two-second
crop excludes the ramp. Full-trial motor and contact checks still include it.

Pass the 40 completed native run directories to `dataset inspect` through
repeated `--replay` arguments and a fresh `--output`. Inspection recomputes
tracking and force metrics, checks file hashes, rejects reset/terminal pairs,
and reconstructs features from native telemetry. It writes `audit.json` and
an unsigned `review.json` template. It reports missing commands and phases.

James reviews each video with achieved forward/left/yaw motion and six-foot
contact changes. He records `direction_accepted` and `tripod_accepted`, bound to
report and video hashes, and sets `reviewer` to `palerdr`. The low-speed static
screen can admit standing within its 0.025 m/s error bound, so a numeric pass
alone cannot establish the requested gait. Force metrics describe loads and
add no new acceptance limits.

After James accepts the clips, use `dataset export --review ...` with the same
replays and a fresh output directory. The exporter retains controls 100:1000
from phase zero, matching the existing screen's 2 s settling exclusion. It
keeps the complete trial for acceptance. Each command contributes 900 pairs:
18,000 pairs over 20 commands. It never joins clip endpoints. `load()` verifies
the manifest and file hash; `sample(arrays, batch_size, rng)` returns `[B,122]`
raw pairs with equal command weights. Half-phase clips remain audit evidence.
The audit and dataset manifests record processing-code hashes and the NumPy
version, in addition to each clip's native and optimizer source identities.

You can test the pipeline before native admission:

```sh
uv run --locked python -m unittest discover -s locomotion/priors/tests
uv run --locked python -m unittest locomotion.tests.test_amp
```

Tests check cycle geometry, retained forward feasibility and native feature
parity. Synthetic dataset tests exercise rejection and export; they supply no
motion evidence. Native admission and James's motion review remain required.
The classical tripod remains a comparison baseline. This bank uses optimized
motions alone. Downstream five-seed comparisons belong to #37 and do not block
construction or inspection of this dataset.
