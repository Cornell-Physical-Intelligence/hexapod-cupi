# Physical MKII training campaign

This campaign implements the user-authorized restart on the physical linkage,
with scratch policies and explicit provisional motor dynamics. Live outcomes
are recorded separately; source implementation alone does not admit training.

See [STATUS.md](../STATUS.md) for the current execution state and
[campaign evidence](../artifacts/mkii_fourbar_2026-09-05/) for immutable results.
A complete nominal pass alone cannot admit training: the refined repeat and
solver-convergence comparison must also pass.

## What runs

- Task: `Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0`.
- Mechanism: 31 bodies, 30 articulation coordinates and 18 active motors in
  every candidate. Knee and rod coordinates have no independent actuator.
- Actions: six coxa, six femur, then six pushlever motor coordinates, resolved
  from imported names. Actual articulation order remains independent.
- Reset: all 30 coordinates derive from the 18 motor positions; per-leg CAD
  phase offsets are explicit. Plate reset height is 0.142970 m. No independent
  passive jitter. Initial qualification uses zero joint jitter.
- Control: 800 Hz explicit motor dynamics (1.25 ms), sixteen physics substeps
  per 50 Hz policy update; 84 observations include 18 motor overload-headroom
  states. Anatomical forward is body -Y, left +X.
- Solver: TGS with external forces applied every position iteration; nominal
  64/1 position/velocity iterations, refined 128/1 at the same timestep.
- Baseline commands: ±0.15 m/s on both horizontal axes and ±0.30 rad/s yaw, with
  20% standing commands. Actions have ±0.30 rad offsets and a 0.04 rad/20 ms
  target slew limit. These are initial simulation settings, not hardware limits.

The dedicated environment has velocity tracking, orientation/height, actuation,
slip/contact, action-rate and limit objectives. There is no gait clock, tripod
phase, scripted swing path, or old checkpoint adoption. Full-range workspace and
self-collision qualification remain separate; self-collision is initially off.

Select the intended physical formulation explicitly; the CLI default remains
`mkii_fourbar_v3` for compatibility and does not imply that v3 passed admission.
The three registered bundles live under
`robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_vN/`:

| `--asset-model` | Physical loop formulation |
| --- | --- |
| `mkii_fourbar_v3` | Six excluded revolute closure joints, five constraint rows per closure |
| `mkii_fourbar_v4` | Six excluded planar D6 closures, two in-plane constraint rows per closure |
| `mkii_fourbar_v5` | Twelve native PhysX bilateral mimic constraints inside the articulation, zero external closures |

The v5 constraints transmit forces through the simulated mechanism and enforce
the CAD parallelogram coordinate relations; they are not a prescribed gait or
per-step passive pose overwrite. Reset still derives all passive coordinates
from the motor coordinates. The earlier URDF mimic variant used for visual
checks is distinct from this versioned physical USD formulation.

## RS05 v2

`packages/hexapod_core/hexapod_core/rs05_v2.json` pins vendor source hashes and
distinguishes the 48 V torque-speed curve, 1.2 N·m continuous stall, 1.6 N·m at
100 rpm with the 70 mm aluminum reference plate, and finite overload durations.
Applied effort must satisfy both speed/voltage and remaining overload budget.
Raw PD demand, actual clipping, continuous excess, delivered overload exposure
and current proxy are separate diagnostics.

The physical task explicitly selects controller profile
`mkii_pd_damping_030_v1` (Kp 30 N·m/rad, Kd 0.30 N·m·s/rad). The baseline
RS05 JSON and default factory profile remain Kp 30/Kd 0.60. The instantiated
profile, actual parameters and implementation hashes are bound into each
runtime manifest, and the environment rejects a different task profile.
This experiment reduces amplification of simulated velocity noise while
retaining static proportional stiffness. It may also reduce damping of real
motions; only live stability/convergence checks can establish whether it helps.
No hardware tuning is inferred from this simulation setting.

This is a provisional engineering model. The metal heat path is confirmed as a
design intention; its thermal resistance/capacity and repeated-burst recovery
are unmeasured. The initial half-full overload budget and 60-second low-load
recovery are labeled assumptions. Above 100 rpm, the continuous bound uses a
constant mechanical-power cap rather than assuming the rotating rating remains
continuous at all speeds. A symmetric absolute-speed braking bound does not
claim a measured four-quadrant motor map. Battery voltage is still unknown;
48 V is a declared simulation reference. The model is not a temperature estimate.

## Admission sequence

1. Audit all mass/inertia properties, 1,927 visual instances, 77 source meshes,
   171 primitives, joint frames and portable dependencies. Run the checker in
   a separate standalone-USD process, then again in Kit's native USD runtime.
2. Run a small import/standing probe; preserve failures and exact source hashes.
3. Run 32 environments for 1,000 standing control steps, then 2,400 driven steps:
   each motor independently at ±0.04 rad, followed by simultaneous group tests.
   This checks individual mapping and the physical mechanism under small loads.
4. Repeat with 128/1 position/velocity solver iterations versus the nominal 64/1.
   Compare standing and driven height/torque; both runs must pass independently.
   The same selected model, dependency hashes, motor contract and runtime settings
   must match; only the specified solver position-iteration count may differ.
5. Combine those reports with `tools/qualify_mkii_fourbar.py`. The resulting
   admission is restricted to monitored exploratory simulation learning.
6. Run a small PPO cycle, save and reload actor, critic, optimizer and observation
   normalization, perform finite inference, then resume in a new process.
7. Profile before increasing environments/iterations. Screen directional tracking
   and stops before rough-terrain, randomization or navigation experiments.

During validation, every physics substep measures actual link-pose pin closure,
hinge-axis agreement, all primitive ground-clearance bounds, all 31 body contact
streams, passive-coordinate residual and delivered motor bounds. Initial numeric
limits are 0.10 mm pin separation and 0.10° axis agreement; these are simulator
criteria, not measured manufacturing tolerances. Primitive support checks avoid
relying on a tibia's aggregate contact centroid to distinguish pad and shaft.

The first PPO runs retain continuous closure and actuator checks before episode
resets. Numerical-model failures invalidate a run; ordinary falls remain learning
outcomes. This monitoring does not establish physical accuracy or terrain skill.
Before constructing its wrapper or learner, training requires exact equality
between the admitted nominal runtime manifest and the loaded environment. This
includes the selected bundle and dependency hashes, motor parameters, solver
settings and observed joint order. The comparison is saved in the training report.

## Reproduction and compute ownership

On Spark, isolated source is
`/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`; outputs are in sibling `runs/`.
The original project mirror and archived task/checkpoint contracts are retained.

```sh
python3 isaaclab/deploy/run-mkii-fourbar validate \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source \
  --asset-model mkii_fourbar_v5 \
  --num-envs 32 --steps 1000 --solver-multiplier 1 --timeout-seconds 3600

# Repeat the same model with --solver-multiplier 2.
# Only two complete passing reports can produce admission.
python3 tools/qualify_mkii_fourbar.py NOMINAL_REPORT REFINED_REPORT ADMISSION_JSON

# This remains blocked until the admission gate above passes.
python3 isaaclab/deploy/run-mkii-fourbar train \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source \
  --asset-model mkii_fourbar_v5 \
  --admission ADMISSION_JSON --num-envs 64 --iterations 3 --timeout-seconds 3600

# A later process may add --checkpoint PATH_TO_CHECKPOINT_PT.
# Its adjacent .pt.json must match both bytes and the current complete contract.
```

Include `--source-commit` with the isolated checkout's baseline commit when
launching. Each run archives the actual source bytes and SHA-256 manifest.
The host supplies canonical in-container USD paths for the selected model;
arbitrary host environment overrides cannot select another bundle.

For diagnosis, use the separate selector and report type:

```sh
python3 isaaclab/deploy/run-mkii-fourbar diagnose \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source \
  --diagnostic-usd physical_mimic_v5 --diagnostic-motion groups \
  --num-envs 8 --steps 200 --solver-multiplier 2 --timeout-seconds 3600
```

`--asset-model` is rejected in diagnostic mode. A group diagnostic includes 600
driven plus 100 zero-action recovery steps after standing. Its trace must cover
every physics substep, with matching force-write/sample counts and archived NPZ
hashes. A completed diagnostic reports `diagnostic_complete`, `pass=false` and
both admission flags false, even if its measured physical bounds pass.

The host supervisor holds the existing GPU lock, waits at a CPU admission barrier,
requires at least 16 GiB available host memory, rejects unrelated GPU workloads,
and checks source identity before and after execution. Cleanup addresses only the
run's immutable container ID and ownership label. A zero exit without the required
report and verified checkpoint is a failure.
Validation and diagnostics have a 7,200-second maximum; training has a
21,600-second maximum. The campaign defaults to 3,600 seconds for each short
phase and 21,600 seconds for the full PPO phase. These are execution ceilings,
not predicted completion times.

Use full available compute until sharing is requested in
`/home/orionh/SPARK_COMPUTE_COORDINATION.md`. Change its single
`HEXAPOD_SHARE_STATUS=NONE` line to `HEXAPOD_SHARE_STATUS=REQUESTED` and describe
the workload. A file change during learning requests a checkpoint and pause at
the next PPO iteration; an unresponsive owned job is stopped after 120 seconds.
MPS and a 60/40 allocation are not installed. See [the shared note](SPARK_COMPUTE_COORDINATION.md).
An unrelated producer remains a veto. A real `flock` process queued on
`/opt/wx/gpu.lock` is exempt only when kernel lock, descriptor, executable and
child-process evidence proves it is blocked; GPU PIDs still veto independently.

## Following work

The first learned checkpoint is a software/dynamics integration baseline. Before
calling it an omnidirectional gait, evaluate signed forward/lateral/yaw commands,
diagonals, starts/stops and reversals with trajectory error, support/slip, contact,
motor duty and failure counts. Then qualify workspace/self-collision and progressively
expand terrain and uncertainty. Match simulator motor/friction/latency parameters
to the vertical leg stand as measurements arrive. Camera/LiDAR/IMU navigation and
polygon coverage remain parallel software work, independent of survey payloads.

## Full first-stage campaign

`isaaclab/deploy/run-mkii-fourbar-campaign` runs the short probe, both nominal/refined standing and driven checks, qualification, 64-environment / 3-iteration scratch PPO, and then a separate 512-environment / 1,000-iteration resumed PPO process. The latter produces 12,288,000 additional transitions. Each phase stops the sequence on failure; the full run does not inherit the old mock policy. Its separate process must restore the exact scratch policy and optimizer state.

Pass `--asset-model mkii_fourbar_v5` explicitly when investigating v5. The
campaign records and forwards that choice through every phase, including reused
probe checks, scratch training and resumed training. A failed nominal run stops
before refined qualification or PPO; the current v5 failure is preserved.

The full supervisor's 21,600-second ceiling is a hard timeout, counted from host preparation, not a graceful checkpoint request. Periodic checkpoints are written every ten iterations. If observed throughput predicts exceeding the ceiling, request a cooperative pause before it by writing the owned run's `stop_requested` marker; preserve its verified checkpoint and report before explicitly resuming. A timeout or exit zero without the required report is not a passed run.

## TGS recipe revision

The original 32/4 and 64/8 recipes completed every driven step but failed the unchanged closure bounds (0.246159 mm and 0.366053 mm pin separation respectively). The subsequent 5 ms recipe selected TGS, external forces on every position iteration, and nominal/refined counts of 64/1 and 128/1. That timestep is historical and has been superseded by the 1.25 ms recipe below. Previous manifests, reports and source commits remain preserved and cannot admit a changed runtime.

## 800 Hz physics revision

At 5 ms, the explicit-force 64/1 recipe failed the first tibia reversal after 1,252 driven steps, reaching 0.883917 mm physical pin separation and 0.0571375 rad passive residual before closure termination. Applied torque remained inside its instantaneous envelope. The follow-up uses **1.25 ms physics, sixteen substeps and unchanged 20 ms policy control**. Motor budget/overload/recovery integration is tested over equal physical durations at 5, 2.5 and 1.25 ms. Every physical substep remains measured; the full 1,000 standing plus 2,400 driven-control-step sequence and its bounds remain unchanged.

A reduced linearized inertia analysis motivates the timestep: ideal closed-loop modes and light unconstrained lever modes have different timestep margins. It does not replace the actual constrained/contact simulation checks. Each physical formulation still needs passing nominal/refined validation, complete checkpoint tests and matching training source/runtime identities. No current formulation has that admission. Actual throughput must be measured before quoting a training ETA.

## Motor target interpolation candidate

The next source revision delivers each existing 50 Hz motor endpoint as sixteen linear position increments, with zero velocity feedforward. It preserves gains, limits, assets, validation motions and acceptance bounds; see [the scheduling contract](MKII_MOTOR_TARGET_SCHEDULING.md). It requires fresh validation and is not yet admitted. The existing hardware action adapter still emits endpoints; embedded scheduling and bus latency must be implemented and measured before transfer.

## Measuring throughput and preserving a long run

`progress.json` is written after every completed PPO update. Its
`iterations_completed` counts updates in that process; `last_iteration` is the
absolute zero-based RSL iteration. Measure elapsed wall-time differences over
several warmed updates. `collect_seconds` and `learn_seconds` describe the last
update, but omit some hashing, logging and checkpoint overhead. A 512-environment
update contains 12,288 transitions. The six-hour full-run ceiling requires less
than 21.6 seconds per update before setup and finalization; no measured full-run
ETA exists until this candidate actually learns.

If measured throughput will exceed the ceiling, create `stop_requested` only in
the exact owned training output directory, comfortably before its deadline.
The trainer checks it after a completed update, saves `checkpoint.pt` and its
SHA sidecar, verifies model/optimizer/normalizer/adaptive-learning-rate reload,
and exits with `pass=true`, `paused=true`. Verify the supervisor's successful
report, unchanged source and exact container removal. The campaign records
`state=paused` and exits 2; it intentionally does not resume automatically.

For three scratch updates followed by one thousand additional full updates,
the final target is `next_iteration=1003`. Resume using the same source, selected
asset and admission, the verified paused checkpoint, and
`--iterations 1003-minus-checkpoint-next-iteration` (calculate the positive integer
first). Use a new output directory and the guarded training launcher. A resumed
process restores learner state but starts fresh simulated episodes; this is not
continuous restoration of robot/terrain trajectories. Never edit the source or
reuse output paths to extend a run.
