# Physical MKII training campaign

This campaign implements the user-authorized restart on the physical linkage,
with scratch policies and explicit provisional motor dynamics. Live outcomes
are recorded separately; source implementation alone does not admit training.

## What runs

- Task: `Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0`.
- Asset: `robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda`.
- Mechanism: 31 bodies, 30 articulation coordinates, six excluded revolute loop
  closures, and 18 active motors. Knees and rods have no actuator or mimic.
- Actions: six coxa, six femur, then six pushlever motor coordinates, resolved
  from imported names. Actual articulation order remains independent.
- Reset: all 30 coordinates derive from the 18 motor positions; per-leg CAD
  phase offsets are explicit. Plate reset height is 0.142970 m. No independent
  passive jitter. Initial qualification uses zero joint jitter.
- Control: 200 Hz explicit motor dynamics, 50 Hz policy; 84 observations include
  18 motor overload-headroom states. Anatomical forward is body -Y, left +X.
- Baseline commands: ±0.15 m/s on both horizontal axes and ±0.30 rad/s yaw, with
  20% standing commands. Actions have ±0.30 rad offsets and a 0.04 rad/20 ms
  target slew limit. These are initial simulation settings, not hardware limits.

The dedicated environment has velocity tracking, orientation/height, actuation,
slip/contact, action-rate and limit objectives. There is no gait clock, tripod
phase, scripted swing path, or old checkpoint adoption. Full-range workspace and
self-collision qualification remain separate; self-collision is initially off.

## RS05 v2

`packages/hexapod_core/hexapod_core/rs05_v2.json` pins vendor source hashes and
distinguishes the 48 V torque-speed curve, 1.2 N·m continuous stall, 1.6 N·m at
100 rpm with the 70 mm aluminum reference plate, and finite overload durations.
Applied effort must satisfy both speed/voltage and remaining overload budget.
Raw PD demand, actual clipping, continuous excess, delivered overload exposure
and current proxy are separate diagnostics.

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

## Reproduction and compute ownership

On Spark, isolated source is
`/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`; outputs are in sibling `runs/`.
The original project mirror and archived task/checkpoint contracts are retained.

```sh
python3 isaaclab/deploy/run-mkii-fourbar validate \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source \
  --num-envs 32 --steps 1000 --solver-multiplier 1 --timeout-seconds 900

# Repeat with --solver-multiplier 2, then qualify the two exact report paths.
python3 tools/qualify_mkii_fourbar.py NOMINAL_REPORT REFINED_REPORT ADMISSION_JSON

python3 isaaclab/deploy/run-mkii-fourbar train \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source \
  --admission ADMISSION_JSON --num-envs 64 --iterations 10 --timeout-seconds 900

# A later process may add --checkpoint PATH_TO_CHECKPOINT_PT.
# Its adjacent .pt.json must match both bytes and the current complete contract.
```

The host supervisor holds the existing GPU lock, waits at a CPU admission barrier,
requires at least 16 GiB available host memory, rejects unrelated GPU workloads,
and checks source identity before and after execution. Cleanup addresses only the
run's immutable container ID and ownership label. A zero exit without the required
report and verified checkpoint is a failure.

Use full available compute until sharing is requested in
`/home/orionh/SPARK_COMPUTE_COORDINATION.md`. Change its single
`HEXAPOD_SHARE_STATUS=NONE` line to `HEXAPOD_SHARE_STATUS=REQUESTED` and describe
the workload. A file change during learning requests a checkpoint and pause at
the next PPO iteration; an unresponsive owned job is stopped after 120 seconds.
MPS and a 60/40 allocation are not installed. See [the shared note](SPARK_COMPUTE_COORDINATION.md).

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

The full supervisor's 21,600-second ceiling is a hard timeout, counted from host preparation, not a graceful checkpoint request. Periodic checkpoints are written every ten iterations. If observed throughput predicts exceeding the ceiling, request a cooperative pause before it by writing the owned run's `stop_requested` marker; preserve its verified checkpoint and report before explicitly resuming. A timeout or exit zero without the required report is not a passed run.

## TGS recipe revision

The original 32/4 and 64/8 recipes completed every driven step but failed the unchanged closure bounds (0.246159 mm and 0.366053 mm pin separation respectively). The new recipe explicitly selects TGS, applies external forces on every iteration, uses one velocity iteration, and compares 64 against 128 position iterations at the same 5 ms physics timestep. This revision requires new complete validation reports; old source-identity reports cannot admit it. The previous manifests, reports and source commits are preserved.

## 800 Hz physics revision

At 5 ms, the explicit-force 64/1 recipe failed the first tibia reversal after 1,252 driven steps, reaching 0.883917 mm physical pin separation and 0.0571375 rad passive residual before closure termination. Applied torque remained inside its instantaneous envelope. The follow-up uses **1.25 ms physics, sixteen substeps and unchanged 20 ms policy control**. Motor budget/overload/recovery integration is tested over equal physical durations at 5, 2.5 and 1.25 ms. Every physical substep remains measured; the full 1,000 standing plus 2,400 driven-control-step sequence and its bounds remain unchanged.

A reduced linearized inertia analysis motivates the timestep: ideal closed-loop modes and light unconstrained lever modes have different timestep margins. It does not replace the actual constrained/contact simulation checks. The new recipe needs fresh nominal/refined validation, complete checkpoint tests and a new training source identity. Training has a six-hour maximum to accommodate the increased integration work; actual throughput must be measured before quoting an ETA.
