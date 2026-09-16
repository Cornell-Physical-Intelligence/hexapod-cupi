# Forward trajectory optimizer

You can generate and test a walking reference without the physical robot. This
experiment solves one forward tripod cycle for the approved 19-body, 18-joint
model, then applies the resulting motor targets in the admitted Isaac setup.
The first replay moves forward but fails the existing speed-tracking limit.

## Method and scope

[Liu et al., Section III-A](https://arxiv.org/html/2511.03167v1#S3.SS1)
use optimized motion examples to guide reinforcement learning. This prototype
implements that reference-generation idea for our robot. It covers a single
forward command on a flat floor; it does not reproduce the paper's training or
hardware results.

`model.py` reads the approved mass and inertia data. It constructs floating-base
Newton-Euler inverse dynamics for all links. The optimizer uses a fixed tripod
contact schedule, point contacts and midpoint collocation. CasADi supplies exact
derivatives to IPOPT. The initial inverse-kinematics path seeds the solve.

`optimize.py` solves a 1.2-second cycle at 0.05 m/s with 60 motor commands. It
constrains force balance and friction, joint travel and speed, torque below
1.6 N·m, and cyclic target changes below 0.040 rad per 20 ms. It converts the
computed torque into position targets through the unchanged PD law. It saves
source copies and audits the stored arrays. Point contacts omit the mesh contact
patches and impacts; the native replay measures these effects. Midpoint
feasibility does not establish continuous-time feasibility.

`prepare.py` creates a fresh source pack and launch binding. It retains the
admitted environment and evaluator bytes, including the guard and supervisor.
`replay_native.py` applies the cycle through the 400 Hz motor controller, with
50 Hz held targets and the existing startup slew limiter. It records actual
state transitions and a native video. It performs no PPO update or pose forcing
after the initial reset.

## Recorded result

The [CPU solve](../../artifacts/trajectory_optimizer_20260917/solve_002/RESULT.json)
converges after 154 iterations. Peak planned torque is 1.29064 N·m; the largest
cyclic target change is 0.0223182 rad. The
[native audit](../../artifacts/trajectory_optimizer_20260917/review_001/RESULT.json)
verifies 35 transferred files and 8,000 physics steps. It reclassifies 131,634
contact patches and reproduces the original gate checks.

| Measurement | Native result |
| --- | ---: |
| Full-trial forward world displacement | 0.967614 m |
| Mean forward velocity after settling | 0.0494640 m/s |
| Planar tracking error | 0.0332708 m/s; fails 0.025 m/s limit |
| Forward velocity range after settling | −0.0141956 to 0.108552 m/s |
| Peak computed motor demand | 1.401391 N·m |
| Tilt RMS after settling | 0.126689° |
| Nonfoot contact samples | 0 / 8,000 |

The full replay completes 1,000 controls without a reset or termination. Native
motor, joint and contact checks pass. Startup uses the unchanged 0.040 rad
limiter; the optimizer's smaller cyclic limit applies once playback reaches the
cycle. The sampled video shows repeated foot lifts and floor-grid translation.
The [visual review](../../artifacts/trajectory_optimizer_20260917/review_001/VISUAL_REVIEW.json)
records its sampling limits. James has not accepted this gait.

You can inspect the
[20-second video](../../artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation/rollout.mp4)
and the complete 1,000-row `recorded_transitions.npz`. The replay marks those
rows ineligible for the accepted motion-prior dataset because the tracking gate
fails. Omnidirectional motion, stops and terrain remain outside this experiment.
The result supports continued reference work; it establishes no robust learned
policy or Stage 2 acceptance.

## Reproduce

```sh
uv sync --locked
uv run python -m unittest discover -s experiments/trajectory_optimization/tests
uv run python -m experiments.trajectory_optimization.optimize --output tmp/trajectory-reproduction
uv run python -m experiments.trajectory_optimization.prepare --trajectory-directory tmp/trajectory-reproduction --output tmp/trajectory-replay-pack --remote-root /home/orionh/HEXAPOD_runs/restart_20260914/trajectory_reproduction
```

Both commands require fresh output directories. Before a native allocation,
follow [OPERATIONS](../../docs/OPERATIONS.md) and recheck reservation ownership.
The generated binding pins the exact source and input hashes. Preserve the
original admission mounts and dispatch through the retained `launch_spark.py`.
Do not reuse the completed allocation's output path. The container uses the
existing Isaac runtime; CasADi runs on the CPU host that prepares the cycle.

The [attempt inventory](../../artifacts/trajectory_optimizer_20260917/ATTEMPTS.json)
retains the first solve's reporting failure and the undispatched first pack.
The directory suffix `20260917` is an allocation-label error; dispatch and
terminal records place the experiment on 16 September 2026 UTC. The source
snapshots bind the executed versions. Later audit additions do not change the
saved solver or replay files.

## Force and motor-load reports

Future native replays write `force_metrics.json` through the replay wrapper.
The wrapper preserves the original evaluator and adds a hash-bound sidecar.
The report separates startup from commanded locomotion and includes contact
loads in N and motor torque in N·m. Check that the report is available before the
next experiment in the optimizer/PPO sequence. The
[load-tracking reference](../../docs/TRAINING.md#locomotion-load-tracking)
defines the averages and the normal-force-only measurement scope.
The [derived first-replay report](../../artifacts/locomotion_force_metrics_20260916/forward_replay_001.json)
preserves the original capture and its failed tracking verdict.
