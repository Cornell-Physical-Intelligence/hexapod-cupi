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

## Improved reference and standard PPO comparison

The [improved native reference](../../artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
passes the forward screen at 0.05 m/s. Planar error decreases to 0.003029 m/s
from 0.033271 m/s. The optimizer uses `--root-velocity-weight 1` and
`--target-curvature-weight 0.1`; the original constraints remain unchanged.
The target-curvature statistic increases, while speed tracking improves.
See [TRAINING](../../docs/TRAINING.md#standard-ppo-and-improved-forward-reference-17-september-2026)
for the motor-load tradeoff and the result's scope.

`vanilla.py` adapts the admitted simulator to RSL-RL 5.0.1. `vanilla_native.py`
trains standard PPO from scratch, without imitation or a motion-prior reward.
It uses the existing command bank and task reward. The critic receives measured
base velocity; the actor receives the existing proprioceptive history. The
adapter resets selected terminal replicas and preserves their terminal rewards.

`prepare_vanilla.py` freezes a new source pack and guarded binding. Training
uses the admitted 128-replica layout. A policy evaluation requires its checkpoint
hash and matching declaration, including learner source and configuration.
Training load summaries cover 400 Hz body forces and native applied torque.
The separate evaluation uses the exact contact-patch classifier and records video.

```sh
uv run python -m experiments.trajectory_optimization.prepare_vanilla --output tmp/vanilla-pack --remote-root /home/orionh/HEXAPOD_runs/restart_20260914/vanilla_reproduction --updates 1200 --seed 20260914
```

Run the host preflight with `admission_host.json`; the container binding retains
`admission.json` and its original mounts. A prepared pack starts no compute.
Dispatch through the existing guard after checking the retained reservation.

The improved replay's contact log has a lossless public gzip copy. The original
log remains on Spark and in the local result directory. After cloning the public
record, restore that log before running the saved audit:

```sh
gzip -dc artifacts/ppo_reference_comparison_20260917/replay_pack_001/contacts.jsonl.gz > artifacts/ppo_reference_comparison_20260917/replay_pack_001/replay_001/standing/evaluation/native400hz/contacts.jsonl
mkdir artifacts/ppo_reference_comparison_20260917/review_reproduction_001
cp artifacts/ppo_reference_comparison_20260917/review_smooth_001/verify.py artifacts/ppo_reference_comparison_20260917/review_reproduction_001/verify.py
uv run python artifacts/ppo_reference_comparison_20260917/review_reproduction_001/verify.py
```

The audit verifies the original hash. Its result writer refuses to overwrite
the preserved `RESULT.json`; use a fresh result directory for a new audit record.

The [completed PPO comparison](../../artifacts/ppo_reference_comparison_20260917/COMPARISON_001.json)
records 1,200 standard PPO updates and 3,686,400 transitions. The final policy
fails forward walking, quiet standing and stopping. Its forward error is
0.051091 m/s, compared with 0.003029 m/s for the improved reference. The
[comparison figure](../../artifacts/ppo_reference_comparison_20260917/review_vanilla_001/forward_comparison.png)
shows the complete trials. The result supports using the improved reference as
a motion example; it establishes no benefit from reference-assisted PPO, which
has not run.

The final checkpoint and the three policy videos have SHA-256 identities in the
comparison and native audit. Training keeps its 48 intermediate checkpoint
files on Spark; `vanilla_train_001/REMOTE_INVENTORY.json` records their hashes.
The local and Spark copies retain the expanded metric and contact logs. Public
gzip copies preserve those bytes while reducing the repository payload. Restore
them before running the saved full audit:

```sh
gzip -dc artifacts/ppo_reference_comparison_20260917/vanilla_train_001/metrics.jsonl.gz > artifacts/ppo_reference_comparison_20260917/vanilla_train_001/run/standing/metrics.jsonl
gzip -dc artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/evaluation_00.contacts.jsonl.gz > artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/run/standing/evaluation_00/native400hz/contacts.jsonl
gzip -dc artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/evaluation_01.contacts.jsonl.gz > artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/run/standing/evaluation_01/native400hz/contacts.jsonl
gzip -dc artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/evaluation_02.contacts.jsonl.gz > artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/run/standing/evaluation_02/native400hz/contacts.jsonl
mkdir artifacts/ppo_reference_comparison_20260917/review_ppo_reproduction_001
cp artifacts/ppo_reference_comparison_20260917/review_vanilla_001/verify.py artifacts/ppo_reference_comparison_20260917/review_ppo_reproduction_001/verify.py
uv run python -B artifacts/ppo_reference_comparison_20260917/review_ppo_reproduction_001/verify.py
```

The training native process exits with code 0, but a competing COLMAP container
blocks the launch wrapper's final resource check. Its original failed exit,
recovery image and successful cleanup remain recorded in OPERATIONS. The policy
evaluation service completes with exit code 0. Neither receipt releases the
Spark reservation or changes a behavior gate.

## Controlled forward-example PPO pilot

The [protocol](../../artifacts/forward_example_ppo_20260917/protocol_001/PROTOCOL.json)
compares a fresh actor with an actor that first copies the passing reference's
recorded actions. Both arms receive 1,200 PPO updates, with 128 replicas and 24
controls per update. They use seed 20260914 and a fixed 0.05 m/s forward command.
The prior mixed-command baseline is outside this paired comparison.

Both arms initialize their observation normalization from the first 700 recorded
rows. The example arm fits actor MLP weights to those actions for 1,000 Adam
steps. It keeps the critic and action variance unchanged, then discards the
imitation optimizer and restores the PPO random state. PPO starts with an empty
optimizer in both arms. It uses the existing task reward without an imitation
loss. Report the extra imitation work apart from the equal PPO sample budgets.

Evaluate checkpoints at updates 0, 300, 600 and 1,200. Use the unchanged forward
screen as the primary outcome and record quiet/stop probes as diagnostics.
Record force and torque at 400 Hz, plus a native policy video. A pass before
PPO establishes a copied policy; later passes test its retention during PPO.
This one-seed pilot cannot establish omnidirectional or statistical robustness.
The final 300 demonstration rows check action prediction on later cycles of the
same recording; they are not an independent locomotion trial.

The [CPU check](../../artifacts/forward_example_ppo_20260917/CPU_VALIDATION_003.json)
verifies actor-only fitting, preserved PPO random state, checkpoint zero and
40 optimizer steps per parameter in a two-update interface test. That test uses
recorded observations without physics and makes no walking claim.

To reproduce preparation, choose new local and remote directories:

```sh
uv run --with rsl-rl-lib==5.0.1 python -m unittest experiments.trajectory_optimization.tests.test_forward_experiment
uv run python -m experiments.trajectory_optimization.prepare_forward protocol --output tmp/forward-reproduction/protocol
uv run python -m experiments.trajectory_optimization.prepare_forward pack --protocol tmp/forward-reproduction/protocol/PROTOCOL.json --output tmp/forward-reproduction/scratch --remote-root /home/orionh/HEXAPOD_runs/restart_20260914/forward_reproduction/scratch --arm scratch
uv run python -m experiments.trajectory_optimization.prepare_forward pack --protocol tmp/forward-reproduction/protocol/PROTOCOL.json --output tmp/forward-reproduction/example --remote-root /home/orionh/HEXAPOD_runs/restart_20260914/forward_reproduction/example --arm example
```

Use `--smoke` for a separate two-update validation allocation. For evaluation,
use `pack --mode evaluate` with the exact checkpoint path, its SHA-256 and its
JSON sidecar SHA-256. The package retains the same arm and protocol identity.
Follow OPERATIONS for review, live resource checks and guarded dispatch. The
preparer creates launch inputs; it does not launch jobs or grant admission.
