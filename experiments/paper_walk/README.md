# Canonical paper-informed walking prototype

The active goal is to satisfy the existing Stage 2 requirements on the confirmed
mass-corrected robot, train PPO, and deliver video of the resulting policy running
in Isaac Sim. [ARCHITECTURE.md](../../ARCHITECTURE.md) owns that acceptance scope;
[the progress registry](../../site/project.json) owns measured progress. This
directory contains maintained experimental source, not evidence that the goal
has been achieved.

The robot stays bound to URDF SHA-256
`9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78`:
19 bodies, 18 direct-drive joints and nominal mass 7.466088235 kg. The native
environment retains explicit 400 Hz PD, 50 Hz policy control, the speed-dependent
1.6 N·m ceiling and 0.040 rad/control target limit. The declared walking neutral
`[0, -0.30, 0.40]` radians per leg passes the original standing gates on source002
at [one replica](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_one_002/standing/standing/standing_report.json)
and [all 32 batch replicas](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_batch32_001/standing/standing/standing_report.json).
[Independent recomputation](../../artifacts/restart_2026-09-14/paper_walk_execution_001/admission_001/verification.json)
matches both reports exactly. These 14 September standing passes apply to that
source, stance and native configuration; they do not change robot geometry,
revise the failed inspection-zero results or establish learned walking.

The subsequent [source003 native replay](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_replay_001/standing/native_replay/report.json)
accepted 31 of 32 complete demonstrations while retaining all 21 commands and
1,860 training transitions. The excluded duplicate recorded 4.896% worst-joint
saturation against the 2% replay screen; its failed evidence remains intact.
The [first source004 PPO run](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/standing/state.json)
completed 200 updates and 153,600 native transitions on 32 replicas without
behavior cloning. Its [checkpoint receipt](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/standing/learner/latest_checkpoint.json)
binds SHA-256
`5d48928dd4ba5fdfbd8ff6c6706a5eb8c3951aeaa9aacb8e38ee65383a31a752`
and exact serialization/model/optimizer reload checks. This is completed
training, not qualified walking or quiet stopping.

A [source007 reporting continuation](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_002/standing/state.json)
added 30 updates and 23,040 native transitions from checkpoint200, producing
[checkpoint230](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_002/standing/learner/latest_checkpoint.json)
with SHA-256 `fa043b43dc28c41cd8de7f20f53531629662b187230b77c4c2c14ec94acefa61`.
This restores learning state with a fresh simulator and command schedule; it
does not claim exact continuation of the previous physics state. Across the
recorded continuation, moving-command signed speed was −0.0000173 m/s and four
episodes terminated. Its mean quiet penalties were 14.916 against an absolute
mean task reward of 14.656: the penalties dominate this measured objective.
These are stochastic training measurements, not a deterministic walking verdict.

[The recorded Fable review and root decision](../../artifacts/restart_2026-09-14/paper_learning_review_001/DECISION_001.json)
adopt a fresh experiment with gentler large-error tails for the two quiet costs.
For the existing normalized mean square `u`, the cost remains `u` through 1 and
becomes `1 + ln(u)` above 1. The weights, scales, other rewards, controller and
every acceptance gate remain unchanged. This experiment started fresh
without behavior cloning. The
[completed train003 run](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_003/standing/state.json)
now supplies its actual result: 200 updates and 153,600 native transitions on
32 replicas, saving checkpoint
`27c7ef7c15cc61faf90fdb10d636f2bbd7893f213cc14afdf45ec2656cfaed9d`.
Moving-command signed speed was 0.000557 m/s, with 92 terminations and
108 timeouts. The improved reward scale did not produce qualified walking.

The first [camera evaluation attempt](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_001/standing/evaluation/report.json)
returned missing or blank RGB before any policy control. The
[second attempt](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_002/jobs/standing.json)
was stopped after capture stalled before the first frame or policy control.
Both attempts and their cleanup receipts remain intact. Evaluation005 also
failed before any control while inspecting an unresolved SDK camera type.

The subsequent actual Isaac captures are preserved in
[evaluation006](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_006/standing/evaluation/report.json)
(first checkpoint, 1,000 controls and 500 frames) and
[evaluation007](../../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_007/standing/evaluation/report.json)
(train003 checkpoint, 669 controls and 334 frames). Both fail signed tracking
and requested torque-demand gates. Evaluation007 also fails the full duration,
joint-bound and vertical-motion checks. Its
[independent audit](../../artifacts/restart_2026-09-14/paper_walk_execution_001/terminal_audit_evaluate_007/README.md)
finds a left-middle tibia lower-limit crossing at 13.38 seconds, with neither
height nor tilt fall predicate triggered. All 5,352 native samples and motor
inputs are consistent; the native applied torque cap remains intact.

The [CPU analyses](../../artifacts/restart_2026-09-14/paper_walk_execution_001/analysis_evaluate_007/analysis.md)
and [direct visual review](../../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_001/review.json)
find in-place jitter rather than coordinated gait. The two recordings have
different lengths, so no matched full-trial improvement or passing result is
claimed. Original recordings, incomplete prefixes and cleanup are retained.

The [second Fable consultation and decision](../../artifacts/restart_2026-09-14/paper_learning_review_002/DECISION_001.json)
adopt a fresh 200-update, 32-replica contrast with initial standard deviation
0.1 instead of 0.4 and behavior cloning off. Physics, learner, task reward,
replay bytes and gates stay unchanged. Lower exploration is a testable
hypothesis, not an established remedy for mean-policy jitter. The first
allocation, train004, failed simulator startup before AppReady or any learning;
its native startup failure and cleanup remain separate from policy performance.

`run_learning_probe_suite` runs 13 additional low-speed trials sequentially on
one robot in one app, with optional video for one case and a bounded deadline.
It retains the original 20/32-second quiet and 8-second move plus 13-second stop
windows, exact 400 Hz capture and all failed prefixes. Only a recorded native
termination allows a later independent trial; acquisition errors stop the suite.
These probes leave all 96 required cases missing. Full evaluation and the
existing human visual comparison remain required for Stage 2.

The implementation adapts the motion-prior approach discussed in
[arXiv:2511.03167](https://arxiv.org/abs/2511.03167). It combines PPO with a
transition discriminator, five proprioceptive frames, a supervised velocity
estimator and a privileged critic. The actor consumes measured history,
requested motion and applied action context; the simulator supplies critic and
estimator training targets. PPO actions produce bounded joint targets followed
by native dynamics. Reference poses never replace a policy's simulated rollout.

The initial tripod reference is a model-specific bounded inverse-kinematics
fit. It is a declared adaptation rather than a reproduction of the paper's
whole-body dynamic trajectory optimization. Native replay measures whether the
actual actuators and contacts can realize those targets, and records realized
reference transitions. Replay feasibility thresholds remain separate from
Stage 2 gates; its 2% saturation screen does not replace the existing 0.5%
qualification threshold. [The recorded design decision](../../artifacts/restart_2026-09-14/paper_stage2_review_001/DECISION.md)
retains the review, adopted changes and limitations.

| Source | Responsibility |
| --- | --- |
| `env_config.py`, `env.py` | Exact model identity, native dynamics, observations and pre-reset telemetry. |
| `generate_prior.py`, `replay.py` | Kinematic references and physical replay with recorded feasibility checks. |
| `learner.py` | PPO, motion-prior discriminator, velocity estimator and learning-state checkpoints. |
| `task.py` | Versioned movement/stop command changes and rewards from completed native controls. |
| `train.py` | Native diagnostics, admission-bound training and actual Isaac policy recording. |
| `evaluation.py` | Separately report the unchanged direction, transition, quiet-stop and formal comparison screens. |
| `evaluate.py` | Run the complete native evaluation manifest or 13 additional learning probes, retain 400 Hz evidence and actual policy video; diagnostic subsets cannot qualify. |
| `evaluation_config.py` | Declare the longer evaluation timeout while preserving every inherited physics setting. |
| `camera.py` | Native RGB capture with recorded image readiness and checks that rendering preserves physical state. |
| `analyze.py` | Inspect saved native policy traces and checkpoint observations on CPU without changing their recorded verdict. |
| `launch_spark.py`, `reservation.py` | Bounded launches with source/reservation binding, shared locks and exact cleanup. |

New learning requires authentic matching one-robot and batch standing admission
for this source, stance, geometry and replica layout. Full 400 Hz distal-contact,
non-foot-contact and clearance checks remain required; compact tibia-force
telemetry alone cannot qualify standing. A completed checkpoint or video cannot
set `stage2_complete`. Qualification also needs held-out motion and stopping
evidence, the unchanged torque/contact limits and the existing human comparison
with the historical accepted gait.

The sole dispatcher follows [OPERATIONS.md](../../docs/OPERATIONS.md), freezes
each attempted source, and records exact model, prior, checkpoint, native inputs
and output identities. Attempts and recordings belong under a new artifact
identity. Failed attempts and earlier model evidence remain immutable.
`tests/` contains CPU contract checks; those checks do not establish native
admission, learning quality or Stage 2 completion.

The v3 learner supports explicitly enabled rollback of proposed Adam steps
whose mean whole-actor Gaussian KL exceeds the configured target on the full
collected rollout. A rejected proposal restores model and optimizer state and
retries the same clipped gradient at smaller learning rates. Exhausted retries
leave the last accepted model state in place; the discriminator completes its
unchanged schedule. Three consecutive updates with no accepted model step stop
the reviewed candidate allocation and save its checkpoint and diagnostic receipt.
This is an empirical policy-change bound on collected observations, not a bound
on motion or unseen states. Default settings retain the earlier stepping rule.

The separately enabled actor-statistics freeze keeps the checkpoint's loaded
observation mean, variance and count unchanged. Critic and discriminator
statistics continue updating. Existing checkpoints require an explicit,
hash-bound migration into the new schema; the runtime loader never silently
converts them. Learning state is preserved, while each native allocation starts
a fresh simulation. Consult the progress registry for executed configurations
and results; implementation availability alone is not a training result.

`--mode evaluate --eval-scope startup --startup-arm cold` selects an additional
20-second deterministic BC forward diagnostic at 0.05 m/s. The `neutral4` arm
instead issues zero normalized actions and zero command for 200 native controls,
then runs the same BC policy for 1,000 controls. The dispatcher runs each arm in a
separate fresh application allocation with the same checkpoint, seed and physics.
No new fitting or checkpoint conversion happens in this evaluation path.

The warm arm preserves all 1,200 controls, 9,600 native samples, contact packets
and 600 video frames when acquisition completes. There is one trial reset before
capture and none at the handoff. Measured history continues through native steps;
the held target stays at the existing neutral during the prefix. Each row labels
`action_source` as `scripted_neutral` or `bc`; `issued_action` and the legacy
`policy_action` field contain the action actually sent to the environment, while
`actor_mean_action` separately records the policy proposal. Equality is required
on every BC row. Analysis must use these explicit labels; treating scripted rows
as policy outputs would give a misleading actor-reconstruction comparison.

The unchanged scorer receives only the 1,000 recorded BC rows, retaining original
timestamps and its existing 100-control exclusion. The complete raw trace remains
intact, and the report identifies both the policy and first-100-control onset
slices. Prefix, policy and whole-run native physical windows are independently
checked, with zero non-foot contact permitted in the scripted neutral prefix and
the original moving fraction in the policy window. An early acquisition, native,
camera or identity failure cannot be hidden by scoring a later slice. Scripted
neutral does not establish learned quiet standing or stopping. Both arms remain
outside the original 13 learning probes and all 96 required Stage 2 cases; neither
replaces cold-start qualification or the final visual comparison.
