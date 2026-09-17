# Historical training contracts and gates

## Controlled forward-example PPO pilot, 17 September 2026

The successor froze a [paired protocol](../artifacts/forward_example_ppo_20260917/protocol_001/PROTOCOL.json)
to test action-imitation initialization from the passing forward reference.
Both arms use seed 20260914 and the same fixed 0.05 m/s forward command. Each
arm receives 1,200 PPO updates and 3,686,400 transitions on 128 admitted replicas.
The earlier mixed-command PPO run is a separate baseline, outside this pair.

Both arms initialize their observation statistics from the first 700 example
rows. The example arm then fits its actor's action predictions with 1,000 Adam
updates. The scratch arm retains its initial actor weights. The fit preserves
the critic, action variance and PPO random state; PPO starts with an empty
optimizer. Both arms use the same task reward during PPO, without an imitation
reward. The later 300 example rows measure prediction error on the same
recording; they do not provide an independent locomotion test.

The [native checks](../artifacts/forward_example_ppo_20260917/NATIVE_SMOKE_001.json)
verify two PPO updates, checkpoint saves and 49,152 force/torque samples per arm.
They establish the execution interface, without a learning conclusion. The
protocol requires native evaluations at updates 0, 300, 600 and 1,200, with
ground-contact force, motor torque and actual policy videos. Forward tracking
is the primary comparison. Quiet and stop cases are diagnostics because the
training command stays forward. A pass before PPO would establish imitation
of the example; later passes would test its retention through PPO. This pilot
does not reproduce AMP or qualify omnidirectional walking. Stage 2 stays open.

We completed both full training budgets and verified the
[initial saved states](../artifacts/forward_example_ppo_20260917/PAIRED_INITIALIZATION_001.json).
Each run records 29,491,200 force/torque samples and exits with code 0. The
[experiment record](../artifacts/forward_example_ppo_20260917/README.md) links
both training receipts and explains raw-file retrieval.

Before PPO, the [copied actor](../artifacts/forward_example_ppo_20260917/review_001/example_update000000.json)
fails the forward screen with 0.056366 m/s planar error against the unchanged
0.025 m/s bound. Its mean forward speed is 0.005338 m/s. Its quiet trial ends
at control 149 with joint-limit and joint-speed violations. Its forward-to-stop
trial ends at control 467, before the stop command at control 500. We preserve
the failed prefixes and their original verdicts.

The [completed paired comparison](../artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
establishes no benefit from this example initialization under the frozen rule.
Both arms fail their forward screens at updates 0, 300, 600 and 1,200. At the
final checkpoint, mean forward speed is 0.004811 m/s for scratch and
0.000961 m/s for example initialization, against the 0.05 m/s command. Planar
error is 0.051045 and 0.053322 m/s, above the unchanged 0.025 m/s limit. Both
final policies also fail the motor-demand bound.

The final forward captures record mean vertical support of 73.246 and 73.227 N,
with mean absolute applied motor torque of 0.488782 and 0.405900 N·m for scratch
and example initialization. These loads describe failed, low-speed motions;
they do not establish efficiency at matched walking speed. The experiment
record preserves the raw captures and final policy videos. One paired seed
and one initialization method do not establish a result for AMP or other
motion-prior methods. Stage 2 remains incomplete.

## Successor refit comparison, 16 September 2026

The successor completed both cold BC refit trials under frozen source020 and
matching admission002. Each trial contains 1,000 policy controls and 8,000
physics steps, with the 0.040 rad per 20 ms limiter. The
[comparison](../artifacts/restart_2026-09-14/paper_walk_execution_001/bc_refit_analysis_001/RESULT.json)
verifies 76 new result files and reclassifies 355,109 new contact patches.

| Measurement | Parent cold | Uniform refit | Onset-weight20 refit |
| --- | ---: | ---: | ---: |
| First-control requested-target RMS error against nearest onset label (rad) | 0.030758 | 0.006188 | 0.004124 |
| Later 900 controls: planar RMSE (m/s) | 0.050931 | 0.058028 | 0.049995 |
| Original planar error, bound 0.025 m/s | 0.050608 | 0.052897 | 0.049994 |
| Original native physical windows | Pass | Fail: one speed-bound step | Pass |

The original forward gates fail for both candidates. Onset20 settles with mean
forward velocity 0.00000585 m/s for a 0.05 m/s command. The sampled recordings
show onset motion followed by a fixed stance. The
[decision](../artifacts/paper_bc_refit_001/DECISION.json) keeps BC diagnostic.
Onset20 remains an optional initialization candidate for the later three-probe
screen; no warm start occurs in this increment. One trial per checkpoint and
nearest teacher labels cannot establish statistical or causal improvement.
The uniform allocation overlapped a manual reconstruction container; its
service failed the final exclusivity check before the successor completed a
separate cleanup. Do not use that allocation for throughput comparison.

### Next increment after the direct-task review

The [PR 22 review](../artifacts/restart_2026-09-14/mkii_rs05_review_001/ROOT_REVIEW.json)
finds two contact-capture defects: world contact points reach replica-local
link poses, and one average point replaces the original per-patch nonfoot
check. Resolve these defects before direct-task admission. Then bind fresh
standing captures at one and the intended batch layout to the exact source and
model. Record the 1024-replica throughput measurement before a larger allocation.
No direct-task native admission or throughput result exists in this increment.

Keep the approved tripod-reference and bounded-residual design in new maintained
package modules. Preserve the frozen prototype and existing tests. First screen
the reference without a learned residual, then run the approved PPO pilot if
its existing gates pass. At a stop request, retain the last motion reference
through touchdown and verify measured foot support before freezing phase;
selecting the neutral table row at once would bypass that transition. Record
controller state in the new observation and checkpoint identity. These modules
remain unimplemented. James removed the Fable consultation requirement on
16 September; the prior failed consultation remains historical evidence.

**Previous dispatcher paused for handoff — 15 September 2026 UTC.** James said:
“take a pause for now and let someone else continue work, push all non commited
changes”. This pauses the current dispatcher's research after publication;
another designated lead may continue the existing authorized Stage 2 goal
without another user permission. No successor is named or new task queue
created. The [pause receipt](../artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
records no native job, container or GPU compute app at 01:58:15 UTC. The exclusive
reservation remains retained and the historical heartbeat is already PAUSED.
The [operations runbook](OPERATIONS.md) records the changed coordination hash
and fresh guard/binding requirement. Stage 2 remains unqualified.

These task contracts apply to their named mock, C-study or four-bar lineage.
The approved direct-drive robot requires its own explicit contracts and admission
under [ARCHITECTURE §§2–4 and §8](../ARCHITECTURE.md). Existing thresholds remain
unchanged. [STATUS](../STATUS.md) owns progress; this reference authorizes no run.
Earlier priority overlays remain in the [documentation archive](archive/README.md).

The current corrected-model PPO/motion-prior implementation is documented in
[experiments/paper_walk](../experiments/paper_walk/README.md). Its walking neutral,
native admission, checkpoint identity and evaluation remain separate from the
historical task contracts below. [STATUS](../STATUS.md) records its latest
standing, replay and training evidence and the remaining Stage 2 requirements.

The first corrected-model run completed 200 PPO updates and 153,600 native
transitions on 32 replicas, without behavior cloning. Its
[completed acquisition](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/standing/state.json)
and [checkpoint receipt](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/standing/learner/latest_checkpoint.json)
bind checkpoint
`5d48928dd4ba5fdfbd8ff6c6706a5eb8c3951aeaa9aacb8e38ee65383a31a752`
to frozen paper-walk source004. The native replay reference covers all 21
commands using 31 accepted demonstrations; one duplicate remains rejected for
excessive saturation. This records completed learning, not qualified walking or
quiet stopping. Full evaluation and the existing human visual comparison remain
required.

A [source007 reporting continuation](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_002/standing/state.json)
added 30 updates and 23,040 native transitions from checkpoint200, saving
checkpoint230 with SHA-256
`fa043b43dc28c41cd8de7f20f53531629662b187230b77c4c2c14ec94acefa61`.
Its moving-command signed speed remained near zero and four episodes terminated.
Measured mean quiet penalties of 14.916 dominate the absolute mean reward of
14.656. [The adopted experiment](../artifacts/restart_2026-09-14/paper_learning_review_001/DECISION_001.json)
therefore preserves both quiet costs through their existing scales and applies
`1 + ln(u)` above normalized mean square `u = 1`. It uses fresh learning state,
leaves all other rewards and acceptance gates unchanged. The reporting
continuation is not exact physics resume.

Two initial policy-recording attempts failed before producing a frame or policy
control: the first returned blank RGB and the second stalled during capture and
was stopped. Their receipts and cleanup remain preserved. Evaluation005 also
failed before controls because camera source inspection received an unresolved
SDK type. Later native capture succeeds; recording alone does not qualify gait.

The [fresh quiet-tail run](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_003/standing/state.json)
completed another independent 200-update, 153,600-transition experiment on
32 replicas. Checkpoint `27c7ef7c15cc61faf90fdb10d636f2bbd7893f213cc14afdf45ec2656cfaed9d`
passed serialization and restore checks, but moving-command signed speed was
only 0.000557 m/s and 92 episodes terminated. These terminations are not all
classified as falls. Higher training reward does not establish better walking.

Actual deterministic Isaac recordings now exist for the first checkpoint
([evaluation006](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_006/standing/evaluation/report.json),
20 seconds) and the fresh quiet-tail checkpoint
([evaluation007](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_007/standing/evaluation/report.json),
13.38-second failed prefix). Both fail direction tracking and requested torque
limits. The latter stopped when the left-middle tibia crossed its lower joint
limit, while remaining upright; the
[independent terminal audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/terminal_audit_evaluate_007/README.md)
confirms the exact event and recorder integrity. Native applied torque stays
within its cap. The preserved
[visual review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_001/review.json)
finds persistent leg jitter without coordinated travel. Unequal record lengths
do not support a matched full-trial ranking or acceptance.

The [second Fable consultation and adopted decision](../artifacts/restart_2026-09-14/paper_learning_review_002/DECISION_001.json)
select a fresh 32-replica, 200-update contrast with initial action standard
deviation 0.1 instead of 0.4, behavior cloning off and unchanged reward, learner,
physics and gates. This tests an exploration hypothesis; it is not a demonstrated
cause or fix. Its first allocation, train004, failed simulator startup before
AppReady or any training, and its cleanup is retained. The
[prototype reference](../experiments/paper_walk/README.md) links the evidence.

The [train005 retry](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_005/standing/state.json)
completed 200 updates and 153,600 native transitions with initial standard
deviation 0.1 and behavior cloning off. Three episodes terminated; moving-command
signed speed was 0.000739 m/s. In
[evaluation008](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_008/standing/evaluation/summary.json),
all four completed direction trials fail planar tracking. A fourth trial
completed after the stop decision was recorded; the controlled interruption
leaves nine probes missing. Neither completion nor fewer terminations establishes
walking, and the saved interruption and cleanup remain part of the evidence.

A separate [CPU behavior-cloning fit](../artifacts/paper_bc_fit_001/REPORT.json)
completed 1,000 fitting steps on the accepted steady replay rows, with zero PPO
updates and zero newly simulated training transitions. Its neighboring-cycle
holdout is not independent gait validation. No onset or stopping examples were
added. The fitted checkpoint's
[native forward trial](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_009/standing/evaluation/report.json)
completes 20 seconds and 500 video frames but fails planar tracking. The
[direct visual review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_003/review.json)
records bobbing and lateral drift without useful forward travel. Its separate
[quiet and stopping trials](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_010/standing/evaluation/summary.json)
also complete and fail motion, joint-rate, target-motion and six-toe support
checks. Stopping additionally fails requested-torque saturation; applied torque
remains capped. These are BC-only policy results, not PPO training results.

A [second data-only BC fit](../artifacts/paper_bc_fit_002/REPORT.json) adds native
start and zero-command examples. Its
[evaluation011](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_011/standing/evaluation/summary.json)
completes all three selected trials without an acquisition failure: 20-second
forward, 20-second quiet, and uninterrupted 8-second forward plus 13-second stop.
All three fail. Forward planar error is 0.050192 m/s against 0.025 m/s; quiet
joint speed reaches 0.180679 rad/s RMS against 0.03 rad/s and also fails joint
range, target motion and six-toe support. Stopping additionally fails drift,
heading and requested-torque saturation. Applied torque remains capped and no
nonfoot contact is recorded. Ten omitted probes and all 96 required Stage 2
cases remain missing; exact-container cleanup is verified.
The [direct video/trace review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_004/REVIEW.json)
finds a nearly fixed stance after initial adjustments, with no sustained forward
stepping. This is a failed policy result, even though its native capture completes.

The separately reviewed [velocity-supervised CPU contrast](../artifacts/paper_bc_fit_003/VERIFICATION.json)
completes 1,000 BC steps per arm with zero PPO updates or newly simulated
transitions. Its candidate's aggregate velocity RMSE on the related cycle-three
holdout is 0.006127 m/s, compared with 0.903256 m/s for the legacy-parity arm.
That neighboring-cycle comparison is calibration evidence, not independent
walking validation. Separate BC optimizer state is discarded, PPO state remains
fresh, and strict checkpoint restoration passes. The new learner's optional
divergence safeguard and supervised BC settings remain explicit source/config
identities; previous policies and native results retain their original bytes.
Its [three-case native evaluation012](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_012/standing/evaluation/summary.json)
completes all three selected trials and fails each one. Forward planar error is
0.050608 m/s against 0.025 m/s. Quiet joint speed and target motion still exceed
their original bounds. Stopping also fails joint range, six-toe support and a
native velocity bound: one recorded right-front tibia SDK velocity reaches
-114.842804 rad/s against a 50.265484 rad/s limit. The angle-derived rate differs;
the simulator/contact cause remains unestablished. Acquisition completes, but
no failing sample is discarded. The [actual visual review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_005/REVIEW.json)
finds startup oscillation followed by a nearly fixed stance without sustained
forward stepping. Ten diagnostic probes and all 96 Stage 2 cases remain missing.
A separate [fresh CPU fit004](../artifacts/paper_bc_fit_004/REPORT.json)
records a 128-replica configuration with zero PPO updates. Its model and RNG
match the fit003 candidate, but it establishes neither native batch admission
nor simulation resume and does not rewrite the earlier checkpoint.

A separate [128-replica raw audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_batch128_001/audit.json)
recomputes the original standing gates exactly and passes all 128 replicas,
including 8,000 native substeps and full contact-packet checks. The lead then
adopted [admission002](../artifacts/restart_2026-09-14/paper_walk_execution_001/admission_002/verification.json)
for the exact one- and 128-replica simulation layouts, retaining admission001
unchanged. Full raw traces remain on Spark, with local hash-bound compact
metadata and the independent audit. This is simulation standing admission; it
does not establish walking or permit a
32-replica checkpoint to resume under a different configuration.

The admitted [train006](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_006/standing/state.json)
completes 200 PPO updates and 614,400 native transitions from the fresh fit004
state on 128 replicas. The learner records 637.943 seconds of training and saves
checkpoint `eb2847e9bce7db0b090debdbc2b0b3395679ff3e2b78f8ecf7407aaa4a8b2d1a`.
Strict restoration and exact-container cleanup pass. The run records 142
joint-limit terminations, no height/tilt terminations and no compact nonfoot
events. Aggregate requested saturation is 1.5613%; moving-command signed speed
is 0.005313 m/s cumulatively and 0.001116 m/s in the final update. These are
training diagnostics, not qualification results.

The same [learner trace](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_006/standing/learner/metrics.jsonl)
shows that every final-50 update stops model optimization after its first step;
their mean final analytic KL is 0.159151 with target 0.02. The reporting-only
normalizer-change KL reaches 5.797131 at update two. The post-step safeguard
retains the crossing step and does not gate normalizer changes. The subsequent
[Fable review and lead decision](../artifacts/restart_2026-09-14/paper_ppo_stability_review_004/DECISION_001.json)
adopt source018: roll back a crossing model/Adam proposal and retry its identical
clipped gradient at the base, half, quarter and eighth learning rate. Every trial
is compared with the original collected policy on the complete rollout at target
KL 0.02. Loaded actor normalization remains frozen, while critic/AMP normalization
and all 20 discriminator steps keep their schedules. Three consecutive updates
with no accepted model step save a checkpoint and stop this learning diagnostic.
The task reward, physics, curriculum and physical gates remain unchanged.

[Evaluation013](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_013/standing/evaluation/summary.json)
now completes its three selected native trials and fails each. Forward tracking
remains near zero. During the fixed 4–20-second quiet window, joint RMS is
0.558420 rad/s and all 6,400 physics samples lack six-foot support, predominantly
holding one fixed tripod. The fixed 10–21-second stop window improves joint RMS
to 0.005963 rad/s but retains three right-rear support losses and therefore still
fails. Its preceding forward movement also never tracks the command. All 96
required Stage 2 cases remain missing. The
[actual visual review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_006/REVIEW.json)
finds a nearly fixed posture after initial adjustment, without sustained walking.
Only the forward trial has RGB; sampled review and full numerical timelines do
not imply every frame was individually inspected.

The [independent audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_013/README.md)
reproduces all original failed verdicts across 24,400 native substeps and 497,495
saved contact patches. These trials have no joint-position or SDK speed bound
violations, internal resets, active nonfoot contacts or applied-torque cap
breaches. The three stop losses are zero-force right-rear patches with valid
normals; their underlying cause remains unisolated. Its initial CPU comparison
failed only on NumPy percentile arithmetic. That failure remains preserved; the
artifact-only successor explicitly matches native float32 interpolation, changing
no scorer or gate. Recorded mesh-clearance channels are checked, but the audit
does not independently transform every mesh vertex again.

The [explicit CPU checkpoint migration](../artifacts/restart_2026-09-14/paper_ppo_migration_001/run_001/VERIFICATION.json)
preserves learned model/AMP parameters, both Adam states, normalization, serialized
RNG and counters from actual train006 update200. The migrated checkpoint is
`eb561e7f99ddd4e4785121f0940f6c0efd5f50428a75e355f6c2d18005b28f84`.
Own-version strict loaders pass, cross-version loaders reject, and all outputs
match bitwise for 3,050 actual recorded observations. This adds no training
updates; preserved CUDA RNG bytes are not a GPU restoration test and PhysX state
is absent. Source018 is frozen and the separate 549-file v3 source release is
generated. The [train007 dispatch](../artifacts/restart_2026-09-14/paper_walk_execution_001/DISPATCH_train_007.json)
records launch at 00:50:01 UTC on 15 September. The
[completed train007](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_007/standing/state.json)
adds all 20 updates and 61,440 native transitions, preserving learned lineage in
checkpoint220 `e608b454abbed24827810dbb91acf6c17fca677447e49e68cd6fc7e981c2de2a`.
All 17 transferred files verify and exact-container cleanup passes. Its 253
accepted model steps keep full-rollout KL below 0.02: the maximum final-update
value is 0.01983755 and the maximum intermediate accepted-step value is 0.01999543.
Actor-normalizer policy change is zero; all 400 scheduled discriminator steps
run, with no zero-accepted update. These establish the observed update behavior,
not a physical-motion bound or a bound on unseen states.

The continuation still records 14 joint-limit terminations and cumulative
moving-command signed speed of only 0.001242 m/s, with no height/tilt terminations
or compact nonfoot events. The [independent evaluation014 audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_014/README.md)
verifies all three full trials: forward still fails planar tracking, while the
original 20-second quiet and post-command stop windows pass every numeric and
native gate. The latter is quiet recovery after an unsuccessful walking command,
not a demonstrated stop from successful walking. All 24,400 physics steps and
627,468 contact patches were checked; cleanup passes. The long-quiet case,
ten other learning probes and all 96 full Stage 2 cases remain missing.
The [completed train008](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_008/standing/state.json)
then adds 100 PPO updates and 307,200 transitions on unchanged source018, saving
checkpoint320 with strict restoration and all 20 transfer hashes verified.
Maximum accepted-step KL is 0.01999414 and actor-normalizer change is zero, but
74 joint-limit terminations and moving-only signed projection of 0.000677891 m/s
remain. Sampled raw-action clipping averages 51.09%; these training statistics
do not establish improved walking.
The [completed evaluation015 audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_015/RESULT.json)
verifies all 3,050 controls and 24,400 physics steps, with all three selected
policy probes failed. Forward speed after the original two-second exclusion is
−0.000799517 m/s for +0.05 m/s commanded; planar error is 0.051559689 m/s against
0.025 m/s. Quiet and recovery joint-speed RMS are 0.00616294 and 0.01239261 rad/s,
but LF, LR and RM lose contact throughout all 6,400 quiet and 4,400 recovery
substeps; LM, RF and RR remain supported. Physical-bound and cleanup checks
pass without changing the failed policy verdicts. The [root visual review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_008/REVIEW.json)
finds rocking followed by a fixed asymmetric stance without sustained walking.
Further continuation from checkpoint320 is paused; checkpoint220's earlier
quiet/recovery passes remain historical evidence.

The [root-adopted startup diagnostic](../artifacts/restart_2026-09-14/paper_bc_startup_implementation_001/DECISION_001.json)
compares the existing BCfit004 policy from the original reset with the same
policy after four seconds of actual zero-command neutral control, using separate
fresh native apps. The [explicit migration002](../artifacts/restart_2026-09-14/paper_bc_migration_002/ROOT_ADOPTION_001.json)
preserves the policy and prior fitting counters; no new fitting or PPO occurs.
The cold arm records 20 seconds of BC control; the settled arm records its
complete four-second scripted prefix and 20-second BC tail, without a handoff
reset or history replacement. Actor means and issued actions are labelled by
source. Original policy-window scoring is unchanged, and separate full, prefix
and tail physical checks retain every failure. A scripted neutral prefix is
not learned quiet standing; neither diagnostic can qualify Stage 2 or replace
its cold-start cases. The [completed paired audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/startup_pair_analysis_001/REVIEW.md)
verifies 1,000 BC controls in each arm and the additional 200 scripted-neutral
controls in the warm arm, all 78 result-file hashes, 17,600 native physics steps
and 394,457 contact patches. Both arms use migrated checkpoint
`dfb6d3ecc6ff5fae30c77bbac9056875eec6f19c9cb129917f97542d3b68d52b`,
with unchanged BCfit004 model/normalizers and zero PPO updates. Acquisition,
handoff identity and exact-container cleanup pass. Both original planar screens
fail: 0.050607534 m/s cold and 0.059982400 m/s warm against 0.025 m/s. The fixed
later900-control forward means are -0.000047196 and +0.000358904 m/s; complete
1,000-control means are -0.000576008 and -0.000280371 m/s. Keep those windows
distinct. Checkpoint220's earlier quiet/recovery-only passes and checkpoint320's
three failed probes remain separate PPO evidence, not acceptance of this BC parent.

The first warm BC observation is close to the same native onset row as the cold
arm: normalized RMS distance 0.009918 versus 0.217287, with current-joint RMS
difference 0.0853 versus 22.180 mrad. Its first requested teacher-target mismatch
remains 31.864 mrad; the actually held target differs by 6.671 mrad after the
unchanged limiter. Later policy observations leave demonstrated feature ranges.
These are descriptive measurements, not proof of root-height causality or a
validated off-trajectory teacher. The comparison probes physical/contact onset
and recent five-frame observations, not four seconds of recurrent policy memory.
All six plots visibly distinguish the prefix. The generic maintained analyzer's
rejection of Unicode `action_source` is preserved; artifact-only typed loaders
retain numeric finiteness, hashes and exact counters without changing raw data.
The initial timed-out Fable consultation and its
[interpretation caveats](../artifacts/restart_2026-09-14/paper_bc_startup_review_001/CAVEATS_AND_RECOMMENDATION.md)
also remain intact.

The later [adopted BC comparison](../artifacts/restart_2026-09-14/paper_bc_learning_review_001/ROOT_DECISION_001.json)
now has an executed result. The helper moved into the maintained prototype as
`experiments/paper_walk/refit_bc.py` under [protocol 002](../artifacts/restart_2026-09-14/paper_bc_refit_protocol_002/README.md),
which binds the relocated helper bytes to the unchanged recipe: 1,000 additional
CPU BC steps for each of two arms from the same parent and identical uniform
draws, unchanged action MSE versus action-only weight 20 on 23 first-moving-onset
rows divided by the fixed mean weight 4197/3760. The
[paired result](../artifacts/paper_bc_refit_001/REVIEW.md) records zero PPO
updates, identical minibatch and final RNG hashes and strict reloads. The
onset-weighted arm reduces first-moving-onset target RMSE from 0.026681 to
0.001857 rad against 0.008535 rad for the uniform arm; its steady-row RMSE is
0.001653 rad against 0.001262 rad. Velocity errors are identical between arms.
These are fitting errors on training rows. Neither candidate has a native
evaluation; the Spark host was unreachable at fitting time. Fitting error cannot
qualify walking.

The additional 13-case learning probe suite reuses the original numeric scorers
and full 400 Hz motor/contact capture in one native app. It preserves the
20/32-second quiet windows and 8-second move plus 13-second stop case. Each trial
has one initial reset and no automatic reset within its recorded window. These
low-speed diagnostics leave all 96 required Stage 2 cases missing; the full
evaluation and existing human visual acceptance remain necessary.
An explicitly ordered subset retains all 13 declarations, distinguishes omitted
from unattempted selected cases, and cannot pass the diagnostic suite unless
all 13 complete and pass. Video selection must belong to the selected cases.
Reporting-only termination reasons and actual target-slew occupancy distinguish
existing predicates and exclude each replica's first control after reset; they
change neither reward nor physics. The source018 prototype passes 155 CPU tests.

## Direct task on the approved model: MKII-RS05

`Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0` carries the model
`robot/active_model.json` selects into the maintained `hexapod_env` package, so
RSL-RL can train it at thousands of replicas. It is a new identity beside the
paper-walk prototype; no historical task, asset, checkpoint or admission moves.

The task repeats the accepted prototype recipe. Physics runs at 400 Hz with
eight substeps per 50 Hz control step, 32 position solver iterations and no
velocity iteration, external forces every iteration, self-collision enabled,
flat ground with friction 1.0 and no restitution, and 2 m spacing. One control
step emits a joint target from the stance plus 0.35 rad times the action,
clamped to the exact URDF limits and held within 0.040 rad of the previous
target. The policy observation holds five 42-value proprioceptive frames, the
three-value command and the 18 previous executed actions, which is 231 values;
the critic observation adds the privileged base velocity, which is 234. The
reward keeps the prototype's tracking, yaw, tilt, torque, slew and vertical
terms with their original scales, and the episode ends on a low plate, a tilt
beyond 0.85 rad or a joint limit violation.

The actuator is
`packages/hexapod_env/hexapod_env/actuators/rs05_paper_walk_model.py`. It
reproduces `motor_force` value for value: a 12 N*m per radian position gain,
the 18 damping values bound by joint name, the float64 speed curve over the
published knots, zero authority at and above 480 rpm and the provisional
1.6 N*m clamp. The runtime writes efforts alone and keeps the implicit drive,
armature and joint friction at zero. The 1.6 N*m clamp stays an experimental
setting from the [RS05 review](RS05_SPEC_REVIEW.md), not a measured limit.

The task's articulation order is the block order PhysX reports, while every
observation, reward and capture channel uses the canonical per-leg order. The
environment maps between them by name and refuses to run when the articulation
disagrees with the declared order or limits.

`isaaclab/admit_mkii_rs05.py` records a standing capture in the layout the
unchanged `experiments/paper_walk/env.py:score_diagnostic` reads, and that
scorer grades it. `isaaclab/train_mkii_rs05.py` is scratch-only: it refuses
every checkpoint and resume option, defaults to 4096 replicas and headless
startup, and requires a recorded transitions-per-second measurement at 1024
replicas first. CPU tests compare the action path, the observation widths, the
reward terms, the terminations and the capture layout with the frozen source.
CPU agreement is not native behavior, and this task has no capture, training,
checkpoint or video yet; the
[prepared runner](../artifacts/restart_2026-09-14/mkii_rs05_admission_prep_001/README.md)
records why the native steps stayed unexecuted.

## 1. Simulator and framework stack

```text
Simulator: Isaac Sim 6.0.1 + Isaac Lab DirectRLEnv / PhysX
RL:        RSL-RL 5.0.1, PPO, actor/critic [256, 256, 128], ELU
Policy observation normalization: enabled; frozen on model-only recovery
```

The task package is `packages/hexapod_env`, and `isaaclab/hexapod_rl`
re-exports it. `packages/hexapod_env/hexapod_env/register.py` registers the
gym task IDs. The IDs are frozen: checkpoints, launchers, evaluation payloads,
and manifests reference them by string, so do not rename an existing ID. The
current research task ID is:

```text
Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0
```

Assets used by the task:

```text
URDF: robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf
USD:  robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
container USD path: /workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
mass target: 1.5 kg body + six 0.8 kg complete legs = 6.3 kg
```

Asset v1 (ADR-0001), the CAD assembly, has its own task ID, so each ID above
keeps loading the mock it trained on:

```text
task ID: Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0
URDF: robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf
USD:  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda  (tools/assets/import_urdf_to_usd.py)
container USD path: /workspace/hexapod/robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda  (override: HEXAPOD_MKII_V1_USD_PATH)
mass: 8.261 kg from Onshape per-part properties, every RS05 hard-set to 191 g; no sensor payload yet
spec: packages/hexapod_env/hexapod_env/assets/spec.py :: MKII_V1_ASSET
runtime joint order: packages/hexapod_core/hexapod_core/joints_v2.py (provisional until read back from the imported articulation)
```

One `HexapodAssetSpec` describes each robot model (USD, root link, joint and
link names, limits, stance, foot-pad geometry, expected runtime joint order).
An env config points at a spec. `HexapodEnv` reads link names from the config.
If the config names an expected runtime joint order and the imported
articulation's `joint_names` differ, `HexapodEnv` refuses to construct. To add
a robot you write a new spec and a new env config under a new task ID.

## 2. Observation, action, and timing contract

```text
observation dimension: 66, no privileged state
  root linear velocity          3
  root angular velocity         3
  projected gravity             3
  command [forward, lateral, yaw] 3
  joint position error         18
  joint velocity               18
  current clipped action       18
action: 18 joint position offsets, clipped to [-1, 1], scale 0.20 rad
processed joint-target slew limit: 0.040 rad per 20 ms
physics: 0.005 s / 200 Hz; decimation 4; policy rate 50 Hz
episode: 20 s; PPO rollout: 24 steps per update
```

The 66-dimensional layout and the 18-action interface are the deployed
interface. A task that changes either one cannot load an existing checkpoint
model-only. The Stage2G scratch arm is the deliberate exception and uses 68
dimensions (see §4).

The `0.040 rad / 20 ms` processed-target slew limiter is a training setting
and the playback setting of each formal screen on record. Comparisons across
checkpoints are valid only at a common limiter value.

Runtime joint/action order:

```text
 0 revolute_1_1   1 revolute_1_7   2 revolute_2_5
 3 revolute_3     4 revolute_4     5 revolute_5
 6 revolute_1     7 revolute_1_6   8 revolute_1_5
 9 revolute_1_3  10 revolute_1_4  11 revolute_1_2
12 revolute_2    13 revolute_2_6  14 revolute_2_4
15 revolute_2_2  16 revolute_2_3  17 revolute_2_1
```

Indices 0-5 are coxa, 6-11 femur, 12-17 tibia. Do not infer physical leg order
from these names, and do not deploy to hardware before you verify the mapping
on the robot.

RS05 actuator approximation applied to all 18 joints:

```text
continuous/applied effort limit: 1.6 Nm
raw simulated demand ceiling:    5.5 Nm
nominal velocity: 50.27 rad/s; simulation velocity: 55.29 rad/s
stiffness: 30.0 Nm/rad; damping: 0.6 Nms/rad
armature: 0.0007 kg m^2; friction: 0.01; viscous friction: 0.002
soft joint limit factor: 0.95; self-collision disabled
```

## 3. Coordinate contract

```text
anatomical/navigation forward = imported body -Y
navigation lateral            = imported body +X
navigation up                 = imported body +Z
```

This contract fixed the earlier policy that walked sideways while it reported
"forward" progress. The `command_frame: "navigation"` profile, introduced in
Phase1 v5, exposes it, and each later stage inherits it. Do not revert it.

## 4. Curriculum stages

Each stage is a separate registered task and a separate log/checkpoint
namespace. Later stages inherit the earlier stance, limiter, and safety
structure unless stated otherwise.

### Phase 1: forward walking in the body frame, then in the anatomical frame

| Stage | Configuration | Intent |
| --- | --- | --- |
| Phase1 v2 | `phase1_v2_cfg.py`: action scale 0.20, linear-velocity reward 4.0, yaw-rate reward 1.0, yaw std 0.20, fall penalty -1.0 | Safer second pass at forward walking; reduce raw position demand and protect heading and falls |
| Phase1 v3 | `phase1_v3_cfg.py`: yaw-rate reward 2.0, yaw std 0.10, rated-torque excess -0.12, saturation -0.5, 200 iterations | Torque-aware, heading-stable fine-tune from the strongest v2 gait; v2 action semantics preserved so v2 checkpoints can resume |
| Phase1 v4 | `phase1_v4_cfg.py`: yaw-rate reward 2.0, yaw std 0.10, rated-torque excess -0.03, saturation 0.0, 75 iterations | Heading-only refinement that leaves the learned speed/torque tradeoff alone |
| Phase1 v5 | `phase1_v5_cfg.py`: `command_frame = "navigation"`, 500 iterations, scratch run | Corrected anatomical nose-forward axes (§3). Its `model_200.pt` is the last policy with admitted anatomical-forward video evidence |

### Phase 2: recovery and axis acquisition

| Stage | Command structure | Intent |
| --- | --- | --- |
| Phase2 warmup | opt-in omnidirectional velocity-command profile | Bridge from Phase 1 reset-only commands to sampled velocity commands |
| Stage1 recovery | 55/20/20/5 episode-stable gait-preserving buckets | Re-anchor to the accepted anatomical-forward gait under the Phase 2 command contract |
| Stage2 | 40/30/30 forward/yaw/lateral buckets | Force balanced signed y and yaw acquisition while retaining the forward gait |
| Stage2B lateral | 40/40/10/10 forward / pure-y / forward+y / forward+yaw | Acquire signed lateral gait without sacrificing the safe lower stance |
| Stage2C stabilized forward | stand plus forward-only commands with deliberate stationary anchors | Lower, hardware-aware forward gait with deck steadiness as an explicit objective. This is the current research target |
| Stage2D homotopy (C0-C5) | six short stages, one fixed-bucket command per episode, oblique heading rotated toward pure lateral | Rotate the dominant command from forward to pure navigation-frame lateral while keeping Stage2C stance and limits. **Configured; admission pending** |
| Stage2E joystick (E0-E2) | three short stages; reverse-x grows 0.02-0.05, then 0.04-0.10, then 0.08-0.20 m/s; fixed holds shorten 8 s to 5 s to 3 s | Bridge an accepted Stage2D C5 pure-y policy to a real navigation-frame x/y/yaw joystick with command transitions. **Configured; admission pending** |

Stage2D and Stage2E exist as configuration and grading code only. No Stage2D
or Stage2E policy has passed admission, and Stage2C has no admitted candidate,
so those stages have no valid parent yet.

### Stage2C pose and command profile

```text
root reset height 0.185 m
coxa default 0; femur 0.6000 rad; tibia 2.2335 rad
moving/standing deck targets 0.181 / 0.177 m
20% stand, 80% forward; speed 0.16-0.32 m/s; command hold 6-10 s
```

The stance values come from the controlled stance sweep.

### Stage2G: insect-gait rework (two arms)

Rationale, from `isaaclab/hexapod_rl/phase2g_cfg.py`: Stage2C's
speed-conditioned five/four/three-foot support floor combined with heavy
deck-stillness costs rewards a statically supported shuffle. A frozen deck
and a lively tripod gait are exclusive objectives. Stage2G replaces that structure with the alternating-tripod
organization insects use at speed:

- a per-environment gait clock paced by commanded speed
  (`gait_cycles_per_meter = 8.0`, clamped to 1.2-3.0 Hz, duty factor 0.5),
- smooth expected stance/swing windows per tripod with runtime-geometric tripod
  assignment,
- a swing-apex clearance target (`0.030 m`, tolerance `0.015 m`),
- the fixed-target air-time bonus disabled, since the phase reward supersedes it,
- the speed-conditioned support schedule replaced by a plain three-foot support
  floor kept only as a safety bias (`support_shortfall_reward_scale = -0.25`),
- relaxed deck-stillness weights so a stepping rhythm is admissible.

Stage2G inherits the RS05 torque costs and the 5.5 Nm raw-demand termination
unchanged.

| Arm | Task ID | Observation | Action authority | Purpose |
| --- | --- | --- | --- | --- |
| Scratch | `Isaac-Velocity-Omni-Stage2G-Insect-Gait-Hexapod-RobStride-Direct-v0` | 68-dim (adds the two-value gait-phase observation) | action scale 0.25, slew limit 0.08 rad / 20 ms | Learn swing-speed leg motion from scratch with a phase-clock input |
| Adapt | `Isaac-Velocity-Omni-Stage2G-Insect-Gait-Adapt-Hexapod-RobStride-Direct-v0` | 66-dim, Stage2C-compatible | action scale 0.20, slew limit 0.06 rad / 20 ms | Keep the deployed interface so the immutable current-best checkpoint loads model-only; only the reward terms change |

The adapt arm's `0.06 rad / 20 ms` training limiter is looser than the
`0.040 rad / 20 ms` limiter of each formal screen on record. Screen a Stage2G
candidate at the common `0.040` limiter before you compare its numbers with
the Stage2C ledger.

Stage2G has no evaluation artifacts, no mirrored checkpoint, and no SHA-256 in
this repository. See `STATUS.md` for the open contradiction this creates.

## 5. Reward philosophy

The reward is a small set of task terms plus a large set of bounded safety and
stability costs, with every candidate intervention term defaulting to zero
scale. Default scales live in `isaaclab/hexapod_rl/env_cfg.py`; per-stage
profiles override them. Each probe activates one term, so a change in outcome
has one cause.

Term groups:

- **Tracking.** Commanded linear velocity and yaw rate; in Stage2C also signed
  longitudinal progress and a normalized longitudinal tracking error.
- **Deck stability.** Deck-stability composite, vertical velocity, angular
  velocity, flat orientation, base height error. Stage2C weights these high.
  Stage2G relaxes them.
- **Contact and support.** Support shortfall against the stage's support target,
  foot slip, undesired body contact, and (Stage2G only) gait-phase contact
  agreement and swing-apex clearance.
- **RS05 torque safety.** Joint torque magnitude, rated-torque excess above the
  1.6 Nm continuous limit, worst-joint rated-torque excess, torque saturation,
  torque slew, joint acceleration, action rate, and joint-limit proximity.
  These are what keep continuous-duty fraction, burst length, raw peak demand,
  and worst-joint duty inside the RS05 envelope.
- **Probe terms, inactive by default.** Inactive lateral velocity, inactive yaw
  rate, inactive yaw-rate slew, inactive ground-contact yaw moment, and
  inactive bilateral longitudinal contact moment all default to scale 0.

Each probe attempt records the Stage2C effective override values verbatim in
`effective.params`. They include: longitudinal signed progress `+2`, normalized
longitudinal error `-1`, inactive yaw rate `-160`, support shortfall `-0.90`,
rated-torque excess `-0.65`, worst-joint excess `-0.50`, saturation `-1.50`,
torque slew `-0.01`, foot slip `-0.50`, ground-yaw `0`, yaw-slew `0`, and (for
the Probe21 arm) bilateral longitudinal moment `-2` with reference `1.8 Nm`.

### Termination and reset

```text
terminate on: base contact > 5 N
              base height < 0.055 m
              projected-gravity Z > -0.45 (upside-down / severe tilt)
              raw torque demand > 5.5 Nm continuously for 0.10 s,
                after a 0.50 s grace period
truncate on:  20 s episode limit
```

Reset restores the stage pose and velocity, perturbs joints by `+/-0.03 rad`,
clears action and metric histories, and resamples the command and its timer.

### Domain randomization present in Stage 2

Terrain is flat. Startup randomization covers friction and restitution in 64
buckets and an additive root mass in `[-0.20, +0.40] kg`. The task has no
rough terrain, latency, encoder/IMU noise, voltage sag, thermal model,
actuator-strength variation, or push randomization.

## 6. Stage2C acceptance gates

A candidate replaces the current best only after it passes each of these in
the canonical formal screen (deterministic nominal playback, seed 60, 25 warmup
steps, 500 requested steps, `0.040 rad / 20 ms` final target limiter, commands
`(0,0,0)`, `(0.16,0,0)`, `(0.20,0,0)`, `(0.30,0,0)`):

- Yaw-rate RMSE at most `0.080 rad/s` at every moving command.
- Moving normalized deck composite at most `1.000`.
- Zero falls and zero timeouts.
- Achieved forward speed at least `0.240 m/s` at the `0.30 m/s` command.
- RS05 limits held: bounded continuous-duty fraction, bounded burst length,
  bounded raw peak demand, bounded worst-joint duty, and never exceeding the
  `5.5 Nm` raw-demand safety termination.

Beside the numeric gates, the stated qualitative goal is a lower, steadier
gait: less deck wiggle and a more upward-angled femur, so the platform sits
lower.

Screen classes are not interchangeable. Short 6-second probe screens carry 275
post-warmup samples and serve diagnosis only
(`formal_admission_eligible=false`). The canonical formal evaluation is 10
seconds and 475 samples. Only the formal screen can replace the current best.

A sharded formal screen repeats the parent report and the wrapper
action-processing record in each shard. The fail-closed merger compares those
copies for exact parsed-JSON equality, then accepts the exact parent plus
`model_0.pt` through `model_11.pt` membership and ordering.

## 7. Known sim-to-real gaps

Model gaps present in the current task:

- Collision geometry comes from complex visual meshes. The model has no
  explicit foot pads, and self-collision is off.
- Link masses, inertias, and friction are estimates. The 1.5 kg body and 0.8 kg
  complete-leg figures are user-specified mass targets. Nobody has weighed an
  assembled robot. 191 g is the published per-actuator mass. RobStride does
  not publish RS05 housing inertia, joint friction, latency, or thermal
  parameters, and the URDF generator records the estimates it used.
- The model has no backlash and no thermal / I-squared-t behavior.
- The default tibia angle sits near its upper bound.
- No physical observation/action parity test exists.

### Sim-to-real gates still required

- Replace or identify the assumed body/leg mass distribution, housing inertia,
  and friction.
- Add measured CAN/control delay, encoder/IMU noise, voltage sag, and an RS05
  thermal / I-squared-t model.
- Refine link collisions into primitives plus explicit foot pads, and validate
  self-collision.
- Verify observation parity and joint signs on a restrained robot before
  powered walking.
- Enforce the 1.6 Nm training limit, joint limits, watchdog, and E-stop on
  hardware.

No checkpoint in this repository is hardware-ready, including the current best.

## 8. Validation gate

`isaaclab/validate.py` checks 18 joints, 19 rigid bodies, six foot sensors,
finite observations, 1,000 standing steps, falls and timeouts, raw pre-clip
torque demand, and post-settling saturation. It is a structural gate on the
asset and task. It screens no policy.

## 9. Roadmap beyond Stage2C

The locomotion work serves the mission in `ARCHITECTURE.md` §1: survey an operator-drawn
bounded area and hold a steady platform for data collection (ADR-0004). For
training that means:

1. Stable anatomical forward/backward walking with a low, steady deck on
   asset v1 (the CAD assembly). The deck-stability composite is the primary
   grade and speed is secondary. After the team lead fills in the
   deck-steadiness numbers, you re-derive the thresholds from `ARCHITECTURE.md` §1.
2. General joystick locomotion: forward, reverse, lateral, diagonal, and yaw in
   both signs, including the command transitions a coverage planner emits at
   the ends of sweep lines.
3. The `ARCHITECTURE.md` §1 terrain class (grass, gravel, a stated slope) with a
   body-frame height scan (contract C2, ADR-0002). The lidar serves mapping
   and localization and does not feed the policy.

`ARCHITECTURE.md` holds milestones, gates, and ownership. The user has confirmed
RealSense D455 and Livox Mid-360 ownership, permits additional purchases, and
authorizes terrain/perception preparation in parallel with the C-study Stage 2
priority. Terrain training still requires the selected policy's Stage 2 gates
and exact terrain/source admission; authorization does not replace those
checks. Keep ideal terrain information, simulated sensing and calibrated
physical sensing separately identified. `STATUS.md` records execution state.

## C-study fine-tune initialization and allocation experiments

For the isolated C study, `omni.repair_training` describes a deliberately checkpoint-bound fine-tune. It must explicitly record checkpoint SHA, exploration standard deviation, entropy coefficient, optimizer reset and initial learning rate. Verify actor/critic tensors and both observation normalizers exactly after load; a saved checkpoint can overwrite `init_std`, so an initializer-only config change is insufficient. Keep these records distinct from a full optimizer-state resume. The bounded `omni-repair-pair` launcher independently starts each 50-update branch and rejects changes outside the intended plan delta.

A standing raw-action penalty measures sampled normalized intent before wrapper clipping and joint-target slew; feedback remains active. All existing physical, motor, per-direction and visual gates still apply. A short allocation screen cannot qualify a controller or turn a failed result into progress by averaging directions. See [repair 003](../artifacts/omni_diagnostics_2026-09-09/repair_003/README.md) for the exact tested settings and limitations.

### C-study temporal action and reference investigation

The separately versioned [CPU reference feasibility review](../artifacts/omni_diagnostics_2026-09-09/reference_feasibility_001/README.md) preserves failed workspace/rate/transition tests. It motivates support-aware swing/stop timing and a stateful target-velocity action comparison. These are candidate control interfaces, not new acceptance thresholds or qualified gait behavior. Both require executable controller state in observations, exact reset handling and a new checkpoint lineage if action meaning changes. Keep the original PPO actor/checkpoint frozen.

The diagnostic 0.03 rad/20 ms target limiter and archived formal 0.04 comparison remain separate. Benchmark 1's reference authoring bypassed the inherited limiter; its original video is a visual target, not a retrospective formal pass. Measure commanded versus admitted twist, actual target/velocity/acceleration, joint motion, contact slip and motor demand through starts, stops and both turn signs. A smooth reference or CPU test cannot establish quiet physical balance.


### C-study measurement completeness

Shared diagnostic capture records the installed SDK's raw `quaternion_world_xyzw` and explicitly reordered `quaternion_world_wxyz` for quiet scoring. Do not infer old payload component order from its label: preserve original traces and perform versioned reanalysis before reusing historical heading passes. The 2° quiet-heading bound is unchanged.

Full-C terrain admission additionally requires a completed-log check with no reported incomplete contact/friction data. A raw contact gate cannot establish support when its buffer overflowed. The adapter reserves at least 128 point/friction records per prim, preserving larger settings; force, support and motor limits remain fixed. Corrected buffer capacity still requires fresh simulator evidence.

## Optimized forward reference experiment, 16 September 2026

James authorized a trajectory-optimization experiment for the approved robot.
The [maintained implementation](../experiments/trajectory_optimization/README.md)
solves full-body dynamics with a fixed tripod contact schedule and replays the
result through the admitted motor controller. It requires no physical robot.
The first successful solve satisfies its CPU constraints, with 1.29064 N·m
peak torque and 0.0223182 rad peak cyclic target change.

The [native result](../artifacts/trajectory_optimizer_20260917/review_001/RESULT.json)
records 0.967614 m of forward displacement in 20 seconds. Mean forward speed is
0.0494640 m/s after settling, against 0.05 m/s commanded. Speed varies from
−0.0141956 to 0.108552 m/s; planar tracking error is 0.0332708 m/s and fails the
unchanged 0.025 m/s limit. The complete 8,000-step capture passes the native
motor, joint and contact checks. The audit reproduces the servo targets and
reclassifies 131,634 contact patches.

The [video review](../artifacts/trajectory_optimizer_20260917/review_001/VISUAL_REVIEW.json)
finds repeated foot lifts and translation in sampled frames. The replay saves
actual state transitions but excludes them from the accepted motion-prior pool.
This experiment performs no PPO training and establishes no stopping or
omnidirectional result. Stage 2 remains incomplete. The next research decision
concerns speed variation and reference feedback, with the existing gates intact.

## Locomotion load tracking

James requires force tracking before the next optimizer/PPO sequence. The
replay wrapper now writes `force_metrics.json` beside each native evaluation. For future PPO
evaluations and learning probes, use the same wrapper or run the analysis
command below before proceeding to the next experiment. The original
paper-walk evaluator retains its bytes. The report uses all recorded 400 Hz
samples and identifies each case and replica. Before the next experiment, check that the summary is
available. A missing summary records an error; it does not substitute zeros or
change the existing gait verdict.

You can compare total vertical support force and each foot's contact-normal
resultant, in newtons. The full-window foot mean includes swing samples. The
contact-conditioned mean uses samples above 1 N and returns null for an unloaded
foot. Each measure includes its mean and RMS, plus p95 and recorded peak. The
recorder captures normal contact forces; these values exclude tangential
friction. The 400 Hz peak describes the recorded interval force, not an
unresolved impact peak or a measured hardware load.

You can also compare applied and requested motor torque in N·m. The report uses
absolute values to prevent opposite torque signs from canceling. It preserves
per-joint statistics and identifies the joint with the highest RMS torque.
Torque RMS describes motor loading; it does not establish electrical power or
thermal safety.

The report separates the full trial from its startup interval. Its locomotion
window includes samples after two seconds with a nonzero translation or yaw
command. It keeps samples where the robot fails to move. Empty windows remain
empty, and incomplete trials retain their sample counts. Startup duration is an
explicit analysis parameter and does not change any acceptance window.

The [first derived summary](../artifacts/locomotion_force_metrics_20260916/forward_replay_001.json)
uses the preserved optimized forward replay; no new Spark job was required.
Its 7,200 locomotion samples cover the final 18 seconds. Mean vertical support
is 73.24269 N, with a recorded peak of 108.10469 N. Mean absolute applied torque
across the 18 motors is 0.314014 N·m. The left-middle femur has the highest RMS
applied torque at 0.745598 N·m. The mean support is close to the nominal model
weight; foot-load distribution and peaks help distinguish gait loads. The
original tracking failure remains unchanged.

To analyze an existing recording without changing it, choose a fresh output:

```sh
uv run python -m experiments.trajectory_optimization.force_metrics --directory artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation --output tmp/forward_force_metrics.json
```

The new summary pins the input declaration and capture receipt, plus each native
chunk it reads. It does not rewrite the old attempt or its published manifest.

## Standard PPO and improved forward reference, 17 September 2026

James requested a standard PPO baseline and an improved optimized reference,
followed by a measured comparison. The
[protocol](../artifacts/ppo_reference_comparison_20260917/PROTOCOL_001.json)
fixes the model and the 0.040 rad / 20 ms limiter. The baseline uses RSL-RL
5.0.1 PPO with the existing admitted simulator and command/reward task. It uses
no imitation loss or motion-prior reward. This keeps the new DirectRLEnv port
outside the baseline's physics path.

The optimizer adds a root-velocity objective with weight 1 and a target-curvature
objective with weight 0.1. The defaults remain zero for reproduction of the first
solve. The new CPU solution passes the same constraints. Its target curvature
increases despite that penalty; its predicted root-speed error decreases. The
native outcome determines the value of this combined objective change.

The [native audit](../artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
verifies 36 transferred files and 8,000 physics samples. It reclassifies 130,821
contact patches and reproduces the servo targets and original scorer checks.
Forward displacement reaches 0.974348 m in 20 seconds. Planar tracking error
decreases from 0.033271 to 0.003029 m/s and passes the unchanged 0.025 m/s bound.
The full-trial native motor, joint and contact checks pass.

The [load report](../artifacts/ppo_reference_comparison_20260917/replay_pack_001/replay_001/standing/evaluation/force_metrics.json)
covers 7,200 locomotion samples. Mean absolute motor torque decreases from
0.314014 to 0.308198 N·m. Support-force p95 decreases from 82.42784 to 76.23228 N;
mean support remains 73.24 N. The highest joint RMS torque increases from
0.745598 to 0.761535 N·m. The right-rear foot peak increases from 28.98998 to
41.79611 N. You can use these values to assess the load tradeoff. They do not
establish motor efficiency or lower load at each foot.

The [native video](../artifacts/ppo_reference_comparison_20260917/replay_pack_001/replay_001/standing/evaluation/rollout.mp4)
contains 500 decoded frames. The sampled frames show upright support and changes
in foot position. This forward-command screen does not qualify stops, turning,
terrain or Stage 2. Human gait acceptance remains pending.

`experiments/trajectory_optimization/vanilla_native.py` supplies the standard PPO
entry point. It binds the admitted physics and the installed learner source.
Training records 400 Hz normal loads by body and native applied torque. Separate
evaluations retain exact toe/shaft classification and the per-foot summaries.
The two-update adapter check completes 6,144 transitions and 40 optimizer steps.
The [full baseline audit](../artifacts/ppo_reference_comparison_20260917/TRAINING_RESULT_001.json)
verifies 1,200 updates, 3,686,400 transitions and 24,000 optimizer steps. It binds
checkpoint `e5084b221e64413f89c1ecc33e3a6349d81f8691f4fb97a031da9f032e73972b`.
The force summary covers 29,491,200 physics samples across the 128 replicas.
Training records 1,773 physical terminations. Mean movement along nonzero
translation commands increases from 0.000174 m/s over the first 100 updates to
0.008328 m/s over the final 100. These training averages include exploration
noise and mixed commands; they do not qualify walking.

The native training process exits with code 0. A competing COLMAP container
blocks the wrapper's final resource check after the checkpoint save. The
successor preserves its recovery image and stops it, then completes the retained
cleanup command. The service's original failed exit remains in the evidence.
This interval cannot support an isolated throughput comparison. The separate
policy evaluation uses the final checkpoint and the declared fixed cases.

The [policy audit](../artifacts/ppo_reference_comparison_20260917/review_vanilla_001/RESULT.json)
checks 24,400 physics samples and reclassifies 530,534 contact patches across
the complete forward, quiet and stop trials. CPU inference from the saved
weights matches the recorded policy actions within 0.00000144. The audit
reproduces the original scorer verdicts and force summaries. All three probes
fail. Quiet standing includes three native joint-speed violations; stopping
includes one nonfoot contact sample. The evaluation service exits with code 0
and removes its owned container. Completed acquisition does not qualify behavior.

The [comparison](../artifacts/ppo_reference_comparison_20260917/COMPARISON_001.json)
uses the same 0.05 m/s forward command and the existing limiter. Each force
average covers the final 18 seconds, including samples with failed movement.

| Measurement | Original reference | Improved reference | Standard PPO |
| --- | ---: | ---: | ---: |
| Mean forward speed, m/s | 0.049464 | 0.049468 | 0.000423 |
| Planar tracking error, m/s | 0.033271 | 0.003029 | 0.051091 |
| Forward screen | Fail | Pass | Fail |
| Mean absolute applied torque, N·m | 0.314014 | 0.308198 | 0.491920 |
| Support-force p95, N | 82.42784 | 76.23228 | 90.87415 |
| Highest joint RMS torque, N·m | 0.745598 | 0.761535 | 1.263698 |

You can inspect the [complete forward traces](../artifacts/ppo_reference_comparison_20260917/review_vanilla_001/forward_comparison.png)
and the [actual PPO video](../artifacts/ppo_reference_comparison_20260917/vanilla_evaluate_001/run/standing/evaluation_00/rollout.mp4).
The improved reference reduces the original tracking error by 90.9 percent.
Its steady mean speed resembles the original reference, while its speed varies
less within each gait cycle. The PPO checkpoint makes little forward progress
and exceeds the requested-torque-demand fraction limit.

This one-seed pilot establishes the recorded baseline at 3,686,400 transitions.
It does not establish PPO convergence or its best performance. The optimized
reference covers one command and uses no learned feedback. Reference-assisted
PPO remains untested. A paired learning comparison and broader direction/stop
coverage remain the next research questions; Stage 2 remains incomplete.
