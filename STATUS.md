# Current project status

Updated 10 September 2026 (UTC). **Stage 2 is not complete.** The user-selected C study remains the execution priority: useful-speed, smooth translation, turning, arcs and paths in every direction, with quiet standing and visual quality matching accepted forward Benchmark 1. Both latest 50-update PPO pilots completed but failed all 48 quiet-stop trials. A complete 38-second recording of the selected CAPS checkpoint is available; its host job failed afterward on a missing final audit receipt. The bounded batch has ended and the exact deferred forecast replay has been restarted. This is a published snapshot, not live GPU telemetry.

## Major checkpoints

1. **Walking:** [Benchmark1](artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) is the user-accepted forward visual reference. Its requested-torque failure remains on record; this is not hardware or all-direction qualification.
2. **Omnidirectional movement:** in progress. All directions, both yaw signs, combined arcs, paths and starts/stops must pass unchanged numeric gates and match the benchmark’s smoothness. Training completion alone does not close this checkpoint.
3. **Terrain and perception:** preparation continues in parallel. Full robot traversal, a frozen qualification envelope, untouched held-out layouts and causal sensing remain required.
4. **Autonomous surveying:** the final no-RTK mission is coverage of an operator-defined area while holding a steady data-collection deck on the calibrated physical robot.

## Robot and controller lineage

- **Selected C geometry:** coxa yaw-to-hip55.453538 mm in3D (47.718261 mm horizontal), femur hip-to-knee72.5 mm, tibia knee-to-distal mock +Y reference126.000008 mm. The tibia mesh spans135.750005 mm longitudinally; the126 mm value is not a material cut length or the production pad center. The full mock has19 bodies/18 joints, transferred CAD mass/inertia totaling8.26081134 kg and a1.6 N·m applied motor cap.
- **Current policy:** direct-position RSL-RL PPO,315 actor/318 critic values and18 actions. Formal comparisons retain the0.040 rad/20 ms limiter, source002/native003/host002, direct solver16/4 and the pinned actuator/external-force configuration. The reference/residual846/849 branch has different contracts and has not recovered the direct policy’s all-direction capability.
- **Production:** the31-body/30-coordinate physical four-bar model and18-motor adapter remain separate. Actuator calibration, accurate foot geometry and hardware qualification cannot be inferred from this C study. See the [C contract](experiments/c_length_study/README.md) and [physical campaign](docs/MKII_FOURBAR_TRAINING.md).
- **Current CAD proposal:** the user is modeling the foot. A separate [rigid-toe sketch](artifacts/foot_design_2026-09-10/rigid_toe_proposal_001/README.md) proposes an editable Ø30/R15 spherical contact cap, with explicit axial center/tip datums. It is unqualified and has not changed current training assets.

## Latest measured PPO results

The [matched pilot bundle](artifacts/omni_diagnostics_2026-09-09/direct_omni_matched_pilots_001/README.md) preserves both independent50-update,1,228,800-transition runs and every raw trace. Both start from original checkpoint1971… and strictly reload actor, critic, normalizer, optimizer and actions. The initial constant and stop evidence match exactly.

| Branch | Final checkpoint | Quiet passes | Worst final quiet stop excursion | Result |
|---|---|---|---|---|
| Stand/stop curriculum |32ce29dbd863…|0/48|1,887 mm|Completed acquisition; quality rejected|
| Same curriculum + CAPS |ca0545760f81…|0/48|75.1 mm|Selected for an unqualified progress preview|

The unchanged drift bound is10 mm. Every final replica has a target-step95th percentile at0.040 rad, above the0.002 rad quiet bound. Raw joint rates, interval-angle rates and requested/applied torques remain separately reported. CAPS improved most tracking cases versus the original policy, but has not established smooth standing or Stage2. Each actual learner took about97 seconds for50 updates; that duration excludes simulator startup and the multiple evaluation phases. Both pilots have all12 owned names/IDs absent and pause056/057 restoration. The full inventory records100 intermediate autosaves as remote-only rather than implying they were published.

The [latest actual recording](artifacts/omni_diagnostics_2026-09-09/direct_omni_preview_001/README.md) uses exact CAPS checkpointca054…: forward motion, left strafe, a combined arc and repeated stop requests, all at1× playback. It contains1,900 controls and950 decoded25fps frames with the full robot, command arrows and actual trail. It continues moving at stop and is not a quality pass. **The host campaign failed** because `recording/final_integrity.json` was absent after app shutdown. The native recording completed and exited0; independent input/source/checkpoint/cleanup audits passed. All28 raw files and the original failed receipt are preserved; no synthetic success receipt was substituted.

The [smoke001 failure](artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_001/README.md) and [successful integration retry002](artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_002/README.md) are separate immutable records. Native learning had created an inference tensor in a normalizer buffer; the [source-bound finalizer repair](artifacts/omni_diagnostics_2026-09-09/direct_omni_native_preparation_002/README.md) fixes strict reload without changing the two learned checkpoint bytes, dynamics, losses or gates.

## Compute and immediate next decisions

The preview has terminated, its owned container is absent, and pause058 restored ordinary forecast timers. Root then restarted the exact previously deferred halo replay under invocation9091c23f9a954fa786ca0c7009cb2ab6, with its own lock verified. The [later restoration receipt](artifacts/omni_diagnostics_2026-09-09/direct_omni_preview_001/later_halo_restore/restored.json) is a restart/recomputation proof, not forecast completion or mid-frame resume. The pre-restart weather archive remains immutable. Root remains the only GPU dispatcher, with ownership guards and shared locks required for the next allocation. The hourly reminder remains a quiet fallback.

1. Prepare one fresh, matched quiet-priority ablation: temporal coefficient 1.0 for valid consecutive exact-zero-command pairs, moving temporal and spatial coefficients 0.1, with the same denominator, original checkpoint, physics, observations and gates. Add mode-separated losses, sparse actor-gradient diagnostics and minibatch KL/rate records without changing optimizer state or RNG. A new smoke must verify the successor before a 50-update allocation. Fifty updates do not establish architectural infeasibility or convergence.
2. Verify the partner audit against exact traces and parent code. Initial independent results reject the proposed stuck-learning-rate and missing-command-redraw explanations; sustained quiet target chatter is real. Do not adopt a speculation as a bug fix.
3. Correct recording finalization in a versioned successor before reusing that launcher; preserve the failed original and complete video.
4. Keep learnable timing/foot placement ready as an independent alternative. The old fixed2-second-swing/±0.02 residual does not prove the best gait family.
5. Close Stage2 only after every numeric and visual gate, then complete the frozen development/held-out Stage3 terrain matrix with retained flat behavior and unchanged motor/contact limits.

## Terrain, sensing and parallel review

The [48 mild fixtures](artifacts/terrain_readiness_2026-09-09/runtime/terrain_mild_fixture_001/README.md) pass actual geometry/ray/PhysX/probe checks. [Full-C terrain005](artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_005/README.md) establishes entry standing only. Neither proves full robot traversal. The [CPU observation/course adapters](artifacts/terrain_readiness_2026-09-09/observation_comparison_cpu_001/README.md) pass13 targeted tests; actor/source/physics and scene integration remain explicit decisions. The [qualification plan](artifacts/terrain_readiness_2026-09-09/qualification_plan_001/QUALIFICATION_PLAN.md) still requires its final numeric envelope and untouched layouts; seeds7103/8209 are regression data.

Mid-360 and D455 are available; purchases are unrestricted. [Sequential replay](artifacts/perception_readiness_2026-09-09/sequential_sensor_replay_001/README.md), [footprint reacquisition](artifacts/perception_readiness_2026-09-09/footprint_reacquisition_001/README.md) and [support supervision](artifacts/perception_readiness_2026-09-09/support_supervisor_001/README.md) are synthetic preparation. Hypothetical mounts are not approved hardware. The250 ms optical map lease remains incompatible with unqualified multi-second swings/stopping without a demonstrated reacquisition/uncertainty contract; no lease was lengthened to pass it.

The [32-work primary-paper review](artifacts/project_review_2026-09-10/legged_policy_literature_001/REPORT.md) is complete. Claude Code Fable5.1 consultations now explicitly use `--effort max`; architecture, PPO implementation, terrain and foot-shape reviews returned actual responses. The [published partner review and independent counterexamples](artifacts/project_review_2026-09-10/claude_partner_review_001/README.md) separate verified findings from rejected speculation. Root and independent agents verify findings instead of deferring to the partner. A broad PPO audit exhausted two output limits before a narrower response; those attempts remain preserved. No unsupported recommendation changes a gate or qualifies performance.

## Earlier branches and unresolved measurement fidelity

The [cold direct baseline](artifacts/omni_diagnostics_2026-09-09/direct_omni_cold_001/README.md) retained useful cardinal motion and yaw but failed standing quality. The [reference ten-update allocation](artifacts/omni_diagnostics_2026-09-09/reference_learning_ppo_train_001/README.md) failed recovery after nine updates at1.62764847 N·m requested demand; no decision010 or final quiet screen exists. [Moving PPO001](artifacts/omni_diagnostics_2026-09-09/reference_moving_ppo_001/README.md) failed support before learning and then hit a reset inference-tensor error. Earlier quiet reference stepping is very slow:0.005 m/s and roughly13.8 seconds per six-leg cycle. It is not the desired Stage2 PPO.

Reported joint velocity can integrate far from actual joint-angle change; the [origin study](artifacts/omni_diagnostics_2026-09-09/origin_rate_002/README.md) does not establish a repair. This affects observations, rewards and quiet/energy interpretation, not just speed reporting. Raw channels, independent angle increments and unchanged scoring remain mandatory. Neither a standing pass nor learner integration qualifies native velocity fidelity or hardware dynamics.

Earlier failures and exact evidence remain in the [diagnostic index](artifacts/omni_diagnostics_2026-09-09/README.md) and dated [handoff history](HANDOFF.md). Direction and contracts are maintained in [PLAN](docs/PLAN.md); [NEXT_RUNS](docs/NEXT_RUNS.md) prepares future campaigns without launching them. Every change must follow the [live research poster contract](docs/PROJECT_SITE.md): major checkpoints first, restrained academic layout, explicit evidence and an append-only central update record, validated and rebuilt on every push.
