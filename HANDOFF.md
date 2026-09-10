# Physical MKII handoff

The earlier training history remains unchanged in
[the archived 2026-08-26 handoff](docs/archive/HANDOFF-2026-08-26.md).
Its mock-robot checkpoints and joint table do not describe the physical MKII.
Use [STATUS.md](STATUS.md) for subsequent execution state and
[the physical campaign runbook](docs/MKII_FOURBAR_TRAINING.md) for launch procedures.

## 2026-09-05 04:17 UTC — physical four-bar probe

Source commit: `ae1a023`. The versioned physical asset has **31 rigid bodies,
30 articulation coordinates, 18 active motors and six excluded physical
revolute closure joints**. There are no mimic constraints or passive drives.
The third motor coordinate is the pushlever pivot; the serial knee defaults
must not be substituted for its per-leg CAD offsets.

The following tree order was observed in Isaac Sim, directly from the passing
probe's `joint_names` array. It is evidence of this import, not an index contract
for future imports; runtime mapping remains explicit by name.

```text
lf_coxa_yaw, lm_coxa_yaw, lr_coxa_yaw, rf_coxa_yaw, rm_coxa_yaw, rr_coxa_yaw,
lf_femur_pitch, lm_femur_pitch, lr_femur_pitch, rf_femur_pitch, rm_femur_pitch, rr_femur_pitch,
lf_tibia_lever_pivot, lf_tibia_pitch, lm_tibia_lever_pivot, lm_tibia_pitch,
lr_tibia_lever_pivot, lr_tibia_pitch, rf_tibia_lever_pivot, rf_tibia_pitch,
rm_tibia_lever_pivot, rm_tibia_pitch, rr_tibia_lever_pivot, rr_tibia_pitch,
lf_tibia_rod_pivot, lm_tibia_rod_pivot, lr_tibia_rod_pivot,
rf_tibia_rod_pivot, rm_tibia_rod_pivot, rr_tibia_rod_pivot
```

The observed `active_motor_names` array matches the canonical policy order:

```text
lf_coxa_yaw, lm_coxa_yaw, lr_coxa_yaw, rf_coxa_yaw, rm_coxa_yaw, rr_coxa_yaw,
lf_femur_pitch, lm_femur_pitch, lr_femur_pitch, rf_femur_pitch, rm_femur_pitch, rr_femur_pitch,
lf_tibia_lever_pivot, lm_tibia_lever_pivot, lr_tibia_lever_pivot,
rf_tibia_lever_pivot, rm_tibia_lever_pivot, rr_tibia_lever_pivot
```

For this observed order only, those 18 motors map to zero-based tree indices
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]`.
The six knee and six rod coordinates remain passive. Positive navigation
forward is body `-Y`; positive navigation left is body `+X`.

Remote source and evidence:

```text
/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source
/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/campaigns/fourbar-campaign-20260905T041751Z-44dac287
  /probe/hexapod-fourbar-validate-20260905T041751Z-041d2072/report.json
```

That report **passes the short physical probe**: one environment, 100 control
steps and 400 physics substeps. Resolved settings are TGS (`solver_type=1`),
64 position / 1 velocity iteration, external forces applied every position
iteration, 5 ms physics and decimation 4. Settled support is six pads, with no
non-foot ground contact; peak settled applied torque is 0.6684 N·m and maximum
closure-point separation is 2.625 µm. Startup is reported separately: peak
applied torque 1.9154 N·m and maximum closure separation 9.161 µm.

At this snapshot, full 32-environment qualification remains pending: 1,000
standing control steps plus 2,400 driven control steps (4,000 + 9,600 physics
substeps), followed by the refined 128 position / 1 velocity iteration check.
Both recipes enable external forces every iteration and retain the original
closure limits. The short probe explicitly records
`simulation_training_admission=false`; it does not establish that PPO has
started or that a walking policy exists.

This remains simulation qualification. The bounded RS05 model needs hardware
calibration; it is not a measured thermal model or a CAN current controller.
Primitive collider fit has material edge and side errors, including sampled
foot-pad discrepancies of several millimetres; see the
[quantitative collider audit](artifacts/mkii_fourbar_2026-09-05/collider_fit/README.md).
Hardware transfer, precise terrain contact and rough-terrain locomotion are
not qualified by this probe.

## 2026-09-05 07:02 UTC — Wi-Fi continuation and exact asset binding

No physical-model PPO has started. Full campaign004 failed closure; completed short v3 group diagnostic055416 stayed below closure bounds but briefly reported zero support. Candidate060625 was interrupted by the host regex misclassifying a weather `flock` waiter. Preserve all failed/interrupted evidence; no generated replacement primary reports.

Current priority guard1407273 holds `/opt/wx/gpu.lock`; status/release/campaign selection files are in `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/priority_20260905T0700/`. User explicitly reaffirmed top priority over weather. Guard expires17:02:04UTC at the latest. Live checks remain necessary.

The new release records actual selected physical bundle identity and requires matching CPU/Kit/runtime/source dependencies, nominal/refined identity, and exact pre-learner admission. CAD kinematics JSON and v3/v4 geometry remain unchanged. Default remainsv3; full v4 runs require explicit selection. Continue candidate comparison, full32×1000standing+2400driven nominal/refined, then scratch and full PPO only if admitted.

## 2026-09-05 16:16 UTC — v5 physical coupling preparation

Complete D6 group diagnostic `072121Z-16d523c4` failed (0.505577 mm closure; raw42.0734 N·m, applied5.5 N·m). Source remains0d1ceab on Spark. Native physical-coupling v5 candidate is separate; see `docs/MKII_PHYSICAL_COUPLING_CANDIDATE.md`. It retains30tree coordinates and31bodies, replacing6external closures with12bilateral internal constraints. Defaultv3 remains; explicitv5 will run the unchanged full admission campaign. Three parallel follow-up reviews hit the account usage limit, so the primary agent continued locally. No physical-model PPO has started.

## 2026-09-05 17:48 UTC — complete native-coupling nominal failure

Campaign 005 completed the full32 ×1000 standing +2400 driven sequence with v5 native bilateral mimic constraints. It failed closure (0.108253 mm), support (minimum0) and simultaneous LM knee direction (−0.00476402 rad); all18 individual direction tests passed. No resets/non-foot contacts; applied peak5.5 N·m, raw16.6328 N·m. Primary evidence is `artifacts/mkii_fourbar_2026-09-05/campaign_005_physical_mimic/`. No PPO started. Sourcecfe0cf5 remains frozen on Spark; test-onlyffa44f6 passed CI and has the same functional identity.

Bounded diagnostic174632Z-36232e95 now compares v5 at128/1, eight environments,200 standing +700 group-motion steps, with full per-substep traces. Priority reservation1465763 (directory `.../mkii_fourbar_diagnostics_v1/priority_20260905T1616/`) is still active, maximum expiry2026-09-06T02:16:34Z. Do not modify the source or shared note during its owned run.

## 2026-09-05 18:20 UTC — scheduled active targets

Short v5/128 group trace completed with21.017µmmaximumclosure, all18directionspositive, four1.25mssupportgaps atsecondnegativekneestep. Compact primary/trace evidence is `artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/v5_groups128_20260905T174632Z/`.

Source9cd8d4c introduces sixteen-step linear interpolation of the existing50Hz active motor endpoints at800Hz, zero velocityfeedforward, unchangedgains/physics/assets/gates.854tests pass;286filemanifest97ee8246a7849cb805ad65f40a3144a2f47a4746e4d74deaa9b862c3750729a6. Frozen source `/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/source`, identityc53071afdf320f6a9d6f91de09ddc6102de74df6f45a203e3a8166a8686548ca. Bounded8envgroupdiag181945Z-ec78eb1e at64/1 isrunning; noPPO. Successfulshortdiagnostics stillrequire full32envnominal/refined qualification.

## 2026-09-05 19:05 UTC — clean diagnostic and full campaign

Ramped-target diagnostic 181945Z-ec78eb1e completed with no physical gate errors,
all 18 group directions passing, at least four driven supporting feet, 25.073 µm
maximum closure and 4.29035 N·m peak torque. Exact sixteen-step scheduling is
verified in the compact `ramped_groups64_20260905T181945Z` evidence directory.

Campaign `fourbar-campaign-20260905T185810Z-c36097a7` is running, PID 1532490;
probe passed and full nominal is underway. Source remains frozen at 9cd8d4c in
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/source`. The current good SSH
socket is `/tmp/hexapod_fourbar_live.sock`; the previous recovery socket timed out.
Inspect live state before acting. No PPO yet. Priority guard 1465763 still holds
the weather scheduler lock, with maximum expiry 2026-09-06T02:16:34Z.

## 2026-09-05 19:29 UTC — first full nominal pass

Campaign 006 nominal `185936Z-b545949f` passed all 32 × 3,400 control steps and
all individual/group directions. Driven closure 0.0819092 mm, passive residual
0.000996530 rad, minimum support three feet, no resets or non-foot ground
contacts. Raw demand peaked at 14.32394 N·m; applied torque stayed within its
5.5 N·m envelope and burst budget. Primary evidence is in
`artifacts/mkii_fourbar_2026-09-05/campaign_006_ramped_targets/`.
The same campaign has advanced to refined validation automatically; no PPO yet.
Keep source 9cd8d4c frozen and inspect campaign JSON for the active phase.

## 2026-09-05 20:53 UTC — batch investigation and GPU isolation repair

No physical-model PPO has started. Campaign006 refined failed; matched Kd0.30
standing checks also failed convergence. Frozen d6d5863 one-robot diagnostics at
origin,+6m X and(2,-2)m complete at0.667–0.681Nm settled, all six pads supporting.
The same-source eight-robot trace at128/1 is now running from
`/home/orionh/HEXAPOD_runs/mkii_placement_diagnostics_v1/batch8_20260905T205224Z`
(PID1616777), to match row0 to the diagonal single exactly. Never infer precision
as the cause from the earlier location correlation alone.

Fixd863663 on GitHub adds explicit GPU/CPU clone collision groups and topology
verification (installedLab3 disabled envIDs and this manual scene skipped auto
filtering). A separate host fix tolerates only exact missing-foreign-container
Docker responses; unknown errors still block.885CPUtests passed65.501s; new290-file
manifest `mkii_fourbar_v1_collision_isolation_pipeline.sha256` SHA256
`edeca5d58572f9fde4354648e68b736241bb52181d956a887936f1b631123b8d`.
Staging new isolated source `/home/orionh/HEXAPOD_runs/mkii_collision_isolation_v1/source`;
live backend overlap proof still pending. Existing sources/manifests untouched.
Good SSHsocket `/tmp/hexapod_fourbar_translation.sock`. Priorityguard1465763 remains
bounded until2026-09-06T02:16:34Z; inspect live status before acting. Avoid transient
CPU Docker readers during the older frozen supervisor's active jobs.

## 2026-09-05 21:20 UTC — collision proof and targeted velocity solve

No physical-model PPO. The single robot at (−2, 0) m reproduces noisy batch row 7
exactly; (2,−2) and origin controls also match their own batch rows. Filtered and
unfiltered spaced-world NPZ bytes are identical. Additional robots are not
required for the transient. Its post-contact mimic velocity residual reaches
4.425369 rad/s and C-pin relative speed about 0.3625 m/s despite small position
error. These are diagnostic findings, not new silently chosen tolerances.

External overlap fixture v2 (4d62eca) completed both native GPU controls on
frozen source d863663: filtered body pair zero contacts/force, negative pair
340 contacts and 19690.2949 N peak, independent ground support in both filtered
robots, all 24 cloned mimic references valid. Both exact containers were removed.
Attempt 1's UInt32 diagnostic-code failure remains preserved. Successful evidence:
`artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/live_controls_attempt_002/`.
This establishes measured body-pair collision response, not full-body training admission.

Candidate source83a9bca uses four final velocity iterations, retaining all other
physical/controller settings and gates.886 CPU tests passed64.064s. Manifest291
files SHA256 `5f786620e955424fa0196b471f00609ec1b9f5135096bd11298ba8f3a3475ba7`.
Frozen source `/home/orionh/HEXAPOD_runs/mkii_final_velocity4_v1/source` is an exact
Git archive of those291 files plus the manifest; prior run logs are excluded,
all runtime identity files retained. Functional identity
`f825a1fb3cfcf33d27dae217cb29aa777a5dd5e0bed96819fa48175a97e7d694`.
Diagnostic `/home/orionh/HEXAPOD_runs/mkii_final_velocity4_v1/standing_refined_20260905T211904Z`
PID1635933 is running8 environments ×200 standing control steps at128/4.
Read live state before acting; source and shared note must remain unchanged.
Additional full-validation velocity telemetry is being developed separately.

## 2026-09-05 21:51 UTC — sixteen velocity iterations and larger standing comparison

The completed eight-world comparison is preserved in
`artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity16_comparison/`.
One → four → sixteen final iterations reduce worst settled passive velocity
residual 4.425369 → 1.104570 → 0.097386 rad/s, and pin speed 0.362515 →
0.020240 → 0.003487 m/s. Sixteen is not uniformly better by position: row 2's
velocity residual and row 4's torque/support regress versus four iterations.
Do not escalate iteration count blindly or infer training admission. The new
per-substep velocity telemetry was independently reproduced from raw traces;
startup remains separate. No physical-model PPO has started.

Frozen source c804169 at `/home/orionh/HEXAPOD_runs/mkii_final_velocity16_v1/source`
has functional identity `d3442002687f4ff7b34bd2e24134a8e3265a87e05221d3cbf33f2630c0b52d36`.
Its separate 292-file manifest is
`isaaclab/deploy/mkii_fourbar_v1_final_velocity16_pipeline.sha256`, SHA256
`6598f8868f8a02240847798bed47cde0700010c28a34c21af0e23d72bbf55319`.
891 local CPU tests passed in 80.730 seconds; GitHub CI 33993230661 succeeded.
The source is an exact lean Git archive of manifest files plus the manifest;
prior runtime reports remain outside source. Do not modify the frozen directory.

Host PID 1646773 launched the matched 32 × 600-control-step standing pair at
21:51:19 UTC. Output directory:
`/home/orionh/HEXAPOD_runs/mkii_final_velocity16_v1/standing_pair_20260905T215119Z`.
The external `standing_pair_launcher_v1.py` invokes the ordinary owned supervisor
at 64/16 then 128/16, 1,200-second bound each, stops on phase failure and cannot
admit training. Compare actual root placements, complete runtime identity and
both old convergence metrics and new velocity telemetry. The 2.4–12 s settled
window matches the previous 32-world Kd 0.30 standing experiment. Live state is
in `pair.json`; do not edit source or shared note during either phase. Priority
guard 1465763 remains bounded until 2026-09-06T02:16:34Z. SSH socket:
`/tmp/hexapod_fourbar_translation.sock`. Inspect live state before acting.

Successful GPU body-pair overlap evidence, the isolated (−2,0) replay and the
four-iteration comparison are now preserved beside the sixteen-iteration result.
The overlap negative control's 340 is the maximum native contact-point count
per directional body query/sample, not the total collision count across the run.

## 2026-09-05 22:03 UTC — qualification review and staged release

Source `fd34f661bd672df9f68e6d8568548ad092ef32cf` strengthens the admission
comparison without changing dynamics. It requires finite, exactly matching
ordered actual reset-root positions; derives truthful solver descriptions;
and compares raw pre-envelope demand using the existing 0.05 N m / 5% torque
criterion so identical clipped peaks cannot hide different requests. The raw
comparison is not an absolute demand cap or proof of trajectory convergence.
The known velocity metrics remain observational and require explicit review.

898 local CPU tests passed in 74.366 seconds; GitHub CI 33994676204 succeeded.
The first suite run found two older campaign fixtures missing the newly
required placement/demand fields; only those fixture inputs were updated,
then the full suite passed. No running Spark source was edited.

Frozen next source: `/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1/source`.
Functional SHA256 `1fcab03b2c9f810a931e3d65fd057312d6dfb3b92003ca30058ccd821ff965e6`.
New 293-file manifest `isaaclab/deploy/mkii_fourbar_v1_placement_convergence_pipeline.sha256`,
SHA256 `3e64b8b0a7cb30ee3a78ae37f28ae7644e719d7d42dcb7d7af1c4f9765e2ae01`.
The lean Git archive contains those 293 files plus the manifest; all hashes
verified on Spark. Prior published manifests remain unchanged. This source
is staged, not yet physically admitted. The ongoing c804169 standing pair
continues independently; it cannot admit fd34f66 or any PPO run.

## 2026-09-05 22:10 UTC — standing comparison passed; full campaign launched

The matched c804169 pair completed both 32 × 600 controls with exact source
and placement matching and clean owned-container removal. Settled applied/raw
peak torque delta is 0.00353038311 N m (bound 0.05), height delta 18.1112497 µm
(bound 1 mm), and both maintain at least five supporting feet. Passive velocity
peaks 0.692814/0.714486 rad/s and pin speeds 0.0127074/0.0125607 m/s remain
observational; RMS decreases in the refined run. All original short physical
gates pass. This is standing screening, not complete training admission.
Evidence: `artifacts/mkii_fourbar_2026-09-05/final_velocity16_standing_comparison/`.

Fresh full campaign PID 1667897 started at 22:09:44 UTC on frozen fd34f66:
`/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1/campaigns/fourbar-campaign-20260905T220944Z-78b3e50d/campaign.json`.
It runs fresh 1×100 probe, full32×1000 standing +2400 driven nominal/refined,
64×3 scratch PPO with checkpoint/inference checks, then separate512×1000
resumed PPO only if admitted. Phase execution bounds are7200 seconds and
full training21600 seconds. No prior report is admitted under the new identity.
Read live campaign/progress files before claiming a phase has started or passed.

The former guard1465763 was identified by exact /proc arguments and stopped
only after successor1667860 queued on the same scheduler lock. No GPU workload
was interrupted. Successor status/release/campaign selection:
`/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1/priority_20260905T220942Z/`.
It expires2026-09-06T08:09:43.812781Z at the latest and is bound to this exact
campaign's terminal state. The shared coordination note was appended before
launch; do not edit its bytes during an active phase. Read `campaign.json`
and the priority status if Wi-Fi is interrupted; never launch a duplicate.

Bootstrap and launch record reside in the same remote run parent:
`launch_reviewed_campaign_v1.py` and `campaign_launch_20260905T220942Z.json`.
The unchanged-source first PPO throughput must be measured before forecasting
full training duration; see `docs/MKII_RUNTIME_PROFILING.md`. No runtime
optimization has been applied or justified solely from GPU utilization.

## 2026-09-06 02:08 UTC — interrupted campaign and user-authorized takeover

Campaign007 stopped because coordination changed, not a recorded model failure.
Last flushed nominal step600/1000 was22:18:24.289578780UTC; first stop marker
mtime22:18:56.719657UTC; campaign terminal mtime22:19:22.910692UTC. No nominal
final report, driven progress, refined phase or PPO exists. Original evidence:
`artifacts/mkii_fourbar_2026-09-05/campaign_007_coordination_yield/`.
The prior reservation1667860 released22:19:23.982830UTC.

The user explicitly instructed full takeover and push. Queue1706414 identity
was verified by recorded start ticks and exact /proc script arguments, then
sent SIGTERM; its own handler stopped its GPU child group. Outputs preserved.
At02:08:18UTC, that parent and all GPU processes were gone. Successor bounded
guard1711343 holds `/opt/wx/gpu.lock`; status/release files:
`/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/priority_20260906T020649Z/`.
It expires2026-09-06T12:06:50.120523Z at the latest; attach the next exact
campaign via campaign_path.txt once launched. Shared note now records fresh
exclusive user priority and HEXAPOD_SHARE_STATUS=NONE. No MPS/concurrency.

Sourcefd34f66 and interrupted evidence remain frozen. The next supervisor
release fixes prose edits causing false handoff requests while retaining
explicit/invalid control handling and all resource/physics gates. It needs
its own source manifest, tests, immutable staging and fresh full validation.
Takeover evidence is `artifacts/mkii_fourbar_2026-09-05/coordination_recovery_20260906/`.

## 2026-09-06 02:22 UTC — coordination fix and fresh campaign 008

Release `1239159c185cd504c359bb20e98bde9986acbbb4` separates the canonical
sharing control from status-note prose. Exactly one valid `NONE` permits work;
`REQUESTED` or malformed/missing/duplicate control still latches a controlled
yield. The runner rechecks immediately before barrier release and during
execution. Training retains checkpoint grace and resource checks. Historical
frozen runners retain their old behavior; concurrency/MPS is not enabled.

906 local tests passed in 70.128 seconds; GitHub CI 34005889807 succeeded.
The 294-file release manifest is
`isaaclab/deploy/mkii_fourbar_v1_coordination_control_pipeline.sha256`, SHA256
`b75f8ca4d3045e058c9e2eb27deebea9e08aa789b56f50fbf7fd215f6960a1f8`.
Functional identity:
`c06e56ed68508f944363bfe6564a13d0760066e4a338164b664214edb695fb5a`.
Frozen source: `/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/source`.
The exact Git archive and all staged manifest entries were verified on Spark.

Fresh campaign PID 1712641 launched at 02:15:07 UTC:
`/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/campaigns/fourbar-campaign-20260906T021507Z-9d668815/campaign.json`.
Its 1 × 100 startup probe passed, report SHA256
`2a85e74824a8fc28f8f74907e1324e94db5fa2728681d6685115dfe9bb35b9b2`.
Nominal supervisor 1716170 began at 02:17:40 UTC in
`nominal/hexapod-fourbar-validate-20260906T021740Z-96e3036c/`.
The 02:22 UTC snapshot had reached standing step 200/1000 with no recorded
invalid samples. This is partial progress, not nominal admission. All full
standing/driven, refined convergence and checkpoint/inference gates remain
required before the campaign's separate 512 × 1,000-update PPO stage.

Guard 1711343 is now bound to this exact campaign through `campaign_path.txt`
in `priority_20260906T020649Z/`. It releases on the campaign's terminal state
or at 12:06:50 UTC at the latest. No competing CUDA workload was present at
launch. Do not replay the bootstrap or launch a duplicate after Wi-Fi loss.

A live prose-only shared-note update at 02:21:46 UTC was observed by this
nominal supervisor at 02:21:52 UTC, without a pause, stop marker or container
interruption. Evidence:
`artifacts/mkii_fourbar_2026-09-06/live_coordination_prose_check/`.
Release, launch and completed probe evidence:
`artifacts/mkii_fourbar_2026-09-06/coordination_control_release/`.
Read live campaign and supervisor state before claiming further progress.


## 2026-09-06 03:25 UTC — completed motion failure, full compute cleanup and exact replay

Campaign 008 completed all 32 × 3,400 controls at 03:09:38 UTC and failed:
C-pin separation 0.213821 mm, zero-support samples and four direction-response
scores below the existing bound. Raw demand peaked at 85.207611 N m; applied
peak remained 5.5 N m. No resets or non-foot contact occurred. The raw report,
source identity, completed log and exact owned-container removal are preserved
in `artifacts/mkii_fourbar_2026-09-06/campaign_008_motion_failure/`. No refined
validation or physical PPO followed. Aggregate maxima do not identify the
failing environment or cause.

The user explicitly requested removing competing processes, continuing until
PPO trains, finding a different approach when evidence shows a dead end, and
capturing a video of the resulting policy. All 21 identified weather CPU
processes, including the exact materializer service, were stopped through
verified SIGTERM/service actions at 03:13:57–03:14:00 UTC; outputs were not
deleted. Evidence: `artifacts/mkii_fourbar_2026-09-06/cpu_takeover_20260906T031357Z/`.
Spark then had 118 GiB available memory and no CUDA producer. Successor guard
1772925 holds `/opt/wx/gpu.lock`, bounded until 13:10:41 UTC. Its directory is
`/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/priority_20260906T031041Z/`.
The shared coordination note records the fresh priority and stopped jobs.

The new diagnostic-only `validation_prefix` replays 1,000 standing controls
and the first 15 individual motor tests through LR: 2,500 controls / 40,000
physics steps, same 32 environments, seed, reset and episode horizon. Physical
metrics and force-write counts cover all substeps. Detailed traces cover
controls 2200–2499 (LF/LM/LR), while compact control telemetry and per-env
end-hold means recover the discarded response information. Float32 response
reduction matches the validator; no physical parameter or gate changed.

914 local CPU tests passed in 61.846 seconds, with independent replay review.
The separate 295-file manifest is
`isaaclab/deploy/mkii_fourbar_v1_motion_prefix_pipeline.sha256`, SHA256
`22e3429f14cd42feace994b48d18b0ba7d342ebaa22e3c2b556cdd57f5141706`.
Both archived and current lineage checks pass. This is source verification,
not physical admission. Compare actual reset positions, runtime parameters
and earlier phase responses against campaign 008 before calling it an exact
native-dynamics reproduction. Live recovery state is
`/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/recovery.json`.


## 2026-09-06 11:40 UTC — exact failure replay, collision controls and candidate campaign

The motion-prefix diagnostic completed all 32 × 2,500 controls at approximately
03:50:35 UTC: 1,000 standing plus the first 15 individual motors through LR,
40,000 physics samples and force writes per environment. It retained all 32
original grid placements and the full preceding action history. Both standing
windows, all 15 primary response minima, maximum C-pin separation
0.213820967474 mm, raw demand 85.207611083984 N m, passive velocity residual
221.326507568359 rad/s and relative pin velocity 7.051413536072 m/s match
campaign 008 exactly. LM/LR direction, closure and support checks still fail;
applied torque stays within its envelope. No reset or non-foot contact occurred.
The native replay is completed diagnostic evidence, not training admission.

Compact original evidence and the read-only verifier are preserved in
`artifacts/mkii_fourbar_2026-09-06/prefix_replay_001/`. Its six detailed trace
NPZs and 31 control-boundary NPZs remain on Spark in
`/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/prefix_replay_001/hexapod-fourbar-diagnose-20260906T032844Z-d73ff362/`.
The remote inventory records exact paths, hashes, sizes and modification times;
the local SHA manifest does not claim to contain those raw bytes. Detailed
coverage is controls [2200,2500), or physics [35200,40000), all 32 environments.
The replay used commit 3011b0f and functional identity
`9fa39026e79ca4d327d67f58c37b9e2f800700e627994b51b3640bb53d106089`.
The first traced LF-femur velocity impulse precedes clipping; subsequent
artifact analyses preserve the chronology without claiming a unique cause.

Release `ae4f38828181b33c9fc4fd2b8f9d68b03e6c31f1` adds the explicit
`coincident_flat_origin_v1` flat-world layout; `grid_2m_v1` remains the default.
Scene and terrain spacing are set to zero before construction. Actual USD
transforms, native articulation/sensor rows, all authored collider ownership,
all twelve mimic references per robot and exact per-env reset readbacks are
verified. Layout selection is bound to admission and checkpoint runtime
identity. Robot geometry, physical motor bounds, gains, 800 Hz physics /
50 Hz policy timing and existing full physical gates are unchanged.

926 CPU tests passed. The separate 297-path release manifest is
`isaaclab/deploy/mkii_fourbar_v1_coincident_layout_pipeline.sha256`, SHA256
`7c15d6ce4e6f5c516a361d74d7f062ae78a075450a2e123f073f1b535d0d757d`.
Functional identity is
`8fb32bc3e39642338bfbcf37bcc3a7e58d1ab0b8c901174532234248644a93e9`.
Frozen source is `/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/source`.
Artifact-only overlap tools were added in 0ae0607 without changing that identity.
Prior release manifests and failed evidence remain unchanged.

The candidate's one-environment / 100-control startup probe passed. Supervisor
PID 1805891 launched the separate full-body overlap controls at 04:32:03 UTC in
`/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/overlap_controls_001`.
The filtered control passed. The original negative control hit its 600-second
limit without GPU execution/output; this infrastructure failure is preserved.
Negative retry PID 1842070 launched at 11:19:35 UTC and completed successfully:
every source body was excited, with none unexcited. The combined successful
pair report is `/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/combined_overlap_001/pair_report.json`,
SHA256 `3d67408fac04bc2f1484018806a7c3ea40efb260b85107a721c3f754b9f10e34`.
Each case uses a fresh Kit process, two 31-body robots, 62 exact native views and
31 foreign filters per view: 1,922 directed queries. Every source body seeing
foreign contacts does not imply every possible body pair was physically excited. The negative case explicitly
allows inter-environment contacts before physics initialization. Both controls
must retain shared-ground support, exact source/layout identity, finite native
UInt32-to-int64 counts, complete samples and unsaturated contact buffers.
Neither case nor their pair report can grant training/hardware admission.

Fresh campaign PID 1845574 launched at 11:40:47.559716 UTC, with output in
`/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/campaign_001/fourbar-campaign-20260906T114047Z-73b4f841/campaign.json`.
It reuses the verified 100-control startup probe, then requires full nominal and
refined standing plus every individual/group motion check before 64×3 scratch
PPO, checkpoint/inference validation and separate 512×1,000 resumed PPO.
PPO had not started at the launch snapshot. Read the actual campaign/progress
files before asserting a later phase or launching any duplicate.

Current guard PID 1845570 was renewed in
`/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/priority_20260906T114047Z/`
and expires at 21:40:47.521619 UTC at the latest. Root owns the active GPU
supervisor and reservation. The user removed the progress automation; it was
confirmed absent and was not recreated. Actual policy video remains downstream
of the admitted learner. No inference of terrain/hardware readiness follows.

## 2026-09-06 — coincident full nominal failure and recovered pause

The full coincident-origin nominal finished at12:10UTC, failing only the unchanged 0.100mm C-pin separation bound (observed0.111171183mm). All36 direction responses pass; maximum raw/applied4.423689Nm, at least3 loaded feet, no resets/non-foot contacts/nonfinite samples. No refined run or PPO started. Six original campaign/report/supervisor/source-inventory/audit/log files are preserved under `artifacts/mkii_fourbar_2026-09-06/coincident_nominal_failure_001/`, verified against remote SHA-256 values.

The shared Spark file contains a newer13:33UTC user instruction from the weather task: pause HEXAPOD pending explicit resume and never hold a long-lived exclusive reservation. Guard1845570 released13:33:47UTC. Root recovered this state at22:51–22:55UTC; preflight blocked on canonical REQUESTED before creating a new output directory/guard/GPU job. No launch occurred. The five-minute progress automation remains removed. Historical full-priority notes do not override the newer pause. Current work continues locally with actual RSL5.0.1 CPU runner/checkpoint integration and an official-reference CAD export audit.

The corrected RSL-RL release has functional identity `d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54`, a new298-path manifest SHA `418cb5cf1ccc10729407e19e0c2962b83736c6ed17c9b6dd49bbac041c499868`, and931 passing repository tests (75.778s). Both actual CPU API fixes pass strict save/load and continued optimization in the same and a fresh runner; this is synthetic API evidence, not physical robot PPO. New CAD audit documents the pad-envelope undercoverage, reconstructed export structure and official OpenArm/SO101 reference differences. Capture v2 and the one-shot follower have33 focused tests; neither is deployed.

## 2026-09-06 23:12 UTC — explicit user resume

The user replied “take full training priority” to the prepared bounded resume. Shared canonical control changed REQUESTED→NONE with the new policy at the top and prior bytes preserved. Source5d476d4 was already staged and all298 paths verified. Supervisor1959403 (startticks105221947) launched a complete32-robot/1000-standing+2400-driven refined128/16 comparison. `/usr/bin/flock --nonblock --no-fork` holds the existing `/opt/wx/gpu.lock` only for this supervisor lifetime; no reserve_workflow process exists. Exact output/container and launch record are in STATUS and `rsl501_release/refined_launch_001.json`. Actual CUDA1959662 and standing100 were observed. No PPO yet.


# 9 September 2026: user-selected C-study and omnidirectional Stage 2

The following is the separate C-study history. The newest current state is [STATUS.md](STATUS.md). It does not relabel physical four-bar failures or archived checkpoints.

## 2026-09-09 — mock geometry length study (separate from training lineage)

At the user's request, `robot/hexapod_mkii_length_study/` uses the archived
mock's geometry with anatomically registered masses, COMs and full inertia
tensors from the current CAD serial URDF. The grid is 7 × 7: femur and tibia
50–110% in 10% steps, including the unchanged-length baseline. Coxa and mounts
are fixed. Intrinsic link tensors and masses are fixed across this geometry
ablation; longitudinal COM offsets follow length. See the study README and
manifest for the assumptions, precise mapping and hashes. Do not switch the
current CAD training task's USD path to these mock-convention assets.

Tailscale SSH to `spark` (100.82.166.9) verified. After the unrelated StormScope
CUDA job exited, the user instructed us to take the GPU immediately. All work
uses bounded per-job locks; no legacy training service, persistent reservation
or background queue was started. Isolated source and results are under
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/`.

The old `_urdf.ImportConfig` API is absent on the installed Isaac Sim 6 build.
`tools/simulate_length_study.py` now uses Isaac Lab's `UrdfConverter` with the
new importer. It explicitly authors source tensors onto new study USDs and
checks the tensor round trip with OpenUSD's own vector rotation. All 931 link
tensors pass (49 robots × 19 bodies); the separate synthetic CPU test also
passes, with maximum absolute tensor error 1.63e-9 kg m² or less.

The baseline smoke test (`smoke_003`) and first 49-robot batch (`batch_001`)
completed 1,000 physics steps at 400 Hz with finite joint states. This is a
standing comparison using explicit RS05 position control, **not PPO or a
training acceptance gate**. All 49 had the following observed joint order:

```text
revolute_1_1 revolute_1_7 revolute_2_5 revolute_3 revolute_4 revolute_5
revolute_1 revolute_1_6 revolute_1_5 revolute_1_3 revolute_1_4 revolute_1_2
revolute_2 revolute_2_6 revolute_2_4 revolute_2_2 revolute_2_3 revolute_2_1
```

Within each joint family that corresponds to `lf lr lm rr rf rm`; it is an
observation tied to these imports, never a runtime index contract. The runner
resolves positions by name and saves each articulation's actual names/order.
Batch 001 maximum applied torque was 1.600000024 N m (float32 tolerance),
maximum raw computed demand 3.632945 N m. This does not establish low continuous
duty, contact correctness or gait quality. Some common-posture static candidates
are explicitly flagged for insufficient non-foot clearance. The repeated
camera capture is `batch_002`; its final state and image are in
`artifacts/length_study_2026-09-09/`. No asset has been admitted to training by
this task.


### 2026-09-09: walking comparison queued; hardware decision scope expanded

The user requested walking policies for all sizes, an illustrated implementation
writeup when finished, and investigation of credible missed/out-of-bound
femur/tibia solutions. Coxa remains fixed. The original 49-size fixed-inertia
mock study is now explicitly only the first screening stage. Read
`artifacts/length_study_2026-09-09/DECISION_PROTOCOL.md` before continuing.
There is no hardware recommendation and no admitted/trained policy yet at this
entry's creation.

New CPU stance search `robot/tools/prepare_length_study_training.py` writes
`robot/hexapod_mkii_length_study/training_plan.json`. All 49 sizes have at least
one eligible static candidate with 0.20 rad action headroom, predicted hold
load <=1.3 N m, nonfoot clearance >=5 mm and root height >=70 mm. These are not
Isaac admission results. `tools/train_length_study.py` adapts the existing
HexapodEnv process-locally to mock link names and nested contact paths. It uses
400 Hz physics, 50 Hz control, 16/4 solver iterations, unchanged RS05 parameters,
exact transferred mass and fixed friction. Each size must pass 32x1000 control
steps of standing before scratch PPO (256 envs, 500x24 updates, seed57) and
matched 64-env tests at 0.1/0.2/0.3 m/s. `tools/rank_length_study.py` publishes
only completed evaluations and a constrained Pareto shortlist, not a made-up
weighted score or an implementation verdict.

Training entry point is `isaaclab/deploy/hexapod-rl length-study --source ...
--output ...`, implemented by `tools/launch_length_training_spark.py`.
Remote frozen source:
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/training_source_v1`.
The verified batch002 USDs were copied into its package's `training_usd/`.
130 source files are recorded in `campaign_source_hashes.json`. Do not edit
that snapshot during a campaign; preserve evidence and create a new version if
runtime adapter corrections are needed.

Active remote CPU coordinator PID2943469:
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/walking_campaign_001`.
Read its `campaign.json`, then the current job's `state.json`, `run.log`, and
`failure.json` if present. It waits every30seconds without a GPU reservation,
uses both shared locks for actual jobs, stops only its owned immutable Docker
ID if interrupted/competing/low-memory, and has a48hour total bound. A
`stop.request` file stops this campaign. Baseline is first, then the other48
sizes in seeded shuffled order. Failed static/dynamic candidates are not
quietly trained. Infrastructure errors stop the campaign for diagnosis.

At 2026-09-09T17:52Z the GB10 was96% utilized by unrelated StormScope nowcast
PID2942150 (`20260909T1730Z`), which held `/opt/wx/gpu.lock`; project lock was
free. Campaign status was `waiting_for_shared_gpu`, baseline validation queued.
The user was explicitly told no walking policy was training. Do not kill the
weather process or infer the GPU is free from low reported memory use.

A thread heartbeat `complete-hexapod-leg-length-study` is ACTIVE every15minutes
for diagnosis, continuation and the eventual decision writeup. It should stay
quiet on unchanged/non-actionable state, preserve unrelated workloads, and
stop after the full investigation/report or user cancellation. Its prompt
points to campaign001; update it if the campaign moves. This is an actual app
automation, not a promise of manual background monitoring.

CPU checks passed: 288 existing Isaac task tests +11 study tests (8 geometry,
3 training-input/ranking checks), Python compilation and deploy shell syntax.
The new runtime adapter still needs its first Isaac standing import/gate; do
not report runtime success based on these CPU checks.

User inspection viewer: `robot/tools/pack_length_study_viewer.py` packs all49
exact hash-checked URDFs and shared STL meshes into a <1MB fragment. Source
`robot/hexapod_mkii_length_study/preview_template.html`; response fragment
`/Users/andreboufama/.codex/visualizations/2026/09/09/01a086f9-03d8-7861-afc6-5a4ff715a1f2/hexapod-size-inspector.html`.
It supports complete robots, individual links, all49 leg profiles, baseline
outline, angle sliders, selected-link inertials and original URDF download.
Verified via CUA at736/360px light/dark, no JS errors, correct size selection
and mass/inertia picking. It labels mesh scaling and stretched mounting holes
explicitly; no detailed production Onshape CAD was edited.


#### Forecast GPU pause explicitly authorized and executed (2026-09-09T17:57Z)

The user explicitly granted permission to pause the current Spark forecasting
job and defer it, identifying https://github.com/CornellGeoData/Forecasting-Pipeline.
This supersedes earlier instructions against interrupting that particular job.
Only `stormscope-dispatch.timer`, `stormscope-scout.timer` and their two GPU
services were stopped via systemd --user. CPU monitor/verify/publish continue.
The model had no mid-inference checkpoint or completed stage at interruption;
its input and state files were retained. No promise of checkpoint resume was
made. The input run was `scope-202609091730-180` (PID2942150).

Pause evidence, pre-pause state and restoration script live at
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/forecast_pause_001/`.
Run its `resume_forecasting.py` to restore the two GPU timers when the hexapod
GPU work finishes or is cancelled. A native systemd user backstop timer,
`hexapod-restore-forecasting-20260909.timer`, automatically runs that script
48hours later (2026-09-11T17:57Z). It restores scheduling, not the interrupted
model's memory state. Do not silently extend/remove this backstop. The app
heartbeat was updated with this authorization and restoration responsibility.

The hexapod container `hexapod-length-policy-3f0362f583b0` then acquired the
GPU and launched baseline validation in campaign001. It passed import/contact
layout setup and reached800/1000controlsteps by about17:59Z. Read the completed
admission before claiming pass. Walking training follows only if it passes.


#### Walking campaign002: RSL-RL compatibility correction

Campaign001 baseline passed the full32x1000standing gate: 18joints/19bodies/
6feet; zero falls, truncations or nonfoot contact; computed-torque saturation
fraction0.00173611 (0.174% <0.5%); post-settle maximum computed2.157N m,
max applied1.600000024N m; settled root0.211264m. Full admission is archived
locally as `artifacts/length_study_2026-09-09/baseline_walking_admission.json`.
Do not hide the short over-rating samples or treat this as final hardware proof.

The first PPO constructor then failed before any learning with
`MLPModel.__init__() got an unexpected keyword argument stochastic`. Isaac
Lab's config includes deprecated pre-v5 fields; its native train.py calls
`isaaclab_rl.rsl_rl.utils.handle_deprecated_rsl_rl_cfg` before constructing the
runner. The study now uses that same official compatibility helper with the
installed `rsl-rl-lib` version, and explicitly assigns actor/critic to policy
observations. No physics, reward, geometry or PPO hyperparameter was changed.

Preserved failed campaign001/source_v1. New source `training_source_v2` and
active `walking_campaign_002` at the same remote study root. CPU coordinator
PID2949932. It reruns baseline admission then launches PPO, retaining all49
sizes. The app heartbeat now points to campaign002. Inspect actual state/logs
before reporting learning progress. The prior forecast pause and48hour
restoration backstop remain in force.


#### Walking campaign003: GPU ownership snapshot race hardened

Campaign002 completed baseline validation successfully, then its guard stopped
at shutdown claiming a competing CUDA process. An immediate independent check
showed0% GPU use, no CUDA process, and both forecast GPU timers still inactive.
The likely cause is non-atomic nvidia-smi/docker-top snapshots: Kit's CUDA PID
can disappear before the later Docker snapshot. The old guard did not preserve
the alleged foreign PID, so that explanation remains an inference.

`live_competitors` now ignores vanished processes, independently recognizes a
live process's exact owned Docker cgroup, and stops for every other live CUDA
process. On an interruption it records the raw GPU snapshot, Docker-owned PIDs,
foreign cgroups and timestamp. A CPU regression checks that stale/owned entries
are ignored while live weather work is still protected (4 training-input tests,
8 geometry tests and288 preexisting tests passed). Physics and policy settings
remain unchanged.

Preserved campaign002/source_v2. Current remote frozen source is
`training_source_v3`, campaign is `walking_campaign_003`, coordinatorPID2951839,
all under `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/`.
The heartbeat points to campaign003. It reruns the baseline gate and then PPO.
Do not call PPO running until learning updates/checkpoints appear.


#### Confirmed learning, campaign003

Baseline PPO is actually learning in256environments. At about18:07Z the log
reported iteration34/500, checkpoints `model_0.pt` and `model_25.pt`, about
1.37seconds/update and10m51s remaining. User was told about11minutes for the
baseline and a preliminary12–16hours for the initial49size sweep including
validation/evaluation; multi-seed/refinement/CAD work takes longer. No winner
or completed policy was claimed.

Post-processing correction: the LOCAL `tools/rank_length_study.py` additionally
requires MAE <=max(0.03m/s,25% of command) at all three test speeds before a
candidate is eligible. This prevents zero-motion low-power policies from
qualifying. Its regression test includes a standing-still candidate. It does
not change PPO or physics. The active frozen source_v3 keeps its original
ranking helper; do not mutate a running snapshot. Pull completed evaluations
and regenerate the authoritative analysis using the updated LOCAL helper,
and use the corrected helper for subsequent frozen campaigns. Four local
training/ranking tests pass, including live GPU ownership and standing-still
rejection. Total with geometry and preexisting tests:300.


#### Baseline-first video gate (2026-09-09, latest user steering)

User asked to verify one policy and provide its actual recording ASAP before
launching the long all-size sweep. Baseline f100_t100 completed500 updates
(3,072,000 transitions) at18:18:31Z, final checkpoint SHA-256
`9b6f826b2ca36d85979a1ddc9e8d9bdecd18fc8da0be6192d859ff3945870525`.
Latest training telemetry showed weak forward tracking; successful optimizer
completion is not proof of walking. The remaining48 sizes are held.

Campaign003 had already started baseline evaluation when the lock-waiting
helper could acquire the shared GPU lock (flock reacquisition can race a
waiter). After verifying final.pt against completed training state, explicitly
wrote stop.request to interrupt only campaign003's evaluation for video
priority. No trained checkpoint was lost. Its evaluation is partial and must
not be ranked. Campaign003 now has status stopped. The helper exited after
observing this and released its temporary lock.

`tools/train_length_study.py --mode video` loads that exact checkpoint and
admitted geometry, runs1 environment for12s with a0.2m/s forward command,
records actual Isaac RGB frames at1280x720/25fps, and writes trajectory and
checkpoint identity. Training physics/reward/optimizer settings are unchanged.
`tools/launch_length_pilot_spark.py` runs only baseline video then matched
evaluation using the same per-job GPU guards.

First pilot source_v4 failed before GPU launch because the inherited source
hash manifest included mutable Python bytecode. Preserved it. source_v5
excludes __pycache__/*.pyc and the CPU coordinator runs python -B. Active
remote pilot: `baseline_pilot_005`, source `training_source_v5`, both under
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909`. Systemd user unit
`hexapod-baseline-video-pilot5-20260909.service`. Inspect pilot.json and
f100_t100/video/{state.json,run.log,video.json,rollout.mp4}; recording must
complete and be visually checked before claiming a walking video. Matched
evaluation follows automatically. No other morphology is queued by this pilot.
Do not resume49-size training until evaluation/recording and policy behavior
are checked end to end; diagnose weak learning first if confirmed.

The interactive inspector now labels the actual current production reference
`hexapod_mkii_serial.urdf` and embeds its exact downloadable XML (hash prefix
6109956e9e3a). None of the49 mock variants is that production CAD model. A
star identifies only the original mock100/100 baseline, explicitly labeled.
No production CAD edit occurred. Updated fragment size822810bytes.
Twelve geometry/training-input CPU tests passed after video support was added.
Forecasting pause and restoration responsibilities remain unchanged.


Recorder retry: pilot005 hung before AppLauncher ready, CPU~0%, no CUDA,
Python futex wait, matching the older documented Spark pre-startup symptom.
Stopped only its owned container via its stop.request. Source_v6 adds the
existing deploy scripts' Kit telemetry startup flags and90second Python
faulthandler startup diagnostics, plus bounded GPU-lock retry in the pilot.
Current active pilot is `baseline_pilot_006`, source `training_source_v6`,
unit `hexapod-baseline-video-pilot6-20260909.service`. It passed AppLauncher
and generated actual RGB frames at18:22Z. Do not mutate its frozen source.
Both prior failed/hung pilots are preserved. Local300 CPU tests passed.


Baseline video complete: pilot006 recorded300 frames,1280x720 H.264,25fps,
12seconds real-time playback. Local copy:
`artifacts/length_study_2026-09-09/baseline_500_rollout.mp4`, trajectory in
`baseline_500_video.json`. ffprobe confirmed duration/frame count/codec and
first/10second frames were inspected. Mean forward velocity after2seconds
was0.03127m/s for0.2m/s requested; zero falls in this single12s rollout.
It moves slowly with lateral drift, not satisfactory command-tracking gait.
This is a diagnostic rollout, not a successful morphology or hardware result.
Baseline matched64-environment evaluation follows automatically in pilot006.
Keep remaining48 held and diagnose baseline learning before the full sweep.


#### Proposed controlled-gait screen, following user strategy question

User asks whether to hand-pick a few geometries for PPO or screen all without
training each from scratch. Recommendation is a hybrid: mechanical screening
across all49, short torque-limited dynamic trials of parameterized tripod/
ripple/wave controllers with equal tuning budgets, then4–6 promising designs
plus controls for detailed CAD and learned-policy validation. Proposed method
and sources are in `artifacts/length_study_2026-09-09/CONTROLLED_GAIT_SCREEN.md`.
It has not yet been implemented or run. Keep the full PPO sweep held.

Pilot006 evaluation failed at its second speed reset with an in-place update
to an inference tensor outside InferenceMode. Only0.10m/s completed, complete
is false; its very low speed and18.9% requested-torque saturation are not an
all-speed evaluation. Local copy `baseline_500_partial_evaluation.json`. The
local evaluate function now has @torch.inference_mode() so resets and stepping
share the same scope; compilation passes, Isaac rerun still required in a new
frozen source. Do not mutate source_v6 or mark this fix runtime-verified.


#### Stage 1 complete; candidate confirmation required (2026-09-09)

User authorized the controlled-gait strategy, then explicitly requested the
Stage 1 candidates for a smell test before proceeding. Stage 2 controlled
walking trials and all GPU training are held pending that confirmation.

Completed CPU mechanical screening for the original 49 sizes plus nine
boundary/intermediate probes: femur 40/45/55% crossed with tibia 50/60/80%.
Those nine URDFs were regenerated from
`artifacts/length_study_2026-09-09/boundary_stage1_config.json` into
`robot/hexapod_mkii_length_study_boundary/`; coxa is fixed, current mass 8.2608 kg
and transferred inertia tensors are held fixed. No production CAD was edited.

Tools: `screen_length_mechanics.py` provides URDF FK/IK, exact mesh extrema,
force/moment balance with unilateral friction-constrained contacts and minimax
joint load. `run_length_mechanical_paths.py` and
`parallel_length_mechanical_paths.py` evaluate the common pose-selection rules
and tripod/ripple/wave schedules. The full prescribed-motion inverse dynamics
includes COM and rotational accelerations, gyroscopic terms, joint armature
0.0007 kg m^2, Coulomb friction 0.01 N m and viscous friction 0.002 N m s.
The body motion is prescribed for these calculations: no walking controller
or floating-body dynamic trial has demonstrated the target motions yet.

The static search used 56 nominal poses per size. The path search evaluated
1,934 trajectory configurations at 0.05/0.10/0.20/0.30 m/s, two strides
(60/100 mm), 20 mm swing lift and three gait schedules. The shared pose
selection rules retain up to eight diverse poses per schedule. Constraints
include joint soft limits, support margin >=10 mm, nonfoot clearance >=5 mm,
foot-pad discrepancy >=-2 mm, and the 1.6 N m continuous motor region. The
review table imposes nominal belly clearance >=90 mm. Initial phase sampling
was 64; selected paths were rechecked at 256 samples. Candidate 0.20 m/s peak
torque changed by less than 0.6% under this refinement.

Authoritative output under `artifacts/length_study_2026-09-09/`:
- `mechanics_stage1_v3_static/` and `mechanics_stage1_boundary_static/`;
- `mechanics_stage1_motor_complete/` (49) and
  `mechanics_stage1_boundary_paths/` (9), including per-shard frozen source
  copies and hashes;
- `stage1_refined_motor_candidates.json` and
  `stage1_refined_boundary_candidates.json`;
- `stage1_candidate_review.json`, `STAGE1_CANDIDATE_REVIEW.md`,
  `stage1_candidate_profiles.png`, `stage1_torque_map.png`.
Earlier failed, partial and pre-armature screening artifacts are retained but
superseded. In particular, do not use `mechanics_stage1_complete/` or
`stage1_refined_candidates.json` as the final motor-complete screen.

Review candidates: femur/tibia mm, peak estimated N m at 0.20 m/s,
nominal belly clearance mm (256 phase samples):
- A f040_t050: 58/105, 1.0684, 96; lower-boundary probe, motor fit unresolved.
- B f050_t050: 72.5/105, 1.3429, 118; compact.
- C f050_t060: 72.5/126, 1.3807, 142; useful central candidate.
- D f050_t080: 72.5/168, 1.3486, 183; clearance comparison.
- E f050_t100: 72.5/210, 1.4468, 226; long-tibia energy comparison.
- F f055_t060: 79.75/126, 1.4673, 137; more femur room, less torque margin.

All 304 CPU tests passed (16 robot/study plus 288 Isaac tests). Four new
independent checks cover IK/FK and finite-difference Jacobians on all legs,
gravity torque against potential-energy derivatives, LP force/moment/friction
constraints, and rejection of unreachable IK targets. Candidate profile and
torque-map images were visually checked.

No sampled motion passed 0.30 m/s. The original mock 145/210 baseline passed
only a 0.05 m/s sampled case; this is not a proven speed limit. Production CAD
is a separate control, not the mock baseline. Shorter femurs still improve
the sampled boundary, so minimum manufacturable length remains unresolved.
Rigid motor/linkage/mount fit, actual CAD mass properties, self-collision,
impacts/slip/feedback tracking, disturbance recovery and thermal endurance
remain unverified. Mechanical work is not battery consumption. Do not claim
that a hardware optimum or even successful dynamic walking has been found.

While Stage 2 awaits confirmation, restored the Forecasting-Pipeline dispatch
and scout timers at 2026-09-09T19:22:54Z using the preserved
`forecast_pause_001/resume_forecasting.py` on Spark. Both timers verified
active; `forecast_pause_001/restored.json` records the restoration. The native
48-hour backstop was left unchanged. GPU utilization was 0% with no study
container before restoration. Before any later approved GPU work, recheck
ownership and coordinate the already-authorized forecast pause with both GPU
locks; do not assume the GPU remains free.

User smell-test update (2026-09-09): explicitly exclude option A / f040_t050
(58/105 mm) because the femur is too short to fit the motors. This is a
packaging rejection from the user, not a failed dynamics result. Preserve its
screening data for provenance, but remove it from the active candidate list.
Recorded in `artifacts/length_study_2026-09-09/stage1_user_review.json` and
consumed by the report generator. B–F remain under consideration. Assistant
recommendation is C (72.5/126) as the lead for further validation, D (72.5/168)
as the closest competitor, B as the compact comparison, F as a longer-femur
packaging fallback to check, and E as the long-tibia comparison. C offers
about 23 mm more nominal clearance and 16% lower positive-work proxy than B
with nearly identical worst-motor RMS torque. D is close enough that its
slightly lower peak torque and greater clearance warrant direct testing.
No measured dynamic-stability ranking exists. The user asked for a
recommendation; no Stage 2 candidate selection has yet been confirmed.

#### User stance concern: selection bias found and C/D audited

User questioned why longer tibia D has lower peak torque and why the shown
stances are tall. This was a request to investigate, not Stage 2 approval.
Audit found a material selection gap: minimum-static-torque seeds per knee
angle and minimum-height threshold could discard lower postures before the
motion screen. Constant body height/orientation in inverse dynamics also
does not assess actual steady-body motion. The previous recommendation was
too strong about overall project suitability; retain it only as a provisional
geometry lead. A remains excluded for hardware packaging.

New scripts `tools/audit_length_stances.py` and
`tools/report_length_stance_audit.py` generated
`artifacts/length_study_2026-09-09/stance_audit_all_low_poses/` with frozen
source copies/hashes, `audit.json`, `refined_lower_stances.json`,
`same_lengths_lower_stance.png`, and `STANCE_AUDIT.md`. The earlier two-seed
pilot is separately preserved in `stance_audit/` and is superseded.
No original simulation result or URDF was changed. No GPU work started.

Original C/D poses both use mock angles femur10/knee110 degrees. Extending
the inward-pointing tibia brings the nominal lf foot radially closer to the
hip: 61.45 mm C versus 53.66 mm D. The knee offset grows from 9.95 to17.74 mm.
Joint-load optimization also redistributes horizontal ground forces; longer
tibia therefore does not increase every joint moment together. The original
peak difference is only 2.3%. The limiting torque decomposition was rerun at
256 phases and its signed components checked to sum to the actual torque.
This is not proof of a realizable smooth contact-force controller.

Expanded CPU audit completed 459 trajectory configurations: every retained
original-grid pose with belly clearance 80–140 mm for C/D, tripod/ripple/wave,
40/60/100 mm strides, 20 mm lift, speeds0.05/0.10/0.20 m/s. Initial64 phase
samples; the best kinematically/contact-feasible 0.20 m/s case in the95–120 mm
height band for each geometry was rechecked at256 phases:
- C f050_t060: femur40/knee120deg, clearance107.53 mm, tripod100 mm stride;
  peak1.38709 N m at0.20 m/s, passes mechanical gates, work23.8695 J/m,
  worst-motor RMS0.77248 N m. Same geometry as the old141.55 mm /1.38074 N m
  pose. This lower pose was missed by the original seed selection.
- D f050_t080: femur60/knee120deg, clearance116.98 mm, same gait/stride;
  peak1.64994 N m at0.20 m/s, fails continuous-torque gate. At0.10 m/s this
  same tripod path is1.50934 N m and passes; it is not the best slower gait.
These heights are similar, not exactly matched. No global infeasibility or
hardware optimum follows. Both geometries can adopt lower poses; A's short
femur is not necessary for the crouched appearance. Figure uses exact mock
leg meshes with a schematic body block and was visually checked.

Further Stage 1 comparisons must explicitly include common body-height bands,
stance width and motion headroom across all remaining candidates. Do not
reuse the earlier pose pruning as an exhaustive crouch test. Stage 2 still
awaits the user's candidate confirmation, and forecasting stays restored.

#### C selected; full-robot walking policy authorized (2026-09-09, latest)

User: "proceed with C. I'd like to see a fully training walking policy as soon
as possible with the full robot". This supersedes earlier Stage 2 holds and
prioritizes a single C policy over the multi-size comparison. C is f050_t060,
72.5 mm femur /126 mm tibia, coxa fixed, full six-leg19-body18-joint mock-study
robot with current8.26081134 kg transferred mass/inertia. Production CAD was
not edited. Use lower femur40/knee120deg stance, root contact height0.1305325 m,
belly107.53 mm, reset root0.1365325 m. A remains excluded for motor fit.

Prepared separate inputs via `tools/prepare_candidate_c_training.py` into
`artifacts/length_study_2026-09-09/candidate_c_training_inputs/`. The original
49-size training plan is unchanged. New named-joint256-phase reference is
100 mm stance travel,20 mm lift, tripod duty0.65,6.5 cycles/m. The new
`length_reference_env.py` supplies position AND velocity targets through the
same capped RS05 motors, plus learned residual position offsets up to0.12rad.
Actual base motion/contact forces remain unconstrained physics. Each reset
ramps the reference over2s; the actor observes phase via68 observation values.
Reference-only stepping is labeled untrained and is tested before PPO.

`tools/launch_candidate_c_spark.py` through
`isaaclab/deploy/hexapod-rl candidate-c` runs full standing validation,
32-environment reference probe at0.10/0.20m/s, then PPO1024envs:300 initial
updates followed by600-update resumptions. Each chunk records an actual
checkpoint video BEFORE evaluation. All outputs are fresh and hashed. It
stops for review after two consecutive walking-gate passes or6300 updates;
that is not proof of full robustness or manufacturing readiness. Existing
GPU locks/ownership guards remain. SIGTERM requests an orderly stop.

All306 CPU tests passed (18 robot/study +288 existing Isaac), including new
checks of reference asset/name identity, joint range and full six-leg stance
foot velocity cancellation of the commanded forward body velocity. Video
now wraps reset and stepping in inference mode as evaluation already did.
Runtime tests remain authoritative; do not call the prior unverified issues
fixed until their relevant modes finish successfully.

Remote active source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/training_source_v7`,
134 frozen source hashes. Campaign: `candidate_c_001`. Unit:
`hexapod-candidate-c-v7-20260909.service`. Started20:24:22Z, passed AppLauncher
startup and began validation. Never mutate v7. Status in `candidate.json`,
standing in `f050_t060/validate_0`, untrained probe in `f050_t060/probe`, and
training/video/eval chunks in `stage_000/f050_t060/{train,video,evaluate}` etc.
Copy and visually inspect the first completed trained video and deliver it
ASAP; do not wait for the whole campaign. Keep working on weak gait or errors.

Forecasting GPU dispatch was identified and paused again under the user's
existing explicit authorization. Evidence and restore script in
`forecast_pause_002/`. The candidate systemd unit has ExecStopPost to run
that restore script when it exits; original48-hour native backstop remains
unchanged. Verify restoration on exit. Do not interrupt other workloads.

C standing gate passed at20:25:43Z:32envs,1000steps,19bodies/18joints/6feet,
zero terminations/truncations/nonfoot contacts, zero post-settle saturation,
max post-settle computed torque0.56404 N m, mean root height0.129538 m.
Local evidence: `artifacts/length_study_2026-09-09/candidate_c_001/standing_admission.json`.
Observed joint order for C import: revolute_1_1, revolute_1_7, revolute_2_5,
revolute_3, revolute_4, revolute_5, revolute_1, revolute_1_6, revolute_1_5,
revolute_1_3, revolute_1_4, revolute_1_2, revolute_2, revolute_2_6,
revolute_2_4, revolute_2_2, revolute_2_3, revolute_2_1. Reference columns are
looked up by name against this runtime order. Probe followed automatically.

Full C reference-only probe completed successfully (still untrained):32envs,
20s each at0.10 and0.20m/s. Mean forward speeds0.09609 and0.20240m/s; zero
falls and nonfoot contacts. Tilt RMS0.2539/0.1815deg; speed absolute error
0.02720/0.05792m/s; requested-torque saturation2.6809/3.1551%. Thus average
speed is good but instantaneous tracking and saturation still miss the final
walking gate. Local `candidate_c_001/untrained_reference_evaluation.json`.
The prior repeated-reset inference failure did not recur in this completed
two-speed probe. Actual reference motor velocity targets were exercised.

PPO stage_000 started automatically with1024envs. Early iteration time~1.64s,
initial300 updates about8–9minutes plus recording startup. First trained clip
was estimated about10minutes from20:29Z, conditional on that measured pace.
Do not call the reference-only probe a trained policy or a final walking result.

First trained C recording ready and visually checked: stage_000 completed300
PPO updates (7,372,800 transitions), checkpoint SHA-256
`5ecbe978d659413dc050fec93b0e211a3eadf6ce1c807790d87c14044ff245a2`.
Actual12s1280x72025fps300-frame recording is locally at
`artifacts/length_study_2026-09-09/candidate_c_001/stage_000/rollout.mp4`;
checkpoint `policy.pt`, video trajectory, environment/agent configs, recording
audit, and frames1/5/10s are alongside it. ffprobe and checkpoint identity
verified; all three frames visually inspected. Mean forward speed after2s
is0.21994 m/s for0.20 commanded, zero falls in this single rollout. This is
an initial trained checkpoint, not a full convergence/robustness result.
Delivering this video immediately in the current user turn. Do not repost
the same stage_000 clip on unchanged heartbeat checks.

Current campaign mode after recording: stage_000 evaluation. It then continues
automatically into stage_001 with600 further updates unless stopped/failed.
Local helper `tools/sync_candidate_c_results.py --stage N` retrieves completed
identity-matched recordings and checkpoints, checks video metadata and extracts
inspection frames. The existing automation follows progress every5minutes.

#### Benchmark 1 preserved; mission scope restored (2026-09-09, latest)

The user selected the first C trained video as the first benchmark for the new
approach, then clarified that this is only a demo and should not absorb excessive
training time. The final goal remains omnidirectional walking over terrain using
onboard sensors, as described in `artifacts/project_review_2026-09-04/ROADMAP.md`.
They requested a documentation-grounded plan of action. This supersedes automatic
continuation of the forward-only C campaign to 6300 updates.

Benchmark `benchmark_01_c_300` is frozen at
`artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/` and mirrored at
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/benchmarks/benchmark_01_c_300/`.
It contains read-only checkpoint/video/configs, source archive, URDF/reference,
evaluation and a SHA256SUMS manifest. All 134 frozen source hashes were checked;
all 24 benchmark payload hashes were checked again on Spark. Checkpoint remains
`5ecbe978d659413dc050fec93b0e211a3eadf6ce1c807790d87c14044ff245a2`.
Do not overwrite the benchmark or repost its unchanged recording.

The completed nominal evaluation uses 32 environments, 20s per command, seed7057.
At commands0.10/0.20m/s: actual means0.11027/0.21989m/s, absolute speed error
0.01487/0.02533m/s, tilt RMS2.777/2.620degrees, requested torque >1.6Nm in
4.4435/4.8063percent of sampled joint/environment/control-step values. Both tests
had zero falls and nonfoot contacts. Positive mechanical power2.299/3.462W is
not battery power. Torque demand still fails the0.5percent gate; benchmark
selection is a user-preferred demo milestone, not hardware qualification.

The original coordinator had already started stage_001. Its existing campaign
`stop.request` has now been written to stop only the owned extra forward-training
job and preserve all saved outputs. Verify unit exit and restoration of both
forecast timers through the existing ExecStopPost; do not restart this campaign.
The automation was updated to this scope and should be paused once restoration
is verified. Frozen sourcev7 remains unchanged.

Next plan: verify physically buildable C geometry, mass/COM and single-leg motor/
linkage response; extend control to stop/start and signed forward/reverse/lateral/
yaw plus combined commands; qualify bounded terrain, then deployable sensor-based
terrain observations and estimated motion; integrate Jetson runtime and survey
coverage navigation. Preserve the useful phase-guided baseline, and compare any
less constrained controller by evidence instead of treating the demo's fixed
tripod schedule as a permanent terrain requirement. The old roadmap's unguided
policy recommendation predates this accepted new demo; the final behavior and
physical deployment goals remain applicable. Sensor prototype mounts/accounting
are provisional and its old6.3kg budget must not replace current mass accounting.
The user has asked for a plan, not launched a new long terrain/sensor training job.

Cleanup verified at20:51Z: candidate.json status `stopped`, owned systemd unit
inactive, both forecast dispatch/scout timers active, and forecast_pause_002/
restored.json records restoration. The demo-training heartbeat is now PAUSED.
Benchmark 1 is preserved; no new training campaign has been launched.

#### Step 2 authorized: research-backed flat omnidirectional training (2026-09-09)

User explicitly asks to continue Step 2 now, research first, change architecture
if needed, train translation across all360degrees and both yaw signs, and keep
future sensor-based terrain adaptation and the final roadmap central. This
supersedes the previous planning-only hold, while Benchmark 1 remains frozen.
Terrain training is next and has not been launched.

Read `artifacts/omni_flat_2026-09-09/RESEARCH_AND_PLAN.md` for primary-source
research, curriculum, architecture, held-out tests and physical limitations.
Research supports commanded planar velocity/yaw, separate physical regularizers,
and a terrain-aware privileged teacher followed by a sensor-based student.
Quadruped research is method evidence, not proof of this hexapod's performance.

New architecture `omni_history_direct_v1`: scratch PPO,315 actor observations
(5frames of63 IMU/gravity/command/joint/action values),318 critic observations
(add simulator linear velocity only to critic); direct joint-position offsets,
no fixed gait clock or forward stepping reference. Same selected full C study
model,72.5/126mm,coxa fixed,8.2608kg transferred mass/inertia. Production CAD fit
remains unresolved. Named runtime action mapping and caps remain in use.

Commands continuously sample all bearings, pure yaw, mixed motion and standing;
targets change every3–6s with bounded vector/yaw acceleration. New rewards score
requested vector/yaw errors, bounded progress, motor demand, power, smoothness,
slip, nonfoot contact and limits. Flat posture terms must change for terrain.
Actor observation noise is provisional; field dynamics are not calibrated.

All311 CPU tests pass:288 existing Isaac plus23 robot/study tests, including
5new tests for bearing invariance, both yaw signs, overspeed/idle reward,
command coverage, reversal/stop acceleration and reset-safe observation history.
Runtime standing/adaptation/PPO/evaluation/recording still require pilot evidence.

Remote frozen source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_source_v1`
(138source hashes). Campaign `omni_flat_001`, unit
`hexapod-omni-flat-v1-20260909.service`, deploy entry `hexapod-rl omni-flat`.
Status `omni_flat_001/omni.json`. It runs full32env1000step standing validation,
100-update pilot with1024env,90 static command/seed rows plus uninterrupted
transition evaluation and an actual40s labeled recording, then700-update chunks
up to1500updates for review. Neither budget completion nor average reward is a
success criterion. Watch the first pilot before allowing blind long retries.

Forecasting dispatch/scout paused in forecast_pause_003 under continuing user
authorization. Only Forecasting-Pipeline units were stopped; preflight checks
unrelated GPU users. Unit ExecStopPost restores timers via its resume script;
original native backstop remains. Verify restoration on exit. Frozen sourcev1
and Benchmark 1 must not be modified. If a runtime repair is needed, freeze a
new source version and create a fresh campaign output; preserve all evidence.

#### Arc-capable motion contract and labeled path demos (latest Step 2 steering)

The user asks to choose the best movement approach for future path planning,
combine arcs/turning/strafing, keep terrain adaptation central, continue, and show
example paths with labeled arrows on the ground in Isaac. Confirmed explicitly:
this is PPO (RSL-RL); a separate path follower supplies body-twist requests.

Keep the three-component forward/left/yaw command contract. Plan position and body
heading independently. Future candidate local planner is holonomic MPPI informed
by measured gait limits; current demo uses a bounded feedforward/P pose follower
with ideal simulator localization. Do not claim MPPI or real sensing is deployed.

Version1 runtime pilot completed100updates, standing gate,90-row evaluation and40s
video successfully. It mostly stands; only two static standing rows pass. No moving
capability is established. Evidence fetched under artifacts/omni_flat_2026-09-09/
omni_flat_001/stage_000/ (checkpoint1de5137cb9423fd8d82610279173b8f84a52cbc4e2fc520b71059a79542a0ad2).

Latest saved continuation was model425 from stage001. Copied with verified SHA
5ac83e8ce8909af4d33e3aebbe8aff4581ff930dd34228219cf8c0c19e9f074b to
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_transition_001/resume.pt`;
provenance alongside. A stop.request was sent to version1 before migrating. Its
source/evidence and Benchmark1 remain immutable. Verify forecast restoration before
new pause/launch; use a new campaign and source for version2.

Prepared frozen remote omni_source_v2 (139source hashes): same315/318 actor/critic
architecture and PPO optimizer, combined-yaw samples through zero for gentle arcs,
77static scenarios ×2seeds (616env),70s continuous transitions with S-curve and
fixed-heading bend. New tools/omni_path_demo.py renders five separate actual-policy
path trials (69s total) with ground labels, blue travel/reference arrows, gold body
heading arrows and orange actual trail. The path follower cannot move the robot
except through requests to PPO. Failures and path error are displayed and recorded.
All315CPU tests pass. New rendering still needs an Isaac smoke/visual check.

Active continuation is now **omni_flat_002**, frozen **omni_source_v3** (139hashes),
unit **hexapod-omni-flat-v3-20260909.service**. Version2 was never launched; version3
adds initial renderer creation before camera/drawing operations. It resumes the
verified model425 checkpoint, revalidates the full robot under the new plan, runs
25 PPO updates to exercise the updated154-row evaluation and69s annotated path
video, then700-update chunks up to1425additional updates before review. Sourcev1,
v2,v3 are preserved; never mutate any frozen source.

Version1 stop completed and forecast timers were verified active. Continued user
authorization was used for forecast_pause_004; the new unit's ExecStopPost restores
both timers via that folder's resume_forecasting.py. Check restoration on exit.
Current status: omni_flat_002/omni.json; standing: f050_t060/validate_0; stage outputs:
stage_000/f050_t060/{train,evaluate,video}. Fetch completed identity-checked media
with `python3 tools/sync_omni_results.py --campaign omni_flat_002 --stage N`.
Inspect actual frames for leg behavior and visible ground arrows/text. Ground
annotations are non-colliding display meshes; the root/body must never be driven.
The demo's pose feedback uses simulator localization and must be labeled accordingly.

#### First five-path recording and corrected-label render queue

Campaign omni_flat_002 stage_000 completed the 25 resumed PPO updates, evaluation,
and 69-second actual simulator recording. Checkpoint SHA256 is
`3ed406ca46d948ff9c438ac9b9817a58df39d842c6264a3dc96dc78bedfd3ef8`.
Evidence is local under `artifacts/omni_flat_2026-09-09/omni_flat_002/stage_000/`.
All five path trials had zero terminations. P95 position errors were 0.110 m
straight, 0.058 m sideways, 0.103 m combined arc, and 0.098 m fixed-heading curve.
These closed-loop demonstrations do not establish qualification: all 154 static
command rows and all 14 transition gates failed. Requested motor torque saturation
and tracking remain material problems. Safety terminations are not necessarily falls.
Stage_001 is continuing 700 PPO updates under the existing source_v3 unit.

Actual-frame inspection caught mirrored ground text and clipped legends. Do not
present that first recording as the finished labeled demo. Frozen `omni_source_v4`
(140 hashes) fixes ground glyph orientation and widens the camera framing; it does
not change policy or training. `tools/queue_omni_preview.py` is running as
`hexapod-omni-preview-v4-20260909.service`, output `omni_preview_001/preview.json`.
It waits for the next free interval using both existing GPU locks, selects the
latest completed, evaluated checkpoint with matching SHA, and renders without
interrupting PPO. It has a 90-minute deadline. It does not independently pause
forecasting; the main campaign still owns forecast_pause_004 and restoration.

Fetch `omni_preview_001/f050_t060/video/{rollout.mp4,video.json,state.json}` plus
`preview.json` and `inputs/policy.pt`, verify checkpoint identity and video duration,
then inspect ground text and framing before posting the corrected clip. The existing
sync_omni_results.py handles campaign stages, not this separate preview layout.
The corrected renderer is syntax checked but not yet visually validated in Isaac.
Future source_v3 campaign recordings retain the old label renderer; use source_v4
or a fresh frozen successor for presentation. Preserve all source versions and
Benchmark 1. Any follow-up must track both training and this queued preview.
The existing local heartbeat automation file disappeared during this turn; the
attempt to update its prompt failed. It was not recreated. Do not assume scheduled
follow-up is active without verifying the automation state. The remote training and
render queue run independently of that local heartbeat.


#### Completed omni run, corrected recording, and parallel acceleration (9 September 2026)

User explicitly requests Stage2 fastest completion and terrain/perception in parallel, with as many agents as useful and no avoidable Spark idle time. Three bounded agents now own PPO diagnostics, terrain integration, and sensor mount screening; root alone coordinates GPU jobs. Avoid competing GPU workers. Forecast pause authorization persists.

omni_flat_002 finished 22:55:06 UTC after1425 resumed updates. Final stage002 checkpoint SHA2561971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8. All154 static and14 transition gates still fail. Final median planar error0.0463m/s, yaw error0.1143rad/s, requested torque saturation14.26%; applied torque still capped1.6Nm. Stand seed7057 has no terminations but15.12% requested saturation, indicating problems are not solely reset contamination. Budget completion is NOT Stage2 completion. tools/report_omni_progress.py and campaign PROGRESS_REVIEW.md record comparison.

omni_preview_001 completed23:00:44UTC using that exact final checkpoint and frozen sourcev4. Local artifacts/omni_flat_2026-09-09/omni_preview_001 contains69s1280x72025fps movie,policy,state,video,audit and frames. Five paths all0terms/0truncs, P95positionerror20–57mm. Frames05/33 inspected: text normal orientation; straight lowest orange caption partly clipped, arc legend fits. Posted actual movie to user. Follower uses ideal simulator localization; this is not terrain or perception qualification.

New remote unit hexapod-omni-diagnostics-001-20260909.service launched23:21:27UTC via deploy hexapod-rl omni-diagnostics. Root /home/orionh/HEXAPOD_runs/mock_length_study_20260909. Fresh frozen sources omni_diagnostic_source_001_{baseline,slew,solver},142 hashes each; no older source mutated. Each performs full32env1000step standing admission then48env12s pre-reset diagnostic on the immutable final checkpoint. Separate plan changes: baseline none; slew0.03rad/20ms; solver cfg.sim.physics.enable_external_forces_every_iteration=True (verified installed API). Output omni_diagnostics_001/diagnostic_campaign.json and comparison_00..02/f050_t060/evaluate/{diagnostics.json,diagnostic_trace.npz}. Agent adds observation/history, finite-difference translation/heading and per-joint traces; diagnostics cannot qualify a policy. Launcher's evaluation evidence check now distinguishes this explicit mode.

Forecast pause005 recorded exact StormScope GPU cmd/cgroup and unit states before stopping only dispatch/scout timers/services. Unrelated isim-web-viewer-1 left alone. ExecStopPost runs pause005/resume_forecasting.py, runtime bound5700s; verify restored.json and timer states when finished. Existing original native forecasting backstop remains. Both GPU locks and owned-container cleanup remain. After comparisons, choose measured repair before another long PPO run.

Detailed primary-source research plan: artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md. User owns Mid360+D455; extra purchases unrestricted. Proposed LiDAR/forwardcamera plus screened near-foot cameras; no final sensor CAD claimed. Pure CPU prep in tools/terrain_readiness.py and prepare_terrain_readiness.py generated30 fixtures,108 initial mount candidates,payload ledger and uncertainty/age map contract. Separate mount agent expands actual-CAD screening. Training on terrain still requires physical/sensor smoke and exact payload admission.

#### Explicit Stage2 finish standard from user

The user says forward walking is the visual smoothness standard. DO NOT mark Stage2 complete until reverse, both strafes, all diagonal bearings, both yaw directions, combined arcs and path transitions match that level of smoothness, with quiet static stance after stopping. Small disturbance corrections are necessary, but habitual stepping/chatter at zero command is not acceptable. Require representative same-view/same-speed actual-policy videos plus objective motor, stability, tracking and stop/stand checks; one good forward clip, a median score, or budget exhaustion does not complete the stage.


#### Diagnostic conclusion and active quiet-stand repair

All3 controlled diagnostics completed. Source001 baseline requested saturation median14.93%, positive power4.94W,4base-contact terminations. Slew0.03rad/20ms:5.74%,2.30W,0terms. Solver external-forces-every-iteration=True did not help (15.26%,5.27W,4terms); retain False. Slew-only fast-forward and left-arc tracking regressed, so averages are insufficient. Actual standing pose/joint oscillation remains; it is not only noisy endpoint velocities. Evidence and inspected plots are under artifacts/omni_diagnostics_2026-09-09/{README.md,NEXT_EXPERIMENT.md,comparison_report.json,...}.

New opt-in rewards in omni_flat_env.py: stand_joint_velocity and stand_target_velocity, defaultzero for old behavior. Repair overrides: slew0.03; stand_joint_velocity-0.5,stand_target_velocity-0.15,stand_posture-2,action_rate-0.075,saturation-0.75,torque_excess-0.6,worst_torque_excess-0.2. Actor315/critic318 and PPO preserved. Full evaluation now measures before automatic resets, excludes each episode's first2s with per-replica denominators, and retains every failure count.

**Active unit hexapod-omni-repair-002-20260909.service**, launched23:37:55UTC; outputs remote omni_repair_002/repair.json. Frozen sources omni_repair_source_002_diagnostic and _full,142hashes each. First exact-plan32env1000step validation, held-checkpoint48env diagnostic, then100PPOupdates and comparison; up to300+300 more only with measured improvement, no material per-direction regression and further marginal improvement against previous stage. Guard is compute allocation only, not qualification. If no improved checkpoint clears the screen, return needs_repair without expensive known-poor full rerun. Selected improvement receives fresh full-plan standing admission,154static+14transition eval and actual path video; always stage2_complete=False pending visual/all-direction/quietstance/robustness requirements.

Pause008 owns forecasting restoration via ExecStopPost;4hbound+systemd14700s bound,KillMode=mixed. Verify pause008/restored.json and timer states on exit. Source/unit001 was stopped before any training while waiting for GPU; its evidence is preserved. Source002 adds per-direction and incremental allocation screens and cleanup of exact owned DockerID even if Docker client already exited. Focused omni tests19pass.

First terrain fixture runtime (source terrain_smoke_source_001,208hashes; terrain_fixture_smoke_001) stalled beforeCUDAinit. It was stopped to restoreStage2priority. Docker client exit initially left owned cb721d48dad84fcc4b855a247d314ee6f7ff314d9ef39d8fcbe841c621d1f660 running; explicitly stopped that exactID. No unrelated container touched. Local terrain launcher now reads fixtures/validation.json rather than state.json and checks containercleanup independently. Fresh fixture harness defers numpy/helpers untilafterSimulationApp, writes phases and90sfaulthandlertrace; startupcause remainsunproven. Never mutate source001. New fullC terrain standing runner is CPUchecked but needs passed fixture runtime and exactflatadmission beforeGPUuse. See isaaclab/hexapod_terrain/README.md.

Mount screen completed actualCADtriangle tests: first6hip-anchorD405views only16.1%targets visible acrosssampledattitude/lift; expanded120mmoutboard140mmaboveplate90degdown proposal78.1%mean,48.1%worstcase/sector. Neither is finishedCAD or sufficientcoverage. Details artifacts/sensor_mount_study_2026-09-09/README.md; productionCADscreen,notfinaldetailedC. Agentcontinuing CPUcalibrateddepth/pose/mapreplay; terrainagentcontinuing exactsupportqueries/curriculummanifest; PPOagentadding quietstand/stop andallbearingvideoeval.

Created local threadheartbeat **advance-hexapod-stage-2**, every5minutes, toreview/advanceauthorizedwork andnotifyonlymeaningfulchanges. The previously deleted automation was not restored; this is a fresh automation under the user's explicitcontinuousaccelerationrequest. Remote services run independently; localfollowupsdependonapp availability. AllGPUlaunches remainroot-coordinated.


#### Next review hooks and last verified live state

At the latest check, omni_repair_002 passed its new full standing admission and exact-settings diagnostic baseline, then started the100-update PPO pilot on the Spark. Observed iteration time was about1.7s; that is a pilot timing sample, not a Stage2 completion ETA. Baseline reproduced the slew comparison:0terms,5.744%medianrequested saturation,0.03951m/splanar error,0.09178rad/syaw error,2.301Wpositive power and0.750rad/sstandingjointvelocityRMS.

PPO agent completed isolated tools/omni_quiet_review.py and tools/omni_visual_review.py, documented in artifacts/omni_diagnostics_2026-09-09/REVIEW_RUNBOOK.md. They are NOT wired into the active frozen source and have not run in Isaac. Quiet review covers32sstanding plus12stop-from-motion cases; visual review46cases/334s spans16bearings at2speeds, turns, arcs, reversals, paths andquietstand. Integrate their dispatch hooks only into a fresh source; standing admission must match its plan. Proposed quantitative quiet thresholds need reference/standing validation, and visual review remains required.23focused omni tests and288Isaac CPUtests passed.

PPO allocation fixes requested by agent are already in active source002: namedper-direction regressions, improvement relative toprevious chunk and skipping full review when no candidate improves. Source001 never trained. Continue by inspecting omni_repair_002/repair.json, then fetchcompleteddiagnostics andselectedcheckpoint with hashes. Do not let a successful scheduled process overwrite the explicit user quality requirement or markStage2complete.


#### 100-update repair result and next active diagnostic

The omni_repair_002 pilot finished at Unix1788997483.5, after2m54s actual PPO training and about6m48s for the complete standing/baseline/train/evaluation cycle. It returned needs_repair without continuing: planar error improved0.03951→0.02883m/s, but requested saturation worsened5.744→6.640%, standing joint velocity barely changed0.750→0.745rad/s, and forward yaw error worsened0.0889→0.1344rad/s. There were no terminations. No candidate was promoted and Stage2 remains incomplete. Pause008 restoration and no owned leftover container were verified. User asked ETA; answered about5–10minutes per short cycle, with no reliable full smoothness-completion ETA yet.

PPO agent is analyzing exact pre/post traces and smaller decisive corrections. In parallel root launched **hexapod-terrain-smoke-002-20260909.service**, fresh frozen terrain_smoke_source_002 (208 hashes), output terrain_fixture_smoke_002. This traced retry uses deferred geometry imports, --info, phase state and90second stack dumps, five-minute job bound, correct fixtures/validation.json completion check and fixed owned-container cleanup. Pause009 owns restoration; verify restored.json on exit. Do not mutate either frozen terrain source. Terrain agent continues support-query/curriculum preparation.

Sensor/perception agent completed the synthetic CPU pipeline in tools/perception_replay.py: timestamped camera and point-cloud transforms, robot masking, uncertainty/age/observed map and student patch interface.14 targeted tests pass; synthetic steps/pits preserved, unknown/stale observations remain unusable. Artifacts/perception_readiness_2026-09-09/README.md documents limits: no ROS, actual sensors or actor integration yet.


## GitHub integration and current priority

The user explicitly requested ongoing commits/pushes and a pristine current main branch, with this work highest priority. Integration starts from main 096d9ef, preserving the 82 upstream commits absent from the older experiment checkout. The old C runtime is isolated rather than replacing newer packages or physical-model shims. Future verified steps update STATUS and relevant living Markdown, preserve immutable run sources, and commit/push after combined checks.


## 9 September 2026: main integration and controller-probe result

The user authorizes resolving conflicts and keeping main ready, with the current C-study Stage 2 highest priority. Integration starts from GitHub main `096d9efa3dbf0db2573d5791e370d96ad180548f`, preserving all 82 newer feature-branch commits and its merge. Production packages, compatibility shims, physical CAD and archived source manifests remain unchanged. Study imports use the separately pinned 16-file runtime; future source bundles must include its bootstrap and hash manifest. The new release manifest is `isaaclab/deploy/stage2_c_priority_20260909_pipeline.sha256`.

Canonical production tests passed 935/935; all study/terrain/perception tests passed 94/94 after declaring their CPU dependencies in the locked workspace. CI now runs both suites and the pinned-study check. All 24 immutable Benchmark 1 payload hashes pass; complete archived media/checkpoints are included. Production serial structural and stance checks also passed (19 links, 18 revolute joints, 8.26081134 kg), without claiming physical four-bar admission.

Control probes `omni_control_probes_002` completed three exact-plan comparisons with zero terminations. Zero-action PD standing had 0.00158 rad/s joint RMS versus 0.7501 for policy output. Removing observation noise did not fix the oscillation; 80 ms action filtering worsened tracking and saturation. Reject those interventions. Pause 011 restoration was recorded after the unit exited. See `artifacts/omni_diagnostics_2026-09-09/control_probes_002/README.md` for matched identities, per-direction results and the next bounded lower-exploration/quiet-action PPO proposal. Preparation is underway; do not call it running until a new unit is verified. Stage 2 remains incomplete.

The user requests a commit/push and relevant Markdown updates after every meaningful verified step. Fetch current main, preserve teammates’ work, run relevant checks and verify the remote SHA. `STATUS.md` is canonical; historical run histories and frozen payload bytes are retained.

### Main CI follow-up: confirmed sensor contract

GitHub run 34420179566 on b069287 completed all 935 production tests but failed two documentation subcases: an inherited string fixture expected a pending sensor label/photo check, a different sensor profile comparison and another permission gate. The user’s confirmed Mid-360/D455 inventory and parallel-work authorization supersede those requests. Updated the fixture to check confirmed identity and separate qualification boundaries. No physics, sensor-accounting, motor or simulator acceptance gate changed. The follow-up source manifest is `isaaclab/deploy/stage2_c_priority_20260909_docs_contract_pipeline.sha256`; the preceding manifest remains immutable.


## 10 September UTC / 9 September Eastern: paired PPO repair 003

Both independent 50-update branches completed from checkpoint `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. Strict initialization preserved every learned actor/critic and observation-normalizer tensor, reset Adam moments and exploration standard deviations to 0.10, set entropy coefficient zero, and used initial learning rate 0.00005 with the existing adaptive schedule. Only B added weight −2 to mean-square sampled raw actions at near-zero commands. These are matched C-study diagnostics at 0.03 rad/20 ms, not archived Stage2C formal measurements.

A checkpoint: `b87df2b9536460fe67db840784d41c43c1d4ad08339ceacb6aab4959f3757f8f`. B checkpoint: `7e6bb4b40bb56fb0bd547f71352098ef8aecafe3e447cb0a329a1206eefbcfd1`. Both original-checkpoint baselines match. Standing joint RMS changed 0.75014 → 0.73740 / 0.74017 rad/s; median requested saturation 5.744% → 5.676% / 5.531%. Standing saturation itself worsened, roughly 90% of standing target increments reached the limiter, and both existing continuation screens rejected further allocation. No terminations/truncations occurred. No full qualification or new walking video was claimed.

Evidence is preserved under `artifacts/omni_diagnostics_2026-09-09/repair_003/`, including 34 SHA-bound result files, exact plan/source comparisons, checkpoint initialization records, both checkpoints and inspected plots. The service was a **user unit** (`systemctl --user`), `hexapod-omni-repair-pair-003-20260909.service`. Its exit restored both StormScope timers via pause 012; the agent verified no remaining CUDA process before releasing ownership.

Behavioral changes were ported onto main without replacing its pinned runtime bootstrap. Future paired launches additionally enforce identical non-plan source hashes and reject nonfinite applied torque before metrics are scored; these CPU-tested guards are distinct from the immutable source used by the completed trial. The complete study suite passed 102 tests after integration. See STATUS.md for subsequent activity and docs/PLAN.md §9 for the controller contingency decision.


### Full-C terrain entry attempt 001

Source `terrain_robot_source_001` was built from published 9d610779 code, the pinned study runtime and the unchanged full-review plan. Source manifest SHA `734dbbfa87cbb9ee4a865d702d5a65c7e2dfefbeff2407a1032a2b4ee59cfc7b` covered 659 files. Fresh flat standing passed 32 × 1,000 controls, zero resets/nonfoot contacts/post-settle saturation, max settled computed torque 0.5367 N·m, mean root height 0.12953 m. Admission SHA `3c950f546c5bbb6af1337a1dfdda3c4b3f2115ce0087ee96db29726c4f035843`.

The subsequent zero-action terrain-entry process failed before robot stepping: the installed TerrainImporter produced a non-Mesh collider composition, and the strict fixture adapter rejected it. No full-C terrain admission resulted; the original 30 direct-Mesh fixture checks remain separate evidence. Source and admitted asset bytes stayed unchanged. Both owned containers exited, CUDA was empty, locks were free and both StormScope timers were restored under pause 013. Seventeen hash-bound evidence files are published at `artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_001/`; a corrected attempt must use new source/output names.

## 10 September UTC: reference feasibility and terrain entry failures

Published the exact first reference prototype, global-clock continuation and independent review under `artifacts/omni_diagnostics_2026-09-09/reference_feasibility_001/`. Its replay helper uses a fresh copied workspace, verifies26 frozen payloads, reruns13 tests and reproduces the independent report within1.78e-15 maximum absolute roundoff. The100mm and60mm transition results remain distinct. Benchmark1's source bypasses the inherited target limiter; preserve its original video as a visual reference without calling it a common-limiter qualification. The next bounded candidates are support-aware stance/swing/stop handling and a separately versioned observable target-velocity action process.

Terrain robot attempt002 passed flat admission and corrected Mesh import, then failed at reset because installed configclass.copy drops dynamic omni settings. Attempt003 replaced that copy with deepcopy, passed fresh flat32×1000 again and executed1000 terrain controls, but terminated every control. Its post-step metrics are reset-contaminated; no standing admission exists. Both failures, source identities,17-file result maps and forecasting restoration records are published in separate artifact directories. Root independently matched003 result hashes against Spark. No prior source or result was edited.

Installed AssetBaseCfg uses XYZW, while the terrain adapter's WXYZ yaw tuple encodes a sideways rotation. Main includes a one-line correction and an independent geometric regression; actual pre-reset confirmation and a fresh corrected run are required. The Mesh and configuration-copy fixes, bounded two-phase host launcher, exact-container cleanup and7 added study tests are integrated.109 study tests,9 source-lineage tests and the pinned runtime check passed locally. Production packages, robot inputs and every acceptance gate remain unchanged. See STATUS for execution ownership and newer results.
