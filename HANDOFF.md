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
