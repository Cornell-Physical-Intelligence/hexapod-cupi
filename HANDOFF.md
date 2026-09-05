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
