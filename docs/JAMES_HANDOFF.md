# James: start here

Research is paused at the user's request, 11 September 2026 UTC. James is taking over. The Codex continuation is paused, research agents are stopped, and no native validation or PPO job is running. Do not resume the old unattended work loop. The Spark reservation remains in place; pausing this project does not release weather or reconstruction jobs.

The latest execution snapshot is [STATUS](../STATUS.md). The [handoff archive](../artifacts/handoff_2026-09-11/james_001/README.md) contains the unpublished preparation and review work, exact file inventories and pause receipts. All four local worktrees were clean before this handoff. Main is the integration branch.

## Where the project actually stands

The mission is no-RTK coverage of an operator-drawn area while keeping a steady survey deck. The progression remains first walking, omnidirectional movement, terrain/perception, then autonomous surveying. **Stage 2 and Stage 3 are incomplete.**

The accepted forward [Benchmark 1](../artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) is a historical C-study visual reference. The later historical 500-update PPO can move in multiple directions but passes 0/48 quiet trials. Neither checkpoint belongs to the newly approved detailed robot. Do not load old weights through a renamed task or claim that the detailed robot has a learned walking policy.

The canonical model is selected by [robot/active_model.json](../robot/active_model.json): `hexapod_mkii_updated_v1`, 19 rigid bodies, 18 direct-drive joints, 59 meshes and 153 detailed SDF colliders per robot. Its nominal corrected mass is 7.466088235 kg. There is no external four-bar. Nominal dimensions are about 49 mm coxa, 73.502 mm hip-to-knee and 130 mm knee-to-distal tibia reference; these are joint/reference dimensions, not all material cut lengths. [CAD conventions](UPDATED_CAD_IMPORT.md) explain the source geometry, travel and unresolved physical assumptions.

| Current model identity | SHA-256 |
|---|---|
| Corrected URDF | `9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78` |
| Corrected model JSON | `7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881` |
| Prepared USD | `3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c` |

The [detailed CAD video](../artifacts/mkii_updated_2026-09-10/cad_showcase_001/README.md) is a 24-second, 1080p kinematic showcase. It is useful for geometry review and is not a learned walking recording.

## The immediate blocker

Native import, suspended signed-effort response and sampled SDF/source agreement pass. [One robot stands successfully](../artifacts/mkii_updated_2026-09-10/native_standing_006/README.md) with source005 and 32 position / 0 velocity solver iterations. The matching [32-robot batch](../artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md) completed without contention or timeout, but passed only 10/32 combined/physical and 24/32 quiet checks. Twenty-two replicas lost six-toe support on 73 isolated samples; eight also failed the SDK-rate quiet bound. There has been **no PPO execution on this detailed model**.

The [full CPU channel replay](../artifacts/mkii_updated_2026-09-10/standing32_channel_analysis_001/README.md) and [subsequent interpretation](../artifacts/handoff_2026-09-11/james_001/prepared/canonical_standing32_channel_interpretation_001/REPORT.md) preserve the evidence. Each loss lasted one 2.5 ms step and had 128 inactive zero-force records on one tibia. Both normal-force channels read zero, actual tibia position moved downward, and support returned the next step. This cannot be dismissed as only a Python classification error. The shared backend, contact generation and unexported buffer behavior remain unresolved; 128 is an observed signature, not a proven capacity setting. Friction is a separate getter, so horizontal normal-force-minus-momentum residuals are not a total force balance.

The unexecuted next proposal is one robot at batch coordinate `(14, 4)` compared with the origin, with a separately frozen placement identity and unchanged dynamics/gates. This is a proposed discriminator, not an approved or completed experiment. Do not launch it merely because this document lists it. Review the proposal and source identity when James resumes work.

Preserve the 400 Hz PD / 50 Hz control contract, 1.6 N·m software effort cap and 0.040 rad per 20 ms comparison limiter. Original standing scores use 800 control endpoints over 16 seconds after 4 seconds of settling; the diagnostic uses all 6,400 substeps over the same interval. Different rate channels and sample frequencies cannot silently replace a failed gate. The 48 V torque-speed assumption remains provisional; hardware voltage, motor calibration, payload/inertia, thermal limits, attachment ownership and coupled-pose collisions need qualification.

## Prepared code that has not run

All paths below are under [the archive's prepared directory](../artifacts/handoff_2026-09-11/james_001/prepared/). Original frozen payloads are unchanged. Unfrozen files are labeled draft snapshots.

| Preparation | What James needs to know |
|---|---|
| `canonical_ppo_integration_003` | Latest frozen fresh PPO integration: 405 actor / 408 critic values, five observation frames, 18 actions. It preserves 8,000 neutral substeps, then proposes 32 environments × 24 controls × 2 updates and strict checkpoint reload. Matching 32-robot admission is still missing. |
| `canonical_ppo_host_004` | Latest frozen bounded host preparation, with explicit 1,200-second phase / 90-second readiness bounds and exact owned cleanup. Source005 is its physical parent. |
| `canonical_ppo_guard_002` | Disabled, previously unfrozen guard draft. Actual batch-admission fields are null. Its historical prose calls attempt005 pending; the preserved final result actually rejects. Do not fill those fields from a rejected run. |
| `canonical_ppo_spark_setup_001` | Stale unexecuted setup draft with older owner references. Preserve for context; do not execute as-is. |
| `canonical_walk_objective_001`, `canonical_locomotion_training_001` | Unadopted command/history/reward proposals for translation, yaw, arcs and stopping. They are not a task registration or native moving policy. |
| `canonical_locomotion_composition_independent_review_001` | Concrete pre-adoption defects: NaN clocks and fractional counters can pass parts of the moving proposal. The original failing draft is preserved; a successor needs explicit finite/integer checks. This is not a defect newly observed in the standing runtime. |
| `canonical_moving_contact_review_001` | Longer PPO needs per-environment termination/recovery, correct final-observation bootstrap and exclusion of recovery rows from learning/normalization. Stock `dones` alone is insufficient. A three-contact floor is only an unadopted bounded-probe proposal, not a Stage 2 gate. |
| `canonical_contact_cpu_optimization_002`, `canonical_ppo_throughput_design_001` | Unadopted recording/classifier optimizations. Component benchmarks do not establish native end-to-end speedup. |

Earlier PPO/host generations, independent reviews and actual Fable 5.1 MAX responses are included to preserve context. Their older instructions do not override this pause or the current canonical model. The source inventory lists every original freeze and copied byte.

## Spark and retained data

Use the team's authorized Tailscale/SSH access (`ssh spark`, host `100.82.166.9`, Unix user `orionh`). No private keys, tokens or `.env.base` contents are in this handoff. James's access must use the team's own provisioning.

The active campaign directory is `/home/orionh/HEXAPOD_runs/canonical_direct_20260910`. The mirror `/home/orionh/HEXAPOD` is a bind mount, **not a Git checkout**. Clone/pull the GitHub repository in a real checkout and stage explicitly versioned source directories; do not use a blind pull against the mirror.

| Remote path under the campaign directory | Purpose |
|---|---|
| `standing_source_005` | Exact source109, freeze `c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131` |
| `asset_001`, `native_actuation_001/actuation` | Pinned detailed asset and signed-effort evidence |
| `native_standing_006` | Passing single-robot result |
| `native_standing32_005` | Completed rejected batch, all 39 raw files / 5,019,294,627 bytes |
| `standing32_channel_analysis_result_001` | Original read-only analysis result |
| `ppo_source_003`, `ppo_host_004` | Staged but unlaunched PPO preparation |
| `james_handoff_pause_001` | Pause marker and actual timer/readback receipts |

The large batch contact stream, control trace and ten substep chunks remain on Spark. The [remote inventory](../artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/REMOTE_ONLY.json) names each exact absolute path, byte size and hash. These are not silently omitted from a claimed full Git backup. Preserve them; retrieve with sufficient destination space and verify the manifest. Earlier failed/interrupted raw runs and historical checkpoint inventories remain linked from STATUS and their original artifacts.

At handoff, GPU compute was empty. The `isim-web-viewer-1` container remained available; it was not an active training allocation. The enabled reconstruction queue-lock helper is a reservation service, not research. Thirty forecast user units plus Ollama remain masked, along with four system forecast units and the reconstruction launcher block. The old `hexapod-restore-forecasting-20260909.timer` was stopped and its transient units are absent. Keep both GPU locks and exact ownership cleanup on future runs. [OPERATIONS §3.1](OPERATIONS.md#31-complete-external-automation-block) holds the external-job recovery instructions; release still requires explicit user direction.

The Codex automation `advance-hexapod-stage-2` is **PAUSED**. Its saved prompt contains historical C-study instructions; review it against this handoff before any future re-enable. CI and Pages remain enabled. Pausing research does not waive native admission, grant a checkpoint compatibility claim, or authorize automatic restart by an old agent.

## Repository recovery and first reading

The pre-main C-study stash is preserved remotely as tag `handoff/pre-main-c-study-20260909`, pointing to stash commit `8e535d489dace869f78a8cd6506c589c7fb8dcd4`. It is historical recovery material, not the current implementation. **Do not apply it over main.** Current code and the original stash are preserved separately; teammate branches/worktrees were not reset or rewritten.

Start with this document, STATUS, [PLAN](PLAN.md), [TRAINING](TRAINING.md), [NEXT_RUNS](NEXT_RUNS.md) and the linked blocker evidence. Terrain fixtures and perception prototypes are already published, but full terrain traversal and sensing qualification remain open. Mid-360 and D455 are available; purchases were not restricted.

Every contributor must follow [the live research poster contract](PROJECT_SITE.md). Preserve frozen evidence and task IDs; publish a central update and current Markdown with each meaningful change. For handoff verification, use the archive's portable verifier. For repository code changes, run the relevant tests and integrated release check; the canonical suite is `uv run python -m unittest discover -s isaaclab/tests`. Keep the four major checkpoints and the unchanged numeric-plus-visual completion requirements visible.
