# Independent resolution of Fable MAX architecture003 checks

The three proposed pre-pilot blockers do **not** reproduce in actual direct smoke002. Fable correctly described them as checks, not proven bugs. The physical quiet-stop failures remain real: replay of the unchanged scorer gives **0/48 quiet passes**, five terminations across the trial, and two terminations inside the final quiet window. This receipt supports proceeding with the already proposed bounded comparison; it does not predict a performance pass or change any gate.

| Check | Executed evidence | Resolution |
|---|---|---|
| 32-second stop exceeds a 20-second timeout | Exact source002 `train_length_study.py:261` selects 20 s for training and **90 s for omni evaluation**. Both resolved YAML files match. `HexapodEnv._get_dones` still applies timeouts; Omni does not override it. Actual stop has **0 truncations**; training has one timeout at zero-based control31/env24. | No forced stop timeout. `evaluation=True` suppresses inherited random command sampling; it does not suppress `_get_dones`. Training retains randomized initial episode ages. |
| Quaternion wrongly reordered | Installed SDK snapshot documents `root_link_quat_w` as **XYZW**, and `root_quat_w` aliases it. Legacy capture misleadingly names that raw data WXYZ; native003 explicitly saves the raw XYZW and reorders once. Every one of 76,800 recorded converted quaternions equals raw `[3,0,1,2]` bitwise. | Conversion is correct. Independent SciPy projected-gravity error ≤7.273e−7 and navigation-heading error ≤3.350e−7 rad; independent 10° yaw/roll checks pass. This does not validate SDK velocity fidelity. |
| Quiet window still receives a nonzero ramped command | Target is zero from row800 onward. Actual command becomes permanently zero across all48 replicas by row841, endpoint **16.84 s**, well before scoring. All500×48×3 command entries in the scored window are exactly zero. | No command contamination in the quiet window. Retain the original score: 500 samples ×20 ms =10 s; recorded endpoints22.02–32.00 s span9.98 s. No scorer rewrite or reset deletion. |

The first converted W component is 0.99999779–0.99999964, consistent with the near-upright reset. The orientation comparison uses independent rotations and projected gravity, not an assumption that integrating reported angular velocity is exact. The installed SDK snapshot is copied from the earlier frozen readback receipt; this CPU review did not launch Isaac or reread live simulation buffers.

## Learning and target-delta checks

| Actual update | LR at update end | CAPS temporal | CAPS spatial | Weighted CAPS | Valid temporal pairs |
|---|---:|---:|---:|---:|---:|
| 1 | 1e−5 | 0.31087914 | 0.00224995 | 0.03131291 | 0.73567709 |
| 2 | 1e−5 | 0.33920515 | 0.00165433 | 0.03408595 | 0.95833334 |

The configured initial learning rate is5e−5; the two recorded update endpoints are five times lower. The receipt does not record every minibatch learning rate, so it cannot establish an intra-update maximum or a complete trajectory. Both CAPS losses are nonzero; their temporal/spatial ratios are138.17 and205.04. These are loss magnitudes, **not gradient shares or causal proof** that one feedback mechanism dominates. Two updates do not resolve critic adaptation or long-run stability.

The target-delta audit is nonzero in **all1,536 control/replica samples**, with RMS0.031832881–0.039999995 rad. The exact inherited pre-step code snapshots the previous target before processing the new action; the audited reward then subtracts that snapshot. For the eight fully traced replicas, none reset, and recomputing 47 consecutive target differences reproduces the recorded RMS within3.726e−9 rad. This is substantial executed target activity near the declared limiter, not an all-zero logging bug. It does not by itself identify the cause of oscillation.

## Reproduction and provenance

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tmp/claude_fable_architecture_independent_001/review.py \
  --raw artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_002/raw
```

The script prints deterministic JSON and does not write inputs. It verifies all52 published raw payload hashes and sizes, the exact598-entry source manifest, eleven source-bound code copies, native003 builder identity, and the preserved installed-SDK source snapshot before numerical replay. It needs NumPy and SciPy, not Isaac or Torch. `report.json` is its actual result. `ARCHITECTURE_RESULT.json` preserves the model's complete advisory output. Source copies were fetched read-only from source002 or matched byte-for-byte to its manifest; `build_source.py` is a separately bound preparation file, not a runtime source file.

Bound source manifest: `64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e`. Native003 manifest: `20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb`. No frozen source, raw trace, physics setting, acceptance threshold, or tracked repository file was modified.

## Publication note: required site contract

This is an ignored temporary review. Root owns publication under `docs/PROJECT_SITE.md`: add the bounded artifact paths to a new append-only `site/updates/` record, update the affected evidence/roadmap and execution snapshot where needed, and run `tools/project_site.py check --base <integration-base-SHA>` plus `build`. This review resolves diagnostic assumptions without advancing the locomotion acceptance gate.

Use the user's milestone framing: **first walking benchmark achieved; omnidirectional locomotion in progress; terrain/perception next; autonomous survey final**. This does not mean every Stage1 physical gate passed, does not equate the C-study with the physical four-bar robot, and does not turn successful software acquisition into Stage2 completion.
