# Training artifacts

This directory keeps the local evidence needed to reproduce, inspect, and audit
the hexapod training work. Models and videos copied into `artifacts/` are
repository deliverables. The Spark's much larger raw checkpoint fleet remains
under `isaaclab/logs/` and is intentionally excluded, as are simulator caches
and duplicate handoff archives.

## Key binary artifacts

| Role | Path | SHA-256 |
| --- | --- | --- |
| Historical Phase 1 forward policy | `phase1_v5/checkpoint/model_200.pt` | `43129a598f348d433bc724fb9fcb51b6620e8732e623cace0dc6fb1789f2d0e5` |
| Historical single-policy video | `phase1_v5/video/model200_cmd0p30_single/rl-video-step-0.mp4` | `434401fd5bfa6a3c76951ad6e4e63f67954d73a2bcb5826e855f9f18b26c732c` |
| Historical 144-policy army video | `phase1_v5/video/model200_cmd0p30_army144/rl-video-step-0.mp4` | `1df4ee10f76ec227b773d2c247569b337196d7c4b5fe2c47dafaca12fc59e8a2` |
| Current Stage 2C candidate | `phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt` | `a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a` |

The Phase 1 videos demonstrate the corrected anatomical-forward axis but are
not evidence that the current Stage 2C stability gate passes. There is not yet
a Stage 2C showcase video because the current candidate still misses yaw and
high-speed deck-stability thresholds; its README records the exact measurements.

The repository also carries the full set of models and videos that were already
present in this local artifact mirror: the baseline policy/demos, the Phase 1 v4
demo, Phase 2 warmup checkpoints, Stage 1 recovery checkpoints, and Stage 2 seed
and continuation checkpoints. This preserves the local training lineage without
copying the Spark's hundreds of rejected intermediate probe checkpoints.

Recent probe directories also retain one `checkpoints/best_child/` model per
completed experiment. "Best child" means highest-ranked within that individual
probe, not admitted or hardware-ready; the probe README records why each one was
rejected. Only `checkpoints/current_best/` names the active research baseline.

Evaluation JSON, resolved YAML, checksums, compact analysis files, locally
curated checkpoints, and videos stay in Git. Large raw logs remain on the Spark.

Short 6-second probe screens contain 275 post-warmup samples and are explicitly
diagnostic-only (`formal_admission_eligible=false`). They can rank candidates
for triage, but a candidate must pass the canonical 10-second/475-sample formal
evaluation before it can replace the current-best checkpoint.
