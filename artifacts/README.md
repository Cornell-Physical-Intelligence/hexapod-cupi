# Training artifacts

This directory keeps the compact evidence needed to reproduce and audit the
hexapod training work. Raw checkpoint fleets, simulator caches, and duplicate
handoff archives are intentionally excluded.

## Curated binary artifacts

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

Evaluation JSON, resolved YAML, checksums, and compact analysis files stay in Git.
Large raw logs remain on the Spark and in the local ignored artifact mirror.
