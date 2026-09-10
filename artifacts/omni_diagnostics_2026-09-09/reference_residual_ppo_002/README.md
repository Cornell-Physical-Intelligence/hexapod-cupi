# Residual PPO002: bounded integration and forward/stop retention passed

The new finite-position-residual PPO integration passed its complete bounded campaign. Fresh standing, zero-mean and sampled exploration calibration, and post-update quiet passed for all 32 replicas. The learner performed **two stand-only PPO updates, totaling 48 controls**. Both the initial and final checkpoints then passed separate full forward-and-stop retention trials.

This proves that the new reference, bounded residual controller, 846/849 observation interface, learner and immutable checkpoint path can work together without losing the demonstrated slow behavior. It does **not** demonstrate faster walking, learning to walk from scratch, omnidirectional motion or Stage 2 completion. No longer training is automatically admitted.

| Independent raw replay | Initial checkpoint | Final checkpoint |
|---|---:|---:|
| Forward command / duration | 0.005 m/s / 24 s | 0.005 m/s / 24 s |
| Measured forward displacement | 115.242 mm | 114.953 mm |
| Mean reported body forward speed | 0.004632 m/s | 0.004630 m/s |
| Original position/rate gap, limit 5 mm | 4.373 mm | 4.053 mm |
| Qualified measured landings | 11, all six legs | 11, all six legs |
| Reference stop latency | 5.54 s | 5.54 s |
| Scored quiet after another 2 s settling | 12.46 s, pass | 12.46 s, pass |
| Post-settle 400 Hz motor peak | 1.443 N·m | 1.425 N·m |

The differences between two cold trials are not a demonstrated improvement in walking performance. The calibration command and optimization command were zero. During calibration, the sampled residual reached 0.001395 rad maximum offset; all physical and quiet bounds passed. Post-update quiet again passed 32/32. Fresh standing and the smoke phase each peaked at 1.484 N·m after settling.

## Independent evidence

The [raw replay](independent_review/review.json) reproduces every calibration, post-quiet and retention report exactly using the frozen scoring code. It verifies all original substep counters, 2.5 ms cadence, and control-endpoint position, quaternion, joint, torque and reported-rate equality. Every control's 14 sensor clocks follows the installed eight-addition float32 recurrence; observed cache age is zero. These are simulator contact-cache checks, not real-sensor timing qualification. Final observation packets preserve all five reported-rate history slots, the 846-column actor/critic prefix, and separately calculated interval angle rates.

The [checkpoint review](independent_review/checkpoint_review.json) independently loads both actual artifacts through the source-bound RSL 5.0.1 CPU loader, checking actor, critic, normalizer and optimizer equality. It performs no learning or simulation steps. The initial checkpoint has zero optimizer entries; the final has 17, with changed actor weights after the two updates. Exact checkpoint SHA256 values are:

- Initial: `168b57602d66d07ce1e42f52905d2e0c44aaf8b58cc6c24cb5b667aa1849d70e`
- Final: `0920cc1ac125fee9b8de00e983b0ecb6e7fa04b5403ca2db5869082969e15f4e`

Reported joint rates still disagree with actual angle increments. During the 60.96-second post-startup smoke interval, the largest discrepancy was 1.541 rad for one joint/replica, despite almost unchanged measured angle. The raw channel remains intact and the interval-average channel remains separate. Native velocity fidelity is unqualified; the original acceptance bounds were not replaced.

The [read-only remote audit](independent_review/remote_audit.json) verifies the unchanged source009 (926 files), consumer002 (20), host002 (3), bridge (18), observation bundle (160), all 550 admitted assets, and the entire immutable smoke tree. All four exact owned container names and four recorded IDs were absent. The unit finished successfully with exit 0; pause043 restoration is recorded at Unix `1789032028.6315286`. This reviewer made no GPU or service changes.

## Exact raw preservation

All **80 original remote files**, about 616 MiB, are represented by [RAW_STORAGE.json](RAW_STORAGE.json). Ordinary files are copied byte-for-byte under `evidence/`. Two files exceeded the 100 MiB publication limit and are preserved as ordered raw chunks no larger than 64 MiB:

| Original file | Original bytes | Storage |
|---|---:|---|
| `run/smoke/physics_substeps.npz` | 249,262,704 | Four raw chunks |
| `run/smoke/trace.npz` | 202,617,062 | Four raw chunks |

Per-chunk and whole-original SHA256 values and sizes are recorded. There was no compression, recompression, truncation or NPZ rewrite. Both complete originals remain on Spark and in the local results directory. [Actual reconstruction tests](CHUNK_ROUNDTRIP.json) restored both large files and matched their whole-file hashes, then removed only the temporary test copies.

```sh
python3 /path/to/reference_residual_ppo_002/verify_payload.py
python3 /path/to/reference_residual_ppo_002/reconstruct_raw.py --output /new/complete_raw002
```

The portable standard-library verifier streams all chunks, checks every original hash and immutable receipt, and never launches archived code or a GPU. Reconstruction refuses an existing output directory. Numeric replay scripts and their logs are retained under `independent_review/`; they use the original local dependency layout described by [RECONSTRUCTION.json](RECONSTRUCTION.json). An initial review-only missing module-path attempt is preserved alongside the successful complete replay.

Exact [consumer/host/guard002 preparation](../reference_residual_ppo_001/README.md), [original integration preparation](../reference_residual_ppo_preparation_001/README.md), [source009](../reference_physics_009/README.md), [observation005](../reference_policy_observation_005_001/README.md), and [device proof](../reference_device_smoke_001/README.md) are referenced by published hashes instead of duplicating their complete trees. This terminal record contains no later directional, terrain or perception result.
