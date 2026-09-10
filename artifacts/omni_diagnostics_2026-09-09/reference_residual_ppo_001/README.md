# Residual PPO001: parser failure before calibration

The first bounded residual-PPO attempt failed before its smoke process started Isaac. Fresh standing completed **32 × 1,000 controls** and passed all physical and quiet gates. The early command-line parser then interpreted `--device cuda:0` as an abbreviation of `--device-run`, replacing the exact device-proof path with `/outputs/cuda:0`. It failed while opening that nonexistent proof's `campaign.json`.

**No calibration, checkpoint creation, PPO update or retention evaluation occurred.** This was a startup error, not a rejected exploration experiment. The complete raw campaign, logs and independent standing review are preserved.

| Actual001 evidence | Result |
|---|---:|
| Fresh standing physical and quiet | 32/32 pass |
| Standing controls / recorded substeps | 1,000 / 8,001 including initial sample |
| Minimum post-settle distal support count | 6 |
| Post-settle 400 Hz requested/applied peak | 1.484 N·m |
| Smoke AppLauncher / calibration | Not started |
| PPO updates / checkpoints / retention | 0 / none / not started |
| Terminal campaign / systemd unit | Failed / exit 1 |

The [independent actual review](actual/review.json) recomputes standing from the complete raw arrays, verifies substep counters and 2.5 ms cadence, and checks exact control-endpoint position, quaternion, joint, torque and rate parity. It verifies every one of the 19 run files against both the remote audit and the prior compact retrieval map. All three standing NPZ files are included.

The read-only remote audit independently verified the unchanged 926-file source009, 19-file consumer001, three-file host001, 18-file bridge, 160-file observation bundle and 550 admitted asset files. Both exact owned container names were absent, as was the recorded standing container ID. The fast-failing smoke job recorded **no container ID**; none is invented here. The two other JSON files under `jobs/` are contact-audit reports, not additional jobs. Pause042 restoration is recorded at Unix `1789030414.2352366`. No service or GPU process was signalled by this audit.

## Separately frozen parser correction

[Consumer002](correction002/consumer/README.md) changes the early parser to `allow_abbrev=False`. This is the only executable change. Seventeen existing payloads—including learner, physical gates, observations, action semantics and plan—remain byte-identical. README updates and an actual-CLI regression make the successor a 20-file bundle. Old001 remains frozen.

The [independent correction review](correction002/independent_cli_review/README.md) reproduces001's failure and verifies002 with the full host flags in both orders. The two tests run three preflight-only subprocesses; none starts Isaac. The corrected three-file host and two-file guard are included as preparation, along with the exact [remote preflight002 receipt](actual/preflight002.json). These are **not physical002 results**, and no002 completion or training claim is made here.

Ten host tests and eight mock guard-restoration tests pass independently. One initial run of the copied host tests could not locate its external source009 sibling because its path assumes the original preparation layout; that attempt is preserved. The identical frozen tests passed from their original location. The correction source was not edited to change that assumption.

## Contents and verification

All 25 remote payloads are present: the complete19-file run, five pause042 files and preflight002. The immutable23-file compact failure/correction evidence is also retained with its original hashes. Its small raw subset is intentionally duplicated to preserve that freeze. New002 consumer/host/guard and independent CLI receipt are complete. The previously published [69-file preparation001](../reference_residual_ppo_preparation_001/README.md), containing original consumer/host and earlier independent session/RSL/physical reviews, is referenced by its full manifest rather than duplicated. Shared [source009](../reference_physics_009/README.md), observation and device-proof payloads remain referenced by exact hashes.

Run the portable verifier from any location:

```sh
python3 /path/to/reference_residual_ppo_001/verify_payload.py
```

It is read-only and uses only the standard library. `actual/review_actual.py`, `audit_remote.py` and `fetch_verified.py` preserve the original review/retrieval code and its original workspace paths; they are provenance, not launchers. The portable verifier does not run them, import Isaac or allocate a GPU.

This record establishes the first attempt's startup failure and the narrow correction's CPU behavior. It does not qualify native velocity fidelity, exploration, walking PPO, omnidirectional motion or Stage 2 completion.
