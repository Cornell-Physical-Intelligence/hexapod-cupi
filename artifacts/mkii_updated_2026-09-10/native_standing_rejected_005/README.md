# One-robot 32/4 solver diagnostic rejected

Native5 completed 1,000 controls and 8,000 physics steps. Physical checks passed, but the unchanged quiet gate rejected LM tibia SDK velocity RMS **0.038802315 rad/s** and RM tibia **0.039215051 rad/s**, above **0.03 rad/s**. There were no missing six-toe support substeps, nonfoot contacts or requested torque saturation. Peak applied effort was **1.120248199 Nm**. Full numeric results and the original false verdict remain in `RESULT.json` and `terminal/run/standing/standing_report.json`.

This run did not time out. Native wall time was 131.1855 s; the owned phase exited successfully, then the original host validator rejected quiet standing and the owner exited 1. No 32-replica standing or PPO admission follows.

Source004 checked un-authored articulation 32 position / 1 velocity iteration attributes, authored 32/4, then verified them after reset and all controlled steps. `solver_readback.json` preserves each original value and authored flag. These are USD attribute readbacks, not independent solver-backend introspection. Geometry, 153 SDF colliders, provisional 400 Hz PD, 1.6 Nm effort cap, neutral target, raw capture and acceptance gates were unchanged. This one-robot result does not establish acceptance or a solver-native cause.

All 38 original raw files / 234,663,061 bytes are preserved. The contact JSONL has lossless gzip transport encoding; `RAW_SHA256.json` and `RAW_ENCODING.json` bind the original bytes. No trace is truncated or rescored. Source 95, host 55, guard 58 and auditor 11 retain their original freezes and complete payloads.

The historical audit verifies unchanged canonical inputs, exact owned name and ID absence, pause013 per-job cleanup, and continued persistent reservation. It is not a live Spark status claim. The 1,200 s phase / 90 s startup budget, original failure and exact invocation `f70b9d4b929244699a5725e018b38ba9` remain recorded.

Run `python3 -B -S verify_bundle.py` for bounded streaming byte verification and the original solver semantic check. `REPLAY_RECEIPT.json` records the publisher's complete byte reconstruction and unchanged source004 validator rejection: `ValueError('Standing physics/quiet rejected; acquisition is not admission')`. To repeat that negative contract replay, add `--replay-directory <fresh external directory>` (about 235 MB uncompressed; files are retained). This launches no simulator or GPU job.

The canonical asset 9, original actuation admission 19 and ownership-only supervisor 926 are separately published dependencies with exact identities in the audit. Hardware calibration and future walking remain unresolved. Root owns the central project update and any next diagnostic; this artifact changes no acceptance gate.
