# Reconstruction launcher block after an external replacement

This is an append-only supplement to [operations001](../spark_automation_block_001/README.md). That earlier artifact remains a valid historical snapshot; its regular-file entry gate must **not** be described as continuing protection after the replacement documented here. No native allocation, PPO start, retry launch or general privileged-GPU partition is claimed. A later reservation-only probe passed, as distinguished below.

## What failed and what remained held

The actual reservation preflight rejected `/home/orionh/ithaca-reconstruction/queue-worker.py`: the pinned source gate hash was `5073f20f…`, but the file had been replaced with the original ungated bytes `0b5ae6e0…`. The [actual failure stderr](failed_preflight/reservation_probe.stderr), successful source-transfer receipt and exact two-file probe source/policy are preserved. The empty preflight stdout is hash-inventoried and excluded as a result. This was an operational preflight refusal, not a robot-physics rejection.

Root then captured the replacement's exact hash/mtime and original bytes in remote backups. The process or upstream uploader that performed the replacement was not identified by this evidence. The earlier enabled queue-lock holder remained active; its nonblocking lock test returned1, and the saved GPU query was empty. Thus the file gate was bypassed, while the independently held queue lock still prevented the ordinary queue launcher from acquiring its lock.

## New bounded protection and actual probes

Root replaced the known launcher path with a **real directory**, not a symlink. Python's directory-entry `__main__.py` is an unconditional standard-library-free `SystemExit` blocker,140bytes with SHA-256 `2ff83c0fada4bb18a0f568bec267fadf7ab9e7c7a60f83a8ecce38463ac9537c`. It does not import the foreign reconstruction module or CUDA. A reservation note and both remote original-file backups are hash-inventoried in [the actual block receipt](actual/reconstruction_entry_block_receipt.json).

The saved probes verified:

- `/usr/bin/python3 -I -S -B <launcher-directory> --queue <queue-path>` exited1 with the block message and no stdout.
- An ordinary `os.open` for writing at the launcher path raised `IsADirectoryError`.
- A nonblocking `flock` on the existing queue lock returned1; the enabled holder service was active/running with PID3814169.
- No CUDA process appeared in the saved query. The receipt records queue/output preservation and release only on explicit user instruction.

The replacement observation records original-file mtime1789087277.5266109 (2026-09-11 00:41:17.526611UTC). The stronger block receipt was saved at00:46:14.458501UTC. A separate [accepted reservation probe](successful_preflight/reservation_probe002.json) at00:48:04.142091UTC bound the new source/policy, all31 loaded/file masks and the live queue-lock owner PID3814169. All11 reload-yes fields remain visible. That probe explicitly records **zero native calls, no output created and no pause created**; it is not a native launch. Its exact two-file source, freeze and successful transfer receipt are also preserved.

These observations establish the tested directory/Python/file-open behavior. They do **not** establish that all upload protocols are unable to alter the path. A user or privileged process could remove the directory, change its `__main__.py`, upload elsewhere or launch CUDA directly. Root's continuing source/policy checks, masks, both compute locks and foreign-workload monitor remain necessary. No absolute or indefinite guarantee is made.

## Preservation and release

Remote `/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reconstruction_002/` holds `uploader_restored_original.py` and `removed_entry.py`, each6325bytes and both matching the original source hash, plus the replacement observation. The actual source contents, queued requests and outputs are not public in this bundle. The original operations001 unit masks, backup inventory and reservation evidence are unchanged.

A later user-authorized release must verify the current directory/blocker and existing queue-lock identities, preserve any subsequently uploaded files as evidence, verify original remote backup bytes, and restore only the intended launcher and prior scheduler state. No automatic release is implemented by this artifact; the packaging agent performed no remote mutation.

## Portable verification and lineage

```sh
python3 -S /absolute/path/to/spark_automation_block_002/verify_bundle.py
```

This reads saved public bytes only. It checks the manifest, original copy hashes, both complete probe freeze/transfer bindings, the failed old-gate path, replacement/backups, observed new-block/Python/lock semantics and the later successful reservation-only check. It does not execute either operator script, import the launcher or contact Spark. `ancestry/` binds the unchanged operations001 manifest and original reconstruction receipt without duplicating that whole artifact.

Root owns shared state/plan/site changes and the new bounded central update required by `docs/PROJECT_SITE.md`. This supplement cannot retroactively turn either failed preflight into a successful launch.
