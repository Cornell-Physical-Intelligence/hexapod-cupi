# Spark operations

You use [`locomotion.launch`](../locomotion/launch.py) for canonical allocations.
It supervises one named module from a frozen package and cleans its exact
container. Read [compute coordination](SPARK_COMPUTE_COORDINATION.md) first.

## Environment

| Item | Recorded configuration |
| --- | --- |
| SSH | `orionh@spark-e26c`, Tailscale address `100.82.166.9` |
| Repository mirror | `/home/orionh/HEXAPOD`; this directory has no Git metadata |
| Isaac Lab | `/home/orionh/IsaacLab`, version 3.0.0, commit `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Container | Compose service `isaac-lab-base`; use `/workspace/isaaclab/_isaac_sim/python.sh` |
| Simulation | Isaac Sim 6.0.1-rc.7; Python 3.12.13; RSL-RL 5.0.1 |
| GPU | GB10; recorded PyTorch 2.10.0+cu130 |

Recheck versions from actual run receipts. The laptop/CI uv environment is
separate from the Isaac image. Do not sync `.venv` or run `uv sync` in the mirror.
Do not print, copy or commit `docker/.env.base`; Compose consumes that file.

## Before a run

The lead verifies the shared coordination file, actual processes, GPU use,
containers, services, producer descendants and both GPU locks. The current
coordination SHA-256 is
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
Bind those bytes to a fresh source and launch identity. Never alter an admitted
source pack or reuse an attempt's output directory.

Use `python -m locomotion.inputs pack` with a fresh local output and explicit
remote input root under `/home/orionh/HEXAPOD_runs/restart_20260914/`.
Transfer that directory, then pass its `inputs.json` to
[`prepare`](../locomotion/prepare.py). The pack hashes canonical assets and
starts no compute. The default binding declares `foundation_inputs_001` under the guarded restart
root. It does not claim those remote files exist. Transport with Python `tarfile` or disable macOS copy metadata;
extra files fail the frozen-tree check. From the frozen `source` directory,
invoke `python3 -B -m locomotion.launch` with the binding path and SHA-256.
The `--preflight-only` flag checks the host without starting native simulation.
Keep `PYTHONDONTWRITEBYTECODE=1` so imports cannot mutate the frozen tree.

Training and evaluation require matching one-robot and batch standing admission.
Use [`admission`](../locomotion/admission.py) to recompute the captures and create
the subsequent input declaration. A launcher success does not pass a physical
gate or qualify a policy. A source manifest does not grant admission.

## Ownership and cleanup

Hold `/opt/wx/gpu.lock` and `/tmp/hexapod-isaac-gpu.lock` from preflight through
container cleanup. Recheck competing processes after container creation and
throughout the run. A clean GPU snapshot does not prove that a producer cannot
launch a new child. Yield the owned allocation if contention or binding changes
appear, and preserve its partial output.

Signal a container by the immutable ID bound to this run. Check its name and ID
before acting. Docker client exit does not prove container exit; inspect again
after terminating the client to catch a creation race. Keep startup/deadline
bounds and contact-data truncation checks. Use the frozen launcher's
`--cleanup-only` path after a failed wrapper, then record its separate receipt.

James authorized the lead to stop competing user compute. Before that action,
record the workload identity, source, outputs and restart state. Preserve a
recovery image for an auto-removed container. Keep SSH, networking, operating
system services and host health available. Do not signal unidentified jobs.

## Retained reservation and recovery

Reservation root:
`/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001`.
Keep its `ACTIVE` marker and `reservation_policy.json`, the recorded user/system
masks, and `hexapod-exclusive-reconstruction-queue.service`. Verify its exact
helper and `queue-worker.lock` ownership through `/proc/locks`.

Keep the directory entry block at
`/home/orionh/ithaca-reconstruction/queue-worker.py` with its recorded
`__main__.py`. The earlier regular-file blocker failed and must not return.
Check loaded masked state, `/dev/null` links, inactive state and service PID 0.
Some masks retain `NeedDaemonReload=yes`; preserve that field's actual value.
A manual producer can still bypass scheduling, so continue process checks.

[Recovery receipts 007](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/spark_ownership_007/RECEIPT.json),
[008](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/spark_ownership_008/RECEIPT.json) and
[009](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/spark_ownership_009/RECEIPT.json) bind paused
reconstruction producers and saved images. [Incident 010](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/spark_ownership_010/READBACK.json)
records a process that exited before intervention. Use each receipt's own PID,
start time and command hash before restoration. Private source/unit copies stay
on Spark. The [prior runbook](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/OPERATIONS.md)
links earlier recovery receipts and their original scope.

Only James can release or change this reservation. Job completion and reconnects
do not authorize restarting deferred workloads. Restore the recorded prior
states after an explicit release and identity checks. The historical heartbeat
remains paused; do not restart it as part of cleanup.

## Evidence and access failure

Keep each attempt, including failed and interrupted runs, under its own identity.
Preserve raw captures, checkpoint bytes, file hashes, source/configuration and
cleanup receipts. Mirror selected public results without publishing credentials
or foreign workload source. Large full archives can remain on Spark if their
recorded paths and hashes allow retrieval. Compress raw logs without changing
uncompressed identities and retain selected actual policy videos.

If Spark is unreachable, complete local work and state that native checks remain
pending. Do not infer live execution from a saved dispatch, preparation directory
or old receipt. [STATUS](../STATUS.md) records results, not current GPU telemetry.
