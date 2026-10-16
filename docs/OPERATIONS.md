# Spark operations

Start with the [contributor workflow](../SPARK.md) for account access and
personal worktrees. This guide owns the runtime and recovery details.

You use [`locomotion.spark.launch`](../src/locomotion/spark/launch.py) for canonical allocations.
It supervises one named module from a frozen package and cleans its exact
container. Read [ownership](#ownership-and-cleanup) and the
[retained reservation](#retained-reservation-and-recovery) first.

## Environment

| Item | Recorded configuration |
| --- | --- |
| SSH | Use your own account at `spark-e26c.tailf4bbf2.ts.net`; the current laptop alias `spark` selects `james` |
| CUPI workspace | `/srv/cupi/hexapod`; personal clones belong under `/home/<user>/src/hexapod` |
| Isaac Lab | `/home/orionh/IsaacLab`, version 3.0.0, commit `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Container | Compose service `isaac-lab-base`; use `/workspace/isaaclab/_isaac_sim/python.sh` |
| Simulation | Isaac Sim 6.0.1-rc.7; Python 3.12.13; RSL-RL 5.0.1; wandb 0.28.2 |
| GPU | GB10; recorded PyTorch 2.10.0+cu130 |

Recheck versions from actual run receipts. The laptop/CI uv environment is
separate from the Isaac image. Do not sync `.venv` or run `uv sync` in the mirror.
Do not print, copy or commit `docker/.env.base`; Compose consumes that file.

## CUPI workspace

You edit your own clone and worktrees. The `cupi` group grants James, Shaurya
and Julian access to shared evidence. Add future users to that group and give
each user a run directory with the same access rules. The
[contributor workflow](../SPARK.md#know-where-your-files-belong) lists the
directories under `/srv/cupi/hexapod` and their access.

The workspace maintainer owns `inputs`, `evidence` and `maintenance`. Each run
parent inherits a default ACL that grants its submitting user write access to
container-created files and CUPI read access. Preserve that ACL when you copy
a pack. Reconnect SSH after an administrator changes group membership.

The recorded Isaac image runs as UID 1000, which maps to `orionh` on Spark.
That account has no `cupi` group membership. On a fresh attempt, grant that UID
read access to the frozen pack and a default write ACL for new output files:

```sh
setfacl -R -m u:1000:rX "$attempt"
setfacl -m d:u:1000:rwx "$attempt"
```

Check the image UID before use. Preserve the existing CUPI ACL entries. The
launcher mounts source and inputs read-only; the output mount permits writes.
Grant the same UID read access to new admission and compatibility bundles before
you mount them. A CPU admission process can create files under your host UID.

Use a fresh input bundle and attempt name. Pack a committed worktree, transfer
the frozen source, then run its launcher. You can edit the worktree after that
copy. The launcher verifies the frozen copy before execution and cleanup.
Keep `source`, `binding.json` and `PACK.json` with the attempt's `run` output.
Do not launch from a shared checkout or a symlink to a run directory.

Keep historical manifests unchanged when you relocate evidence. A maintenance
receipt maps each old path to its new path and records matching file hashes.
Prepare new bindings for new paths. The launcher accepts the legacy restart
root for historical compatibility; new CUPI runs belong under `runs/<user>`.
The default `configs/locomotion_spark.json` remains a legacy declaration, so
pass the new bundle's admitted `inputs.json` through `--inputs`.

### Prepared flat pilot

You can inspect the prepared pilot at
`/srv/cupi/hexapod/runs/james/flat_pilot_20260930_003/`. Its four packages are
`ppo_mlp`, `amp_mlp`, `amp_paper` and `tripod_reference`. The preparation receipt
records source commit `620d9aa9b23e0c9547893c513b57f77335eb4ef3` for these packages.
The input declaration is
`/srv/cupi/hexapod/inputs/flat_pilot_20260930_001/admission_128/inputs.json`.
The learner arguments retain the [frozen pilot constraints](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/df551b9018505daef56f48e0078282b608d9db45/docs/TRAINING.md#frozen-flat-pilot-issues-50-and-51).

You can read the inventory and file relocation map in
`/srv/cupi/hexapod/maintenance/cleanup_20260930_001/`. Use `relocation.json` to
find an old evidence path. Historical manifests keep their original paths and
bytes. Read `preparation.json` for package hashes, `evidence_verification.json`
for the 40-clip and standing checks, and `preflight.json` for the four package
checks. The CPU checks start no simulator or training. Coordinate a new host
check before dispatch.

The administrator completed the inventoried cleanup. All 60 selected legacy
paths are absent, and `hexapod-rl-training.service` is absent. The account
checks pass for James, Shaurya and Julian; shared-service identities match
before and after cleanup. The receipt is `cleanup_result.json` in that
maintenance directory, with SHA-256
`eadd4754d6914ac25d91e66bb16654ba05533c9c93745c29a389785ac915f8b5`.
All four host preflights pass after cleanup. Training has not started.

You can repeat the PPO host check without creating its output directory:

```sh
cd /srv/cupi/hexapod/runs/james/flat_pilot_20260930_003/ppo_mlp/source
python3 -B -m locomotion.launch \
  --bindings ../binding.json \
  --bindings-sha256 474bb1647dbd0a05f83866511763eaa924c8b9410a84958d88f832c7aa236639 \
  --preflight-only
```

## Before a run

The launcher records other GPU processes, containers and available host
memory, and it requires 16 GiB of available memory. It checks no host
reservation or coordination file, so any account with access to the run tree
and Docker can launch. Never alter an admitted source pack or reuse an
attempt's output directory.

Use `uv run python -m locomotion.spark.inputs pack` with a fresh local output and explicit
remote input root under `/srv/cupi/hexapod/inputs/`.
Transfer that directory, then pass its `inputs.json` to
[`prepare`](../src/locomotion/spark/prepare.py). The pack hashes canonical assets and
starts no compute. The default binding declares `foundation_inputs_001` under the guarded restart
root. It does not claim those remote files exist. Transport with Python `tarfile` or disable macOS copy metadata;
extra files fail the frozen-tree check. From the frozen `source` directory,
invoke `python3 -B -m locomotion.spark.launch` with the binding path and SHA-256.
The `--preflight-only` flag checks the host without starting native simulation.
Keep `PYTHONDONTWRITEBYTECODE=1` so imports cannot mutate the frozen tree.

For the [frozen flat pilot](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/df551b9018505daef56f48e0078282b608d9db45/docs/TRAINING.md#frozen-flat-pilot-issues-50-and-51),
prepare training and probe evaluation with `--allocation-profile flat_pilot_v1
--max-wall-seconds 21600`. A standing diagnostic above 128 robots takes the
same profile, since its capture grows at about 9 s per robot. Preparation binds that native deadline and the
22000-second supervisor cap. The launcher checks both values and retains the
400-second cleanup margin. Standard packs keep the 6200-second default native
deadline and 6600-second default supervisor cap. The pilot profile changes no
locks or cleanup targets. A timeout retains its last complete training update
and fails the allocation unless the requested update count has completed.

Training and evaluation require matching one-robot and batch standing admission.
Use [`admission`](../src/locomotion/evaluation/admission.py) to recompute the captures and create
the subsequent input declaration. A launcher success does not pass a physical
gate or qualify a policy. A source manifest does not grant admission.

## Ownership and cleanup

Hold `/opt/wx/gpu.lock` and `/tmp/hexapod-isaac-gpu.lock` from preflight through
container cleanup. The launcher recreates a lock file that host cleanup removed.
It records each other GPU process it sees during the run under
`other_gpu_processes_seen` in `jobs/standing.json` and continues. Compare timing
results with that record, and preserve the partial output of a failed allocation.

Signal a container by the immutable ID bound to this run. Check its name and ID
before acting. Docker client exit does not prove container exit; inspect again
after terminating the client to catch a creation race. Keep startup/deadline
bounds and contact-data truncation checks. Use the frozen launcher's
`--cleanup-only` path after a failed wrapper, then record its separate receipt.

One lead owns allocation and cleanup. On 14 September 2026 the program lead
authorized full HEXAPOD compute ownership, including the Spark lead stopping
competing user compute after preserving its recovery state. Before that
action, record the workload identity, source, outputs and restart state.
Preserve a recovery image for an auto-removed container. Keep SSH, networking,
operating system services and host health available. Do not signal
unidentified jobs.

## Retained reservation and recovery

GeoData shares this Spark with CUPI and owns its Slurm configuration. CUPI
workspace cleanup retains the Isaac/Docker launcher and both GPU locks. It
changes no scheduler, GeoData service, Qwen workload, other foreign workload
or shared Docker storage. Workspace permissions do not allocate the GPU. Check
live workloads before each allocation and use the sharing policy recorded in
this guide. Keep recovery payloads in place even when their parent directory
has a Hexapod name. Treat the records below as historical state, not live
telemetry; check the host before any recovery.

The launcher no longer pins `/home/orionh/SPARK_COMPUTE_COORDINATION.md` or
checks this reservation. It shares the GPU with other workloads and records
them in each job report. That host file's last recorded SHA-256 was
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`. Preserve
old source packs and bindings. The host state below stays in place until the
program lead releases it. Reservation root:
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

Reserve user compute for HEXAPOD between allocations. Only the program lead
can release or change this reservation. A competing workload request, job
completion or a reconnect grants no release and does not authorize restarting
deferred workloads. Restore the recorded prior states after an explicit
release and identity checks. The historical continuation heartbeat remains
paused; do not restart it as part of cleanup.

The program lead approved the mass-corrected robot selected by
[`robot/active_model.json`](../robot/active_model.json) and the standing →
walking/stopping → terrain → survey sequence. Authorization does not pass
admission or behavior gates. Cleanup regression checks do not restart the
open-ended research sequence.

The [reservation receipt](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/operations_2026-09-10/spark_exclusive_reservation_001/README.md)
and [automation block](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/operations_2026-09-11/spark_automation_block_002/README.md)
retain exact control and recovery evidence. The
[15 September pause receipt](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
retains its timestamp and historical scope. This guide replaces the
[compute policy file](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/b8bd271cf64399ea67465bcea9132b328b8c7df1/docs/SPARK_COMPUTE_COORDINATION.md).
[The pre-cleanup policy](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/SPARK_COMPUTE_COORDINATION.md)
preserves earlier authorizations and quotes without adding them to new run context.

## Evidence and access failure

Keep each attempt, including failed and interrupted runs, under its own identity.
Preserve raw captures, checkpoint bytes, file hashes, source/configuration and
cleanup receipts. Mirror selected public results without publishing credentials
or foreign workload source. Large full archives can remain on Spark if their
recorded paths and hashes allow retrieval. Compress raw logs without changing
uncompressed identities and retain selected actual policy videos.

If Spark is unreachable, complete local work and state that native checks remain
pending. Do not infer live execution from a saved dispatch, preparation directory
or old receipt. The [progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/) shows recorded results; it has no live GPU telemetry.

The [frozen startup-stall incident](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/3ccfd4a12aa7b95347c0884ac3ca466cad97b8aa/docs/incidents/2026-08-26-preappready-stall.md)
preserves the 26 August attempts and cleanup diagnosis. Use this runbook for
current execution; restore historical records through the archive guide.
