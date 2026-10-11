# Spark compute policy

GeoData and CUPI share the Spark. GeoData owns the installed Slurm setup.
The CUPI workspace cleanup retains the Isaac/Docker launcher and both GPU
locks. It changes no GeoData service, Qwen workload or shared Docker storage.
Workspace permissions do not allocate the GPU. Check live workloads before
each allocation and use the recorded sharing policy below.

The program lead authorized full HEXAPOD compute ownership on 14 September 2026, including
stopping competing user workloads after preserving their recovery state.
One lead owns allocation and cleanup. Read [OPERATIONS](OPERATIONS.md) for the
procedure and exact host paths. Keep SSH, networking and host services available.

Reserve user compute for HEXAPOD between allocations. Keep the reservation
marker, scheduler masks, reconstruction entry block and queue lock until the program lead
releases or changes the reservation. A competing workload request, completed
job or reconnect does not grant release. Restore recorded prior states after
an explicit release and identity checks.

The launcher no longer pins `/home/orionh/SPARK_COMPUTE_COORDINATION.md` or
checks this reservation. It shares the GPU with other workloads and records them
in each job report. The file's last recorded SHA-256 was
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
Preserve old source packs and bindings. This repository document is a policy
reference, not live telemetry.

The program lead approved the mass-corrected robot selected by
[`robot/active_model.json`](../robot/active_model.json) and the standing →
walking/stopping → terrain → survey sequence. Authorization does not pass
admission or behavior gates. Cleanup regression checks do not restart the
open-ended research sequence. The old continuation heartbeat remains paused.

The [reservation receipt](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/operations_2026-09-10/spark_exclusive_reservation_001/README.md)
and [automation block](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/operations_2026-09-11/spark_automation_block_002/README.md)
retain exact control and recovery evidence. The
[15 September pause receipt](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
retains its timestamp and historical scope. [The pre-cleanup policy](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/SPARK_COMPUTE_COORDINATION.md)
preserves earlier authorizations and quotes without adding them to new run context.
