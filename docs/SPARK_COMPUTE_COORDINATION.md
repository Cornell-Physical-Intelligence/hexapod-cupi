# Spark compute policy

James authorized full HEXAPOD compute ownership on 14 September 2026, including
stopping competing user workloads after preserving their recovery state.
One lead owns allocation and cleanup. Read [OPERATIONS](OPERATIONS.md) for the
procedure and exact host paths. Keep SSH, networking and host services available.

Reserve user compute for HEXAPOD between allocations. Keep the reservation
marker, scheduler masks, reconstruction entry block and queue lock until James
releases or changes the reservation. A competing workload request, completed
job or reconnect does not grant release. Restore recorded prior states after
an explicit release and identity checks.

Before dispatch, read `/home/orionh/SPARK_COMPUTE_COORDINATION.md` and verify
actual resources, processes, containers, producer descendants and both GPU locks.
Its recorded SHA-256 is
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
Bind changes into a fresh guard between allocations. Preserve old source packs
and bindings. This repository document is a policy reference, not live telemetry.

James approved the mass-corrected robot selected by
[`robot/active_model.json`](../robot/active_model.json) and the standing →
walking/stopping → terrain → survey sequence. Authorization does not pass
admission or behavior gates. Cleanup regression checks do not restart the
open-ended research sequence. The old continuation heartbeat remains paused.

The [reservation receipt](../artifacts/operations_2026-09-10/spark_exclusive_reservation_001/README.md)
and [automation block](../artifacts/operations_2026-09-11/spark_automation_block_002/README.md)
retain exact control and recovery evidence. The
[15 September pause receipt](../artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
retains its timestamp and historical scope. [The pre-cleanup policy](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/SPARK_COMPUTE_COORDINATION.md)
preserves earlier authorizations and quotes without adding them to new run context.
