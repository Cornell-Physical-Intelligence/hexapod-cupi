# Spark reservation policy

James explicitly resumed canonical qualification on 14 September 2026 after
confirming the mass-corrected model in the viewer. The authorized sequence is
standing → walking/stopping → terrain → survey, using the URDF selected by
`robot/active_model.json`, SHA-256
`9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78`.
The 11 September pause remains a historical receipt. Resume authorization does
not assert that native validation, PPO, recording or continuation automation is
currently running; check their actual state before dispatch. It does not admit
physics or release the user's exclusive HEXAPOD compute reservation.

James additionally granted Codex full Spark compute ownership on 14 September:
"you can stop all other processes on the spark" and "take full ownership of the
spark, you have my permissions". The lead dispatcher may stop or suspend
competing user workloads and disable their identified restart triggers without
another approval. Record each workload's identity, original service/enable state
and restart path; preserve source, queued work and outputs. Keep SSH, networking,
essential operating-system services and host health available. This is current
authority, not a claim that resource reclamation has completed.

- Reserve user compute for HEXAPOD between allocations as well as during them.
  Defer forecasting, reconstruction and other user compute until James releases
  or changes the reservation. Preserve their outputs and exact recovery state.
- Keep the recorded scheduler masks, reservation marker and reconstruction queue
  lock in place. An allocation ending or a reconnect does not authorize restoration.
- Keep SSH, operating-system services and host health available. The reservation
  is a policy enforced by recorded controls; it is not a hardware GPU partition.
- One designated lead dispatcher owns all shared compute operations, GPU
  allocation and exact container cleanup. Other workers do not launch or stop
  Spark workloads independently. Hold both `/opt/wx/gpu.lock` and
  `/tmp/hexapod-isaac-gpu.lock` and recheck resources and producer descendants.
- A competing workload's request cannot release the reservation. On explicit user
  release, restore only recorded prior states after identity checks.
- Read `/home/orionh/SPARK_COMPUTE_COORDINATION.md` before an authorized launch.
  Bind any coordination change into the successor's source/guard identity between
  allocations; never invalidate an active job's pinned inputs.

[OPERATIONS §3](OPERATIONS.md#3-gpu-lock-and-shared-workload-protocol) gives the
ownership procedure; [§3.1](OPERATIONS.md#31-complete-external-automation-block)
preserves automation-block and recovery details. The canonical restart requires
those controls; verify their live state before dispatch. This document is not
live host or automation telemetry.

Evidence: [exclusive reservation](../artifacts/operations_2026-09-10/spark_exclusive_reservation_001/README.md),
[automation block](../artifacts/operations_2026-09-11/spark_automation_block_001/README.md),
[follow-up block](../artifacts/operations_2026-09-11/spark_automation_block_002/README.md),
and [pause receipts and prepared sources](../artifacts/handoff_2026-09-11/james_001/README.md).
The [archive](archive/README.md) preserves all prior coordination records and
superseded sharing instructions byte for byte. [STATUS](../STATUS.md) is the
recorded project snapshot; verify live host state when resuming.
