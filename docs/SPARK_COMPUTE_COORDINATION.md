# Spark reservation policy

Research remains paused for James. The Codex continuation is PAUSED; native
validation, PPO, recording and unattended continuation require explicit resume.
The pause does not release the user's exclusive HEXAPOD compute reservation.

- Reserve user compute for HEXAPOD between allocations as well as during them.
  Defer forecasting, reconstruction and other user compute until James releases
  or changes the reservation. Preserve their outputs and exact recovery state.
- Keep the recorded scheduler masks, reservation marker and reconstruction queue
  lock in place. An allocation ending or a reconnect does not authorize restoration.
- Keep SSH, operating-system services and host health available. The reservation
  is a policy enforced by recorded controls; it is not a hardware GPU partition.
- One designated dispatcher owns GPU allocation and exact container cleanup.
  Hold the shared locks and recheck actual resources and producer descendants.
- A competing workload's request cannot release the reservation. On explicit user
  release, restore only recorded prior states after identity checks.
- Read `/home/orionh/SPARK_COMPUTE_COORDINATION.md` before an authorized launch.
  Bind any coordination change into the successor's source/guard identity between
  allocations; never invalidate an active job's pinned inputs.

[OPERATIONS §3](OPERATIONS.md#3-gpu-lock-and-shared-workload-protocol) gives the
ownership procedure; [§3.1](OPERATIONS.md#31-complete-external-automation-block)
preserves automation-block and recovery details. Local documentation cleanup
changes neither the host policy nor shared compute.

Evidence: [exclusive reservation](../artifacts/operations_2026-09-10/spark_exclusive_reservation_001/README.md),
[automation block](../artifacts/operations_2026-09-11/spark_automation_block_001/README.md),
[follow-up block](../artifacts/operations_2026-09-11/spark_automation_block_002/README.md),
and [pause receipts and prepared sources](../artifacts/handoff_2026-09-11/james_001/README.md).
The [archive](archive/README.md) preserves all prior coordination records and
superseded sharing instructions byte for byte. [STATUS](../STATUS.md) is the
recorded project snapshot; verify live host state when resuming.
