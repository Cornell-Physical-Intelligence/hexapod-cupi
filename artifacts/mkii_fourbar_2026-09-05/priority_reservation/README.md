# Campaign 004 cooperative GPU reservation

The user explicitly requested hexapod priority over other Spark GPU work.
`reserve_campaign.py` holds the existing weather scheduler's `/opt/wx/gpu.lock`
for one exact already-running physical campaign. It does not kill workloads,
launch GPU processes, change CUDA quotas or modify the shared coordination note.

At 2026-09-05 04:56:24 UTC, the new guard entered its lock wait before the exact
old guard PID 1232035 was verified by `/proc/PID/cmdline` and sent SIGTERM. New
PID 1333865 then acquired the lock. The acquisition snapshot is preserved in
`reserved_20260905T045624Z.json`; it is historical evidence, not live status.

Live Spark status:
`/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/priority_campaign_004.json`.
The executed copy is the sibling `reserve_campaign_004.py`, with stdout/stderr
in `priority_campaign_004.log`. Its exact campaign is
`campaigns/fourbar-campaign-20260905T044533Z-5465207c/campaign.json`.

The guard releases on that campaign's completion, failure, pause or process
disappearance, on an explicit termination signal or the sibling marker
`release_campaign_004_priority`, or at 14:56:24 UTC (a ten-hour maximum).
The separate run supervisors still enforce their own timeouts, source identity,
memory, GPU ownership and shared-file checkpoint/pause checks. The reservation
does not bypass any of them. No simulator source bytes changed for this renewal.

The remote shared coordination note was left unchanged during the active phase
because any byte change requests a pause. Its old three-hour guard entry is an
earlier dated event. The repository coordination guide records the renewal;
refresh the remote note after a terminal campaign state while retaining any new
request from another agent.
