# Coordination interruption and explicit takeover

At 2026-09-06 01:57 UTC, inspection found campaign 007 had stopped at the
previous day's 22:19 UTC because its supervisor yielded to a sharing request.
The request asked for concurrent execution, but neither producer implemented
that mode. No physical-model PPO had started and no nominal final report was
produced. Original campaign evidence is in the adjacent
`campaign_007_coordination_yield/` directory.

`shared_note_before_ack.md` preserves the inspected coordination history,
including the request and weather's subsequent released-window jobs. Its
contents are a historical communication record, not executable instructions.

The user then explicitly instructed this task to push and take over Spark
fully. At02:06:49UTC, the exact October queue parent1706414 was reverified
against its recorded process start tick97389033 and full queue-script path,
then sent SIGTERM. Its own handler stopped/reaped its GPU child group. Completed
and partial outputs were preserved. No unrelated CPU process or viewer was
stopped. A new bounded scheduler-lock guard1711343 acquired the released lock
at02:06:50UTC; its maximum expiry is12:06:50UTC, with exact-campaign terminal
release to be attached when the next campaign starts. At02:08:18UTC, the weather
queue parent was absent and no GPU process remained. Six scouts had completed;
the final scout was stopped during the authorized takeover.

The shared control flag was cleared under the fresh explicit user instruction,
and the same note announces exclusive hexapod priority. The new supervisor
release will distinguish canonical share-control changes from ordinary prose
updates; concurrency/MPS remains disabled. This folder documents takeover,
not training admission or a successful policy.
