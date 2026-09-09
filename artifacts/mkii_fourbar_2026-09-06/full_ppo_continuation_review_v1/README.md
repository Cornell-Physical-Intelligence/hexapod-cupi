# Independent continuation-coordinator review

**No blocking defect remains in the reviewed coordinator code.** All 24 CPU/mock
lifecycle tests passed independently in 0.184 seconds. The coordinator and test
hashes were identical before and after this run. No deployment, GPU launch,
runtime edit, fallback-branch change or existing-artifact edit was performed.

Reviewed [coordinator](../full_ppo_continuation_v1/continuation.py):
`a9def3ef253d97c05bdda0c535f19801d72958aabfa0aeb849c5573fb108b6d0`.
Reviewed [tests](../full_ppo_continuation_v1/test_continuation.py):
`ae5a8646523ac9e1222b76dbf87b28ca7788888d1b8cfba8f06bcc11bb604a04`.
`review.json` binds the exact command, files and result; stdout/stderr retain
the independent run. This review covers those bytes, not a later revision.

## Findings corrected before this assessment

| Initial finding | Verified correction |
| --- | --- |
| Container owner was taken from the observed container itself, making the ownership comparison self-confirming. | `discover_output` validates the supervisor's recorded owner and compares it with the exact observed container ID/name/label, then rejects later changes to the bound identity. Pause requests recheck the live process and container. |
| A replacement admission could be accepted and its new hash recorded between segments. | `pin_admission` records the original path/hash once, rejects rebinding and checks the bytes during monitoring, acceptance and capture. |
| Imported helper/host code could execute before its hash check. | A stdlib-only bootstrap verifies the pinned source/tool manifests and their files before loading helper/source modules. The modified-module sentinel test proves rejection precedes execution. |
| Wall-clock changes could extend the total or capture deadlines. | Enforcement uses persisted `CLOCK_BOOTTIME` deadlines and rejects a different Linux boot. UTC timestamps remain reporting fields. The per-job budget conservatively starts at the supervisor's process start ticks. |
| An inner pass flag, or an already-complete campaign label, could bypass final supervisor/campaign checks. | Both original-completion paths call the existing full-phase selector and strict input verifier. Rejected supervision cannot be marked original completion. |

The implementation agent made these changes. This reviewer made no changes to
the coordinator, frozen source, accepted gates or its dependency tools.

## Ownership, counting and restart behavior

- The coordinator is pinned to the reviewed `c2af43c` / 1600 Hz source and its
  deployment manifest. It cannot silently switch to the velocity-one fallback.
- Each continuation uses the existing bounded host launcher and a nonblocking
  per-job `/opt/wx/gpu.lock` with `flock --no-fork`. Waiting does not reserve the
  GPU. The source host retains its own resource gates and exact-container cleanup.
- The original full process must finish with verified cleanup before its partial
  checkpoint is accepted. The existing capture follower must have recorded
  `no_video`, exited, and never reserved a capture before continuation dispatch.
- Launch intent is saved before spawning; the spawned PID is saved before
  post-exec identity confirmation. An unresolved reservation becomes terminal
  `launch_uncertain`; it never causes an automatic retry. A confirmed active
  training or capture job is reattached on restart, not launched again.
- The full-update chain starts after scratch iteration 3. Every segment verifies
  checkpoint bytes, sidecar, next iteration, actor/policy and complete saved
  algorithm-state continuity, including optimizer/normalizer state. Requested
  and completed counts remain the original report values.
- The sequential test covers an honest original report requesting 1000 updates
  and pausing after 400, followed by a new 600-update segment. It records exactly
  1000 full updates, excludes the three scratch updates, reaches next iteration
  1003 and dispatches one capture. Active-training and active-capture restart
  tests prove no duplicate launch along those paths.
- Three full segments is a maximum, including the original full segment. Each
  host retains its 21600-second limit; the coordinator retains its overall
  18-hour bound. Physics/source failure, an unowned pause, missing cleanup,
  changed evidence, exhausted budgets or uncertain launch state are terminal.
- A pause on the final full update remains fail-closed: the missing final
  inference check cannot be replaced by a fabricated unpaused report, another
  training update or an unqualified capture. Original uninterrupted success
  leaves capture ownership with the existing follower.

## Limits and handoff

These are CPU and mocked-process tests, not a live Spark lifecycle trial or
proof of training throughput. Timing decisions use completed-update observations
but cannot guarantee that later updates fit the remaining deadline. Exact
policy/optimizer continuity does not preserve the simulator trajectory between
processes; the attestation explicitly records environment restarts.

The implementation owner was still preparing README/configuration and the
external tool manifest when these code/test bytes were reviewed. Bind that
configuration to the actual source, original campaign, follower, wrapper and
process identities, verify its final manifest, and run the CPU-only
`--preflight-only` path on the target Linux host before dispatch. This review
neither deploys the coordinator nor grants physical/PPO admission.

Run `python3 verify_review.py` to confirm the reviewed code/test hashes still
match, and `shasum -a 256 -c SHA256SUMS` here to check this evidence package.
