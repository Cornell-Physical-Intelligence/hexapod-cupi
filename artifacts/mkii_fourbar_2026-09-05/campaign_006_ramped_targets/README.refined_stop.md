# Campaign 006: refined comparison stopped without admission

The [full nominal pass](README.md) remains preserved. Campaign 006 subsequently
failed in its refined phase; no qualification, scratch PPO or resumed PPO phase
ran. This supplement adds terminal evidence without changing the original README,
nominal/probe files or [original SHA256SUMS](SHA256SUMS).

The refined run is
`refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/`, source `9cd8d4c`,
with the same functional identity as nominal:
`c53071afdf320f6a9d6f91de09ddc6102de74df6f45a203e3a8166a8686548ca`.
It requested 32 environments, 1,000 standing + 2,400 driven control steps and
TGS 128/1 rather than nominal 64/1. Physics remained 1.25 ms × 16 substeps,
with the same active-target ramp, Kp = 30 / Kd = 0.60, mechanism and motor bounds.

## Stop decision and incomplete result

At 19:37:21 UTC, the standing-700 log reported a cumulative settled delivered
torque peak of **5.0157961845 N·m**, compared with **0.6681548357 N·m** in the
completed nominal run. The difference is 4.3476413488 N·m; the original
`max(0.05 N·m, 5% of refined peak)` convergence allowance at that point is only
0.2507898092 N·m. Increasing a cumulative refined peak cannot restore this
comparison. The operator stopped the incomplete run rather than treating the
earlier nominal pass as training admission.

The timestamped [operator decision](refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/operator_stop.json)
records SIGTERM to supervisor PID 1565424 at **19:38:31.782537 UTC**, the exact
owned container ID, the nominal report hash and the standing-700 line. Its
[captured log](refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/operator_stop_log.txt)
matches the decision's SHA-256. The final
[container log](refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/container.log)
continued to standing 800 at 19:38:37 UTC during shutdown, with the same settled
peak. This establishes at least 800 standing control steps, not completion of
the requested sequence or any driven test.

**There is no refined `report.json`.** None was synthesized. The saved log
metrics are partial observations, and the operator decision is separate evidence,
not a replacement validator report. At the last standing-800 log, the settled
window contained 9,600 physics samples per environment, with pin gap at most
20.0584 µm, at least four loaded feet and no reported non-foot contacts. These
partial measurements do not resolve the failed torque-convergence comparison.

## Terminal state and source preservation

[Supervisor evidence](refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/supervisor.json)
records:

| Field | Value |
|---|---|
| Execution / supervisor exit | `failed` / 1 |
| Interruption | `Supervisor interrupted by signal 15` |
| Final container exit | 137 |
| Cleanup | `removed_exact_id` |
| Validator report | absent; `missing_after_process_failure` |
| Source identity unchanged at finish | true |

The exact removed container is
`abd0bbbd3348e6a57bbd4e3120d0c543d2b84e395cb84060db9990bba01cee3e`.
The preserved [campaign snapshot](campaign.failed_20260905T193831Z.json) is
`failed`, with probe and nominal passed, refined failed, and no active phase.
Its error is `refined: supervisor exited 1; no retry or subsequent phase`.

An independent [read-only retrieval check](refined/retrieval_status.json) at
19:46:30 UTC confirmed that Docker could no longer inspect this exact container,
that no refined report existed, and that all **1,150 source files** still matched
the run's [source snapshot hashes](refined/hexapod-fourbar-validate-20260905T192754Z-80fadcba/source.SHA256SUMS).
The source hash-list digest is
`4b2bc1ddb2d4d7483ab2a6131acfad4af071ed34f2f4e1dd69bb5b293aabb245`.
The full source archive remains on Spark and was not downloaded here.

Remote campaign:
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/campaigns/fourbar-campaign-20260905T185810Z-c36097a7/`.
The new [dated manifest](SHA256SUMS.refined_stop_20260905T193831Z) covers this
supplement, the terminal campaign snapshot, refined logs/audit/supervisor/source
hashes and the retrieval record. The original publication still validates
against its original manifest. The zero-byte `admitted` file is only the host
resource barrier and grants no physical or training admission.

The next provisional controller experiment is separately named Kp = 30 /
Kd = 0.30. It must pass the same physics and acceptance gates; this failed
candidate is not reused as admission. No model, source, running process or
shared coordination note was changed while collecting this evidence.
