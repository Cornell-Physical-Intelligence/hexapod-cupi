# 1600 Hz campaign capture follower deployment evidence

This records the CPU-only follower deployment at 2026-09-07T00:50:50.811230+00:00; it is not a live status report. The recorded follower PID was 2030164, start ticks 105810847, on boot `d49e857c-a99f-4dcf-815b-517b9efebf20`. It was waiting, had not dispatched capture, and held no shared GPU lock. No recurring app automation was created.

The exact campaign is `fourbar-campaign-20260907T004524Z-8592e14a` under `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/campaign_001`, with campaign PID 2021559/start ticks 105778155. The external wrapper uses a nonblocking `/opt/wx/gpu.lock` only for each actual GPU job. The waiting follower uses no persistent reservation.

## Frozen source and tools

- Source commit: `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`.
- Functional source SHA-256: `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`.
- Deployment manifest SHA-256: `8d6ad0b053e2b2a73e6610a51871443efdeb1234d7c44165dbf27a67498e66fb`; all 308 entries verified remotely.
- Actual imported timing: physics 0.000625 seconds, decimation 32, policy 0.02 seconds.
- Campaign wrapper SHA-256: `36cab0159740b24c10c46de15c8fb3e492685b885d647865813a6aaaffb9055b`.
- Follower v2 manifest SHA-256: `2d372547453266c255bea235404364cc178711701e3f55e7cd5fd9207b07008b`.
- Capture v2 manifest SHA-256: `d76e6601c9327f6c04bde05fb6012d399c8db1463b1ffe68c5f57b17c42959d3`.

`deployment_config.json` preserves the exact launch argv, paths, hashes and identities. `remote_preflight_001.json` records the live campaign/source checks before staging; `staging_001.json` records copying into new external directories; `remote_launch_001.json` records startup and immediate verification. `audit_001.json` is earlier local evidence and remains historical. `follower_v2.patch` records the reviewed wrapper and per-job capture-lock change.

The follower accepts only the exact campaign's successful 512-world, 1,000-update full phase with matching supervision, admission and final checkpoint. It makes at most one guarded capture dispatch, waits for exact container cleanup and verifies the final video. Failed, paused or mismatched campaigns yield no video. This does not establish locomotion quality or hardware admission.

## Read-only follow-up

The actual state is `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/capture_followup_001/state.json`; log is `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/capture_followup_001.launcher.log`. Inspect the current state and exact PID/start-tick/boot/argv before making any lifecycle decision; a PID alone is insufficient.

Do not relaunch the archived argv or reuse this output directory to create a second follower. The one-shot continuation proposal is separately versioned in `../full_ppo_continuation_v1/` and cannot dispatch after an original-success capture reservation. This deployment record does not authorize that proposal's deployment.
