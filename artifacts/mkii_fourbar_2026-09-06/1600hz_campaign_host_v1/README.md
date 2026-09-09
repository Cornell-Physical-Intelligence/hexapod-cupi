# Per-job shared GPU lock for the 1600 Hz campaign

This external host wrapper loads the frozen campaign and changes only `phase_argv`, prefixing each actual validation/training supervisor with `/usr/bin/flock --nonblock --no-fork /opt/wx/gpu.lock`. The campaign owns no persistent shared reservation. Each supervisor retains the shared lock until its own exit, including its existing exact-container cleanup, and still acquires the project lock `/tmp/hexapod-isaac-gpu.lock` through its original implementation. A shared-lock conflict fails that phase; it never forces entry.

Pinned source commit: `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`. Functional source identity: `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`. The wrapper verifies the two original entrypoint hashes before loading the campaign and then verifies the functional identity. Original source, resource gates, phase order, admission, checkpoint checks, signals, timeout bounds and cleanup remain unchanged.

The staged wrapper may be named `run_campaign_per_job_lock.py`; its content hash is unchanged by relocation. Invocation used for this campaign:

```sh
/usr/bin/python3 /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/host/run_campaign_per_job_lock.py \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/source \
  --source-commit c2af43ca0f384a4c2c7ab8f1d627f309dc78a683 \
  --asset-model mkii_fourbar_v5 \
  --environment-layout coincident_flat_origin_v1 \
  --output-root /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/campaign_001 \
  --wait-seconds 2700 --phase-timeout-seconds 7200 --full-timeout-seconds 21600
```

Append `--dry-run` for metadata-only validation: no output directory, lock acquisition, campaign invocation or GPU work. The original follower checks a different parent entrypoint and requires a separately pinned wrapper-aware version; changing argv recognition does not change qualification.

## CPU verification

`python -m unittest discover -s artifacts/mkii_fourbar_2026-09-06/1600hz_campaign_host_v1 -p test_wrapper.py -v` passes three tests against the exact frozen source: all five phases retain their original arguments with the exact prefix; composing phases launches no subprocess; metadata dry run starts nothing; the wrong commit is rejected. `linux_flock_cpu_probe.json` records a separate Spark Linux test on temporary files only. It confirms the actual util-linux flock opens a mode-0444 existing file through its read-only fallback, preserves the child PID with `--no-fork`, excludes a concurrent owner, releases between two jobs and does not change permissions. No real GPU lock or GPU workload was touched by that test.

## Timeout interpretation

The full phase retains its existing 21,600-second maximum. Its 1,000 PPO iterations contain 24,000 rollout controls plus final inference; startup and verification also consume the budget. A throughput estimate must come from actual 512-environment, 1600 Hz PPO progress, not the earlier 32-environment, 800 Hz standing profile. `progress.json` reports collection/learning durations and completed iterations. A hard timeout is not a graceful training pause: an explicit exact-run `stop_requested` observed at an iteration boundary produces a guarded checkpoint and a paused campaign, which needs an explicit later resume. No pause is requested by this wrapper.
