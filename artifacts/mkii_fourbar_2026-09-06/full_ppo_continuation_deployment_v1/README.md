# Full PPO continuation coordinator deployment

This is point-in-time deployment evidence, not a live status report. At 2026-09-07T01:31:33.281299+00:00, the exact coordinator PID2075580/start ticks106045128 was alive on boot `d49e857c-a99f-4dcf-815b-517b9efebf20`, in `monitoring` state. It had no active job, launch reservation/history or completed segments, and held no descriptor for either shared GPU lock. No stop marker was manually written and no GPU job was dispatched during deployment.

The reviewed code/config/test bundle is `../full_ppo_continuation_v1/`, manifest SHA-256 `b89e2daa6cd4d199675fcce678ea275805229227dbfa43e07a13869be3ec243d`. Its exact Linux CPU preflight passed with source SHA `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`, source commit `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, and all bound source/tool/config hashes unchanged. The preflight created no coordinator state directory. Parent task authorization then permitted this one detached CPU process.

- `linux_preflight_001.json`: exact staged file hashes, preflight argv, exit0 and output.
- `remote_launch_001.json`: durable reservation/spawn identity, exact argv, initial bound state, empty GPU-lock descriptors.
- `post_start_001.json`: fresh exact PID/start-tick/boot/argv and state check after startup.

Remote tools: `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_tools_v1`.
Remote state: `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_001/state.json`.
Remote log: `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_001.launcher.log`.
Remote durable launch record: `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_launch_001.json`.

The coordinator may request only the exact owned training checkpoint pause under its reviewed measured-deadline condition, then verify complete checkpoint-chain continuity before bounded continuation. It defers original-success capture to the existing follower. See the frozen code README for exact limits, at-most-once behavior and the last-update pause race. These deployment records do not prove that training or capture subsequently completed.

Do not start a duplicate coordinator or choose a new state directory to bypass a reserved/uncertain launch. Inspect current state and exact process identity before any operational decision. No recurring app automation was created.
