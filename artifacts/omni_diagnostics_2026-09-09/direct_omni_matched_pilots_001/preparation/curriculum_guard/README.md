# Curriculum pilot guard002

This fresh successor is ready for root review and dispatch. It pins the actual completed smoke002, source002 (`64144b2c…`), native003 (`20fbcd40…`), host002 (`19545653…`, runtime `52c4b0bd…`) and original direct315 checkpoint (`1971b782…`). The exact final bindings and actual preflight are in CPU_READINESS_FINAL.json and actual_readonly_preflight.json. The old guard001 draft and all failed smoke001 evidence remain unchanged.

The pilot is exactly 1024 environments × 24 controls × 50 updates, curriculum branch with zero CAPS temporal/spatial weights. It starts from the original policy. Six independent phases are fresh standing, initial constant, initial moving-to-stop, training, final constant and final moving-to-stop. Every original per-phase 600 s bound remains. Outer runtime is 3720 s plus 180 s stop, with a 70 minute restoration fallback. No next branch or performance promotion launches automatically.

Output: BASE/direct_omni_train_pilot_curriculum_001. Unit: hexapod-direct-omni-train-pilot-curriculum-001-20260910.service. Pause: BASE/forecast_pause_056. Source: BASE/direct_omni_train_source_002. Contract: BASE/direct_omni_train_preparation_002. Host: BASE/direct_omni_train_host_002/launch_train_spark.py. BASE is /home/orionh/HEXAPOD_runs/mock_length_study_20260909.

Seven prior campaign/job/pause/restoration files were read directly from Spark, hashed and copied under previous_owner. Smoke002 completed all four accepted phases and two PPO updates with exact reload; all four completed phase trees matched their immutable aliases, eight exact container name/ID inspections proved absence, and pause055 restored its original active timers. The completed smoke remains acquisition/infrastructure evidence; its physical failures are not waived or renamed as acceptance. Root's independent full terminal audit is copied unchanged for provenance.

The guard replays the full host/source/native completed-smoke contract before pausing and again before dispatch, checks all exact prior jobs/absence/timer restoration, and rejects changed, failed or missing receipts. Both GPU locks, unrelated-workload checks, weather semantics, current coordination hash `22c615d6…` and embedded restorer remain unchanged. The halo archive/deferral is separately owned and not modified here. Guard preparation performs no pause, dispatch or GPU operation. Root may defer this pilot if ongoing review identifies a source issue.

31 CPU tests pass. The actual read-only Spark preflight also passed without importing Torch/NumPy, launching an App, creating pilot output or modifying weather. The source code embedded in readonly_preflight.py is used only for its explicit validation functions; main is never called.

```
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_omni_train_pilot_curriculum_guard_002 -p 'test_*.py'
```

Actual 32-environment smoke collection took 1.593/1.262 s and learning .205/.096 s, reporting 427/565 transitions/s. Learn plus final verification took 3.488 s for 1536 transitions. These measurements do not establish 1024-environment throughput; no linear capacity or pilot completion-time claim is made.

CPU_READINESS_DRAFT.json and old test logs retain the earlier unbound draft history; CPU_READINESS_FINAL.json supersedes them. The frozen previous_owner receipt map and final runtime are the only launch bindings.
