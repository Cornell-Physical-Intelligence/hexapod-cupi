# Direct315 cold baseline external guard

This prepares pause053 for exactly two host phases: unchanged new-plan standing (32 × 1,000 controls), then the preserved direct315 checkpoint diagnostic (48 replicas, 12 cases, 12 s). It performs no learning, checkpoint updates, video capture, automatic continuation or reference-policy transfer. Root remains sole dispatcher.

`launch_guarded_remote.py` uses the hardened admission guard's workload-ancestry check, graceful StormScope stop, both GPU locks, bounded owner and exact-container restoration. The embedded restorer is byte-identical to the reviewed parent. It restores only previously active timers and fails closed if owned-container identity/absence is uncertain.

- Owner: `hexapod-direct-omni-cold-001-20260910.service`.
- Output: `BASE/direct_omni_cold_001`; pause: `BASE/forecast_pause_053`.
- Outer bound: 1,800 s; stop grace: 180 s; fallback: 35 minutes.
- Source: `BASE/direct_omni_cold_source_001`, 589-file map `4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b`.
- Preparation: `BASE/direct_omni_cold_preparation_001`, freeze `eb87f1dd456771950f7c1bf50ed0a51c1fc6ccb4631217e29d99fa8a84717095`.
- Original checkpoint: `BASE/omni_repair_003/branch_a/inputs/original.pt`, SHA `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.
- Supervisor tools remain the complete pinned reference009 source, map `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`. They are distinct from the actual cold physics source. This guard imposes no assumed legacy runtime-tree hash; the host owns exact actual runtime binding.

The previous failed learning owner is bound to invocation `804a727558374268b185325305148b1a`, the exact learning campaign and training job, and pause052 restoration. Its one recorded container must be absent by both name and immutable ID. An empty live InvocationID is allowed after transient-unit collection only with all these exact historical pins; any conflicting nonempty ID or active state rejects. `previous_owner/` contains the three exact small receipts for CPU tests.

The guard calls the host's standard-library `verify_inputs(args)` before any pause and again immediately before dispatch. It checks source, supervisor, preparation, checkpoint and host identity; training must be explicitly false. A failure after the authorized pause but before any launch restores directly. An uncertain systemd-run launch stops only the exact new owner before restoration. The host then owns each per-phase lock, AppReady deadline and actual-container cleanup.

Host CLI is:

```sh
python3 BASE/direct_omni_cold_host_001/launch_cold_spark.py \
  --source BASE/direct_omni_cold_source_001 \
  --checkpoint BASE/omni_repair_003/branch_a/inputs/original.pt \
  --contract BASE/direct_omni_cold_preparation_001 \
  --supervisor-source BASE/reference_physics_source_009 \
  --output BASE/direct_omni_cold_001
```

`BASE` denotes `/home/orionh/HEXAPOD_runs/mock_length_study_20260909`; the actual guard passes absolute paths. Final host runtime is `fe55d1a554de2a9ec574ffc8b7a1c34ea677148c7f63d88ae7c63d693ba71eec`; its seven-payload freeze is `efbeac2a0bfb7ebf93efb9ec4dce33363607092b30a7e81ac6351950e1fa6083`. Both are bound and all payloads independently rehashed. The guard still refuses before any process call or pause if a binding is missing. Root verifies remote preflight and decides launch.

Twenty CPU tests pass: twelve ownership/input/call-order checks and eight actual embedded-restorer tests. They include changed source/restoration rejection, unknown Docker state, invocation handling, exact original checkpoint/no-training CLI, before/after-pause verification, and identical restoration semantics. No process, timer, GPU or source was changed by those tests. A separate read-only Spark check confirmed all three prior hashes, failed invocation and exact name/ID absence.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/direct_omni_cold_guard_001 -p 'test_*.py'
```
