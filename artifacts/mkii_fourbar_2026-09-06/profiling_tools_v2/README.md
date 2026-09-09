# External standing profiler v2

This optional diagnostic wraps the selected frozen validator in memory. It
does not edit the validator, SDK, source snapshot, physical metrics, or admission
gates. It does not run PPO. No GPU job was launched while preparing these tools. Version 1 completed its physical workload but lost profiler output when Kit exited before the outer wrapper returned. Version 2 publishes the trace and profile immediately after the validator writes its original final report, before environment/app shutdown. The version 1 files and failed attempt remain preserved.

The host supervisor uses the selected source's actual Compose argument builder,
source identity, workload/coordination gates, and exact-label/full-container-ID
cleanup. It opens the existing shared `/opt/wx/gpu.lock` **read-only** for an
exclusive flock, and the project lock read/write. It keeps the CPU admission
barrier and original Isaac launcher flags. Both tools are mounted read-only,
hashed before admission, and checked again after execution. The execution timeout
is 900 seconds. There are no automatic retries, scheduling, or reservations.

## Workload and timing

- Actual physical mimic v5, coincident flat layout, **32 environments × 100
  standing controls**, zero reset jitter, with the selected source's full
  physics cadence, solver, motor, sensor, and metric behavior.
- Controls **0–19** warm up; **20–49** form the first baseline;
  **50–51 only** have detailed host ranges and Torch CPU/CUDA profiling;
  **52–99** form the final baseline. All intervals are zero-based and half-open
  in the JSON report.
- There are **1,600 captures at 800 Hz**, or **3,200 at 1600 Hz**, and exactly
  **100 drains**. Every existing capture, check, and SDK method still runs.
  CPU regression tests exercise both cadences with the real `SubstepHook`.
- Baselines retain inactive pass-through wrappers, control timers and coverage
  counters; detailed region timers and the profiler are off. They are
  **unprofiled**, not perfectly uninstrumented. Control wall time includes
  `env.step`, the original drain, validator host checks and loop overhead.
- Profiler initialization, stop/finalization, and export are reported separately
  from control timing. There is no added synchronization after individual
  regions, sensors, or physics steps. The normal drain remains unchanged.

The wrapper records inclusive/exclusive host time and calls per traced control
for simulation/PhysX steps, original scene writes/updates, motor model and
budget, target scheduling, dones/rewards/observations, lazy articulation data,
all 31 sensor acquisition/update paths, metrics capture, and drain. It also
attempts the concrete native fetch methods from the inspected SDK. A read-only
native method remains unmodified and is explicitly listed as unavailable.
The runtime class source files and method source files are hashed, rather than
assuming that factory imports or the host SDK tree identify the running backend.

The exact current SDK attribute names were checked against the source evidence
in [the profiling plan](../profiling_plan_v1/README.md): articulation data
`_root_view` / `_physics_sim_view`, contact sensors `_contact_view` /
`_body_physx_view`, and the manager's `_physx_sim`. No sensor `.data` is read
merely to collect profiler metadata. The wrapper preserves the original metric
body, so arithmetic sub-blocks remain grouped and drain includes queued GPU wait.

## Run after the current owned GPU job has exited

Copy this directory to a separate tools directory on Spark, then run a
metadata-only dry run first. The baseline source below is the existing frozen
800 Hz RSL-compatible source. Its functional SHA-256 is explicit; this is not
the Git commit hash. A different source requires its separately verified hash.

```sh
python3 /home/orionh/HEXAPOD_runs/mkii_profile_v2/tools/run_profile.py \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/source \
  --source-sha256 d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54 \
  --solver-multiplier 1 \
  --output /home/orionh/HEXAPOD_runs/mkii_profile_v2/profile_001
```

The dry run prints exact argv and tool/source hashes; it neither acquires locks
nor starts a container. Add `--execute` to that exact invocation to perform the
bounded diagnostic, preserving the external launch log. The same wrapper can
profile the 1600 Hz candidate using its new immutable source/hash. Keep
solver multiplier and other settings identical when comparing performance.

Output files are `supervisor.json`, the original
`validation_report.json` plus its CPU asset audit, `profile.json`, `trace.json`,
`torch_summary.txt`, and container stdout/stderr logs. `profile.json` includes
runtime/source/SDK hashes, each control's timing and coverage, baseline summary
statistics, nested region counts/timings, and whether CUDA events were actually
observed. Inspect the trace for PhysX/Warp coverage before inferring GPU idle
time or native-kernel costs. Torch's timeline may omit some native streams.

The profile succeeds only if the short validator completes, every physics
capture/drain is accounted for, source bytes remain unchanged, and CUDA events
are observed. It **always declares `simulation_training_admission: false`**.
The preserved 100-control validator report cannot satisfy the existing full
1,000-standing/2,400-driven admission requirements. Profiling slowdown and
evolving physical state limit direct before/after comparisons.

## CPU checks completed

```sh
.venv/bin/python artifacts/mkii_fourbar_2026-09-06/profiling_tools_v2/test_profile_tools.py
```

Six tests pass: single-call/restoration for class methods, read-only native
method fallback, lazy-property preservation, full real-hook coverage at both
16 and 32 substeps with exactly two traced controls, exact final-report write
ordering, and a subprocess hard-exit regression proving profile.json is durable
before the validator exits without returning to the wrapper. Separate CPU lifecycle
fixtures using the actual host `compose_argv` also passed success, nonzero child
exit, runtime-gate rejection, postcreation Compose timeout, and precreation
coordination rejection. They verified exact-ID cleanup, original launcher
flags, diagnostic report paths, and successful exclusive opening of a read-only
shared-lock fixture without modifying its permissions. These tests establish
wrapper behavior, not actual SDK/CUDA profiling readiness or measured speed.
