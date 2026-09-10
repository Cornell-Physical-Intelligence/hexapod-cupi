# Reference policy preparation: observation contract and CPU scaling

The [CPU observation contract](observation/README.md) explicitly encodes the reference controller's landing, contact, stop and residual state. Its computed dimensions are 740 actor and 743 critic values. Root independently replayed all 14 tests. It is bound to frozen wave002 and residual002; the later 7 mm wave configuration requires a new binding. No learner, checkpoint or deployment is admitted.

The [scaling study](profiling/README.md) measured a substantial serial CPU bottleneck before a large PPO allocation. On the recorded local arm64 CPU, the separate scalar reference and observation paths cost about 1.30–1.47 seconds per control when linearly extrapolated to 1,024 replicas. This excludes physics, transfers and learning and is not a Spark ETA. The unchanged geometry API's batched 128-replica inverse kinematics was 33.5 times faster than separate calls in the measured fixture, with matching checked outputs. A complete batched controller is still needed.

Preserve the scalar implementations as parity oracles. Move source/schema validation to immutable startup bindings and use typed batched arrays for dynamic checks and control state; do not discard per-step validity checks or alter contact/landing semantics to improve throughput. The two timing paths have distinct source bindings and are not an admitted combined policy.

Both original freezes are retained byte-for-byte. Observation tests are self-contained:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s artifacts/omni_diagnostics_2026-09-09/reference_cpu_preparation_001/observation \
  -p 'test_*.py'
```

The profiling script intentionally retains its original frozen staging paths. Reproduction needs the exact source trees and a fresh output directory described in its README; do not run it over these frozen timing reports. No production or simulator source is modified by this publication.
