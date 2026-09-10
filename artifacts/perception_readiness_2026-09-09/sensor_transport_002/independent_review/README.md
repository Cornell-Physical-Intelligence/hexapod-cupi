# Independent sensor transport review

No concrete blocker was found for adopting this bounded CPU transport correction. The frozen owner remains unchanged. All 15 owner tests and four independent checks pass on Python 3.12.12 / Torch 2.14.0, CPU only. The owner's preserved receipt separately reports Torch 2.8.0; this review does not rewrite that result.

The reviewed owner is `../phase3_sensor_transport_002`, with nine payloads and freeze SHA-256 `c6aab3c11cd64c3cde22950e546df4dc9970c8500f3ec8ddeed258f69a1ff781`. Every payload and the complete inventory were verified. Runtime SHA-256 is `b35c37152c2a7875a26acb47467e26c9102cb6f082d5dcf592a86b3410ea5a7a`; preserved original SHA-256 is `101f6a59488f702117e93da87354b0d68c3d475e50421f445a12d9caf1491336`.

| Contract | Independent conclusion |
| --- | --- |
| Frame ordering | Each row retains its newest valid capture, with enqueue sequence breaking equal-time ties. Late older frames cannot overwrite a newer valid sample. Invalid frames do not renew capture age. |
| Causality and clocks | Capture and read clocks are separately finite, nonnegative and nondecreasing. A capture after the read time is withheld even inside the existing readiness epsilon. Model reads prevalidate all stream clocks before draining any stream. |
| Reset | Selected rows lose both queued and delivered values and IMU initialization. Even an explicit list of every row retains the clock epoch. Only `reset(None)` permits clock rewind and layout reinitialization. |
| Inference rollout | Persistent queue, selected-value, ordering and IMU buffers are ordinary tensors, including when capture/read occur under inference mode. Selective reset outside that mode succeeds. |
| Nominal behavior | Default configuration is exactly equal to the original. All values, masks and public timestamps/ages match the original bitwise across the owner's 401-read configured-cadence test, including selective reset and a stale read. This is a bounded test, not a claim over every configuration or device. |
| Integration test | Owner and proposed repository test ASTs match after normalizing only the two source-path assignments. |

The independent oracle adds 24 deterministic seeds, six rows and 100 reads per seed. It computes each row's latest eligible sample from the full capture history, including latency inversions, equal capture times, invalid masks and an in-flight selective reset. It matches values, sequence, float64 internal capture order and the existing float32 public age semantics.

The public age channel remains float32, as disclosed by the owner. This review does not make it a new high-precision timestamp contract or alter the separate 250 ms terrain-map lease. Noise and dropout remain synthetic; shared random-generator behavior is unchanged. No CUDA, sensor hardware, map coverage, actor integration, terrain traversal or physical stop is qualified here. There is no new frame-drain API.

Reproduce from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tmp/phase3_sensor_transport_002 -p test_transport.py -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tmp/phase3_sensor_transport_independent_review_002 -p test_independent.py -v
python3 tmp/phase3_sensor_transport_independent_review_002/verify.py
```

Root owns adoption. Under `docs/PROJECT_SITE.md`, eventual integration must add the append-only site update for the runtime, test, original fixture and this evidence; update the presentation registry if its sensor implementation description changes, and run the site and applicable lineage checks. This tmp-only review does not change tracked files or execution state. The completed preview shutdown and source-inspection bundles were left untouched.
