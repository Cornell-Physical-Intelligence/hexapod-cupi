# Full-C reference physics screen 001

This is a frozen, CPU-reviewed preparation for fresh standing followed by one slow measured-contact wave. It has no Isaac result yet. It neither trains a policy nor qualifies Stage 2, terrain, perception, physical linkage hardware, or the requested final navigation controller.

The source is `source_001`, with 923 mapped files. Its `campaign_source_hashes.json` SHA-256 is `a13f1534f9f802871a2cacbc0be603eae2d80738905381da40691f7cd4beff7c`. The parent is the exact root-deployed velocity003 source, manifest `00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25`. The pinned 16-file C runtime and all original full-C study assets are preserved. Source preflight verifies every mapped byte, forbids extra files, and binds the exact C40/120 stance, lengths, 0.0025 × 8 physics, gains and motor limits. See `source_preflight.json` and `source_001/source_origin.json`.

The reviewed dependencies are residual002 (owner freeze `b9ef7920b1943506cdea11e04ce6567b887e4d0c65e206705c943f0b41b649a2`) and wave001 (owner freeze `ea01b80ea4414ca6081b8ada3c7cb9d6c010850623a8699edfb6f21cb14dc3ce`). The residual successor fixes persistent inference-tensor storage and has 12 CPU tests, including the actual wrapper reset regression. The wave has 9 CPU tests and 23 synthetic all-bearing/yaw/arc cases. Those synthetic cases are geometry/control tests, not measured physics.

## Dispatch contract

Root owns remote source transfer, independent SHA verification, GPU ownership, forecasting pause/restoration and the outer systemd guard. Only dispatch after live workload and both lock checks. Use a fresh remote source and output. Do not rerun this source by overwriting prior output.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 \
  <remote-source>/tools/launch_reference_physics_spark.py \
  --source <remote-source> --output <fresh-output>
```

The host runs exactly 32 replicas × 1000 controls of zero-residual standing. Only its completed, passed, exact-source admission allows one replica × 2400 controls: 4 seconds settle, 24 seconds at 0.005 m/s forward, then 20 seconds requested zero. This is a reference-only screen; no checkpoint is loaded and no PPO runner exists. The host mounts source and per-run copied assets read-only. Each phase has a ten-minute bound and a 90-second AppReady deadline, with 45-second startup tracebacks and unbuffered markers. There is no automatic retry or continuation.

Both GPU locks are held during each phase. The host inspects and cleans only the exact owned container ID, including Docker-client exit and creation races. Unknown Docker inspection failure does not establish absence. `cleanup_requires_owner_review` means ownership remains uncertain: the outer guard must resolve that before restoring competing GPU workloads. Root must independently verify final container absence, source and 550-asset maps, locks and authorized forecasting restoration after terminal state.

## Measurements and acceptance

Only named joint targets enter physics. No body pose, measured joint state, or contact is prescribed. Reset preserves the actually emitted noisy standing target and PD preload; episode age is explicitly zeroed. Runtime joint ordering and actual 95% soft limits are recorded. Contact buffers reserve 128 entries per prim, and the host rejects any contact/friction truncation warning even when a raw numerical gate passes.

Measured raw SDK quaternions are explicitly `quaternion_world_xyzw`; correctly converted `quaternion_world_wxyz` is provided for the existing quiet metrics. Link-origin position is paired with actual link-origin velocity; COM tracking velocity is separately named. Actual link transforms and angular velocities produce toe reference-point positions/velocities. Those points do not establish complete pad or shaft clearance. Missing measured contact points remain unknown, with a validity mask.

Every control captures state and termination reasons before automatic reset. A failure preserves the first failing physical sample, partial trace and generator state. Wave emission stops on invalid references, requested torque above 1.6 N·m, fewer than five supporting distal contacts, nonfoot contact or any terminal event. The final gate independently requires six distinct measured flights with confirmed measured touchdown, measured lift of at least 2 mm, actual forward motion at least half the requested speed, root-link displacement consistent with integrated link velocity, and at least ten contiguous seconds passing the existing quiet gates after finite reference stop plus two seconds settling. Planner labels alone cannot admit walking or quiet behavior.

Expected outputs are `campaign.json`, `jobs/{standing,wave}.json`, contact-data audits, phase `state.json`, `trace.npz`, `reference_states.json`, environment snapshots, the fresh `standing/admission.json`, startup/runtime logs, and copied input/source identities. A rejected reference is useful diagnosis and must remain rejected; it is not a reason to remove physical gates.

## CPU checks

Twenty focused telemetry, frame, pre-reset, contact, progress, exact-admission and host-cleanup tests pass in `cpu_tests.log`.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s tmp/reference_physics_adapter_001 -p 'test_*.py' -v
```

`build_source.py` refuses to replace an existing source. The source remains frozen; any runtime correction requires a new source/output version and new standing admission.
