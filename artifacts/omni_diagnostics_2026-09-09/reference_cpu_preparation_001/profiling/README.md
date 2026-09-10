# CPU scaling profile: scalar reference is not a 1,024-environment PPO path

**A serial 1,024-replica integration would be a substantial CPU bottleneck.** These are local arm64 CPU measurements, with one Torch thread, no GPU or Isaac, and seven independently prepared repetitions per size/phase. They are not Spark throughput or training-time predictions. Setup, cloning, simulator transfers and PPO work were excluded, so a real integration adds those costs.

| Phase | Reference, 1 replica | Reference, 32 | Reference, 128 | Observation, 1 | Observation, 32 | Observation, 128 |
|---|---:|---:|---:|---:|---:|---:|
| Stance | 0.560 ms | 17.401 ms | 69.999 ms | 0.869 ms | 22.661 ms | 92.199 ms |
| Swing | 0.612 ms | 18.004 ms | 71.517 ms | 1.000 ms | 26.354 ms | 103.817 ms |
| Landing | 0.595 ms | 17.642 ms | 72.595 ms | 1.033 ms | 27.821 ms | 110.945 ms |

The reference timings use frozen **wave003 with 7 mm lift**, in valid synthetic stance/swing/landing fixtures. Observation timings use the separately frozen **wave002-bound 740/743 encoder** and its actual-prefix/synthetic fixtures. These are distinct versioned measurements, not an admitted combined controller. The state API is unchanged, but a successor observation binding is still required. Exact identities, platform, repetitions and raw timing samples are in the two JSON reports.

Linear extrapolation from the measured 128-replica costs gives approximately **1.30–1.47 seconds per 1,024-replica control** for these two scalar paths alone. A 24-control rollout would therefore spend roughly 31–35 seconds in this CPU work before physics, transfers or optimization. This is an engineering warning about the current architecture, not a projected Spark training ETA. Do not allocate a large PPO run around this serial loop.

## Dominant work

The reference profiles consistently identify frozen `SerialGeometry.fk`, `ik`, and measured-COM/support preparation as the largest costs. Each replica separately executes small Torch operations, including repeated FK and Jacobian singular-value checks. Contact-hull checks and the phase state machine remain part of the cost; removing them would change the controller.

The strict observation encoder spends substantial time recursively checking nested dictionaries, converting them for JSON, computing same-step signatures, and running many small validation/encoding operations. These checks were deliberately explicit in the first contract. Their semantics must remain, but their representation should become typed batched arrays instead of per-replica recursive Python/JSON work. The `*_cprofile.txt` files preserve the actual cumulative profiles; profiler timings include instrumentation overhead and should not replace the unprofiled medians above.

The frozen geometry already accepts a leading batch dimension. An isolated probe using the **unchanged** geometry API reduced 128-replica IK from **30.785 ms serial to 0.918 ms batched**, a **33.5×** speedup, with identical checked joint values and validity masks in this representative fixture. At 32 replicas it was 18.9× faster. This is only an IK/FK kernel probe, not a vectorized contact-state controller, observation encoder, or physical result.

## Bounded next kernel

Preserve the frozen scalar reference as the oracle. The next independent boundary is a tensor-only geometry and reference-state extraction module that accepts named batched measured/target joint arrays and batched frame state, and returns FK points/Jacobians, measured COM, IK results and named state in the same units/order. Keep all tensors on the caller's CPU or GPU device; do not transfer each replica through NumPy. Bind 5 mm and 7 mm source/configuration separately even where the geometry is shared.

Before integrating it, require parity for heterogeneous stance/swing/landing batches, individual and partial reset rows, arbitrary valid runtime joint permutations, support/contact masks, finite/unreachable/joint-limit/singularity failures, and both explicit lift bindings. Preserve invalid masks and rejection reasons; never turn clipped diagnostic IK into accepted targets. Source/schema checks may be performed once when constructing an immutable binding, while all dynamic shape, finite, margin and timing checks stay in the step path.

Only after that kernel passes should the generator be split into explicit batched phase-state transitions, desired-point generation, batched geometry, and output validation/commit. Preserve provisional landing/contact confirmation, original endpoint checks, stop behavior and per-replica failure latches. The observation successor should consume those typed arrays directly and retain its complete 740-value state, history isolation and same-step consistency contract. Serialization remains an evidence/export operation rather than the control data path.

No full vectorized controller is included here. Root has authorized preparing this bounded kernel next in a separate directory. Neither this timing report nor faster CPU kernels establish that the robot walks well enough for PPO admission.

## Reproduction

`profile_cpu.py` binds exact frozen source manifests, constructs representative phase fixtures, excludes preparation from each timed interval, and refuses to overwrite a timing report. Run it from a fresh copy of this directory with the recorded source trees available:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B profile_cpu.py --kind reference
PYTHONDONTWRITEBYTECODE=1 uv run python -B profile_cpu.py --kind observation
```

Run the two benchmarks sequentially so they do not compete for CPU resources. All source originals were preserved. The synthetic reference fixtures prescribe ideal measured body/contact behavior solely to select valid code paths; no torque, support, tracking, clearance or terrain admission follows.
