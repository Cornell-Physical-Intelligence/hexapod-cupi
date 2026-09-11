# Detailed-model PPO: smallest throughput path after admission

**Recommendation:** keep the first admitted 32×24×2 PPO integration small. Prepare a versioned contact-block writer and neutral-prefix snapshot alongside the independently owned classifier optimization. Defer a GPU control-path rewrite until component timing identifies its value. This is a CPU proposal: the inspected 32-env acquisition completed but admitted only 11/32; operational timeout and physical/quiet rejection remain separate. PPO002 still binds standing003, so final integration must explicitly bind the accepted successor.

## What is measured

The actual 32 session recorded 8000 physics steps / 1000 controls in **585.272s**, or **54.68 environment-control transitions/s**; complete native wall time was 606.564s. These are collection/recording times, not a PhysX kernel profile. Of 4,969,155,341 raw bytes, **4,620,466,111 bytes (92.98%)** were `contacts.jsonl`; substep NPZs were 306,089,312 bytes and control NPZ 39,318,700 bytes. Full 5 GB remains remote. [Receipt-derived numbers](EVIDENCE.json) verify all 95 standing004 and 50 PPO002 payloads; session/classifier/scorer bytes match actual standing003.

Existing Mac measurements replayed 8000 one-env rows and 31 strata replicated to 32. Their distribution-weighted estimates are 165.64 s classification,63.73s exact-mesh clearance and 38.75 s JSON. Those are **local replicated estimates**, not measured fractions of the 585 s Spark run. The classifier improvement underlying those numbers is already adopted; it cannot be counted again.

New bounded recording measurements use 128 complete one-env frames (8690 patches) and 218 selected actual 32 support-event excerpts (10085 patches). Every field/order and signed-zero value reconstructs the current JSON byte-exactly, including **15,187 inactive records** across both sets.

| Local in-memory encoding | One-env prefix | Selected32 excerpts |
|---|---:|---:|
| Current JSON bytes |3,241,695|3,574,380|
| Binary records + zlib1 bytes |167,391|65,497|
| JSON median milliseconds |42.41|48.38|
| Binary + zlib1 median milliseconds |20.88|38.59|

Seven repetitions ran on shared macOS ARM64 / Python 3.9.6; timings varied. Encoding is 1.25–2.03× faster in this run; **no end-to-end speedup is established**. The selected32 sample deliberately overrepresents inactive failure patches, so its 1.83% size ratio must not be applied to the full run. One-env size ratio is 5.16%. JSON+zlib alone reduced size but took longer to encode. [Full benchmark](BENCHMARK.json) includes ranges and five codec checks. It excludes dictionary construction, classification, device copies, disk writes and native dynamics.

## Narrow implementation boundary

1. **Writer and prefix, one new recording version.** Add `contact_blocks.py` beside a future `standing_session.py`; replace only full-line serialization/write and contact flush/close. Initially keep the classifier, exact-mesh reductions, live gate order and NPZ channels unchanged. Record little-endian float64 values, named body/category dictionaries, all original indices/flags/shape masks, sequence and explicit counter. The benchmark proves representability, not a production stream. Add checked lengths, block hashes, original-JSON digest and an iterator that reconstructs each original row. Keep every zero-force and inactive patch; unused native capacity was never in the ordinary JSON and is not falsely claimed preserved.
2. **Seal bounded immutable blocks.** Close at an 8-substep boundary, with a byte limit as well as a row limit; retain oversized individual records rather than truncate. Change `canonical_direct_ppo/neutral_prefix.py` to seal and bind/hardlink completed blocks. Currently it copies the still-active whole JSON stream before actor construction: the observed32 run would entail an additional 4.62 GB copy and hash pass. Closed chunks remove this duplication without reusing mutable files or resetting physics. `standing_score.py` consumes NPZs, so its numerical code stays unchanged; independent contact replay needs the new iterator.
3. **Use synchronous blocks first.** Avoid another queue/worker in the initial repair. Add async compression only if measured write stalls justify it, with bounded ownership, backpressure, propagated errors and synchronous drain before final receipts. Any write failure must preserve partial/native failure buffers and prevent further stepping; no dropped samples, hidden skipped steps or success before durable inventory.

## Scalable follow-on, only after measurement

The current substep performs 16 NumPy materializations across articulation/contact getters, host PD, force upload/readback, per-patch Python classification and full 19-body mesh clearance. PPO then copies actions to CPU and observations/rewards back to its device at 50 Hz. These are concrete code paths; their individual Spark costs are unknown. Add coarse accumulated timers around getter/copy, servo/upload, `sim.step`, contact classification, clearance, record encoding, NPZ flush, prefix scoring and PPO update. Preserve call order; synchronization overhead must be disclosed rather than attributed to PhysX.

The classifier owner is testing exact-output CPU vectorization independently. If device copies/CPU work still dominate, a later GPU-resident session should retain all 153 SDF meshes and exact clearance clouds, vectorize by named sensor/body, and batch telemetry transfer. Preserve **float32 force×normal then ordered float64 accumulation**, the distinct nonfoot aggregation arithmetic, source frame transforms and all malformed/capacity/overlap rejection cases. Unordered GPU atomics or fp32 reductions can change near-threshold results. GPU parity, memory cost and larger-scene scaling remain unmeasured; no 1024-env throughput claim follows from this design.

**Admission invariants:** same geometry/solver/servo/force settings; 50 Hz hold / 400 Hz PD;  actual 0.040 rad/20 ms limiter; 1.6 Nm cap and 0.5% requested-saturation allowance; exact contact classification, support/nonfoot/clearance, quiet, counter/time and joint gates. Keep raw SDK rates separate from interval-angle rates, all eight substeps and failure prefixes. Preserve the 405/408 actor/critic boundary, selected-row history isolation and strict checkpoint lineage/reload. The zero-command no-reset smoke is integration only; moving commands and episode resets remain separate reviewed contracts.
