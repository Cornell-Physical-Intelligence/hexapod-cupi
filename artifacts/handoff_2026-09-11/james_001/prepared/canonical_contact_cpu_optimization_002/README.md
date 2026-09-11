# Exact contact interpretation memoization

This separate CPU candidate reduces repeated interpretation work in `classify_contacts`. It does not modify frozen standing005, any gate, contact recording, geometry, servo, native API or physics setting. Root owns adoption and dispatch.

Native buffers often contain long ranges of byte-identical inactive patches. Within each sensor range of at least 32 records, the candidate caches only the validated interpretation of identical force/point/normal/separation bytes. Small ranges retain the scalar path to avoid hashing overhead. The length criterion chooses an implementation path; it is not a physical/contact threshold. The cache is discarded for every sensor range and every call.

Each original buffer index is still visited in the same order, overlap/capacity checks remain active, all patch records are emitted, and every force is multiplied and accumulated using the original expressions and order. In particular, no force reduction, float64 product substitution, point quantization, inactive-record omission, centroid proxy or persistent contact history is introduced. Invalid identical records cannot enter the cache: the first is rejected by the unchanged predicate. Byte keys distinguish signed zeros.

## Evidence and timing

The final candidate is exact against the current frozen source005 classifier on:

- All 8,000 retained complete one-robot rows, with every returned array's dtype, shape and bytes plus ordered patch JSON compared exactly.
- All 31 actual contact-count/inactive/nonzero strata, translated and replicated at native float32 precision for 2, 8 and 32 replicas: 93 synthetic replication cases.
- 218 actual exported event-adjacent body subsets from the previous 32-run analyzer002, containing 10,085 patches at original environment/buffer indices. Other sensors are explicitly empty fixtures; this is not a full replay of the 5 GB run.
- Ten focused tests, including short and cached long ranges, signed zeros, nonfinite/invalid normals, range overlap/overflow/capacity, cancelled nonfoot vectors, per-body accumulation and ordered float32 products with float64 accumulation.

| Local classifier-only measurement | Frozen parent | Candidate | Ratio |
|---|---:|---:|---:|
| Actual 8,000 one-robot rows, direct accumulated timing | 6.620 s | 3.901 s | 1.70× |
| Weighted 8,000 one-robot rows, five interleaved timing pairs | 5.568 s | 3.277 s | 1.70× |
| Weighted translated 32-replica fixture, same paired method | 203.973 s | 117.628 s | 1.73× |

These are local Mac CPU component measurements. They exclude native queries, JSON/compression, geometry clearance and GPU work; the synthetic 32 distribution is weighted from actual one-robot strata. They neither predict native 32 physical behavior nor promise a Spark wall-time reduction. Raw paired timings and platform details are retained. The first prototype cached short ranges too, adding overhead on common nonrepeated lists; its code/results remain in `history/`. Absolute timings across separate runs varied substantially, so only within-run paired comparisons support these ratios.

Candidate-versus-parent output is exact. Replaying original Spark-recorded local shape coordinates on this CPU differs in 7,988 rows by at most 2.78e-17 m; that inherited platform-level difference affects both implementations equally. We do not claim byte equality between regenerated shape coordinates and original Spark JSON. No raw input was rewritten or newly downloaded.

## Reproduction and adoption

From the repository root with the existing source/raw fixtures:

```sh
.venv/bin/python3 -B -m unittest discover -s tmp/canonical_contact_cpu_optimization_002 -p 'test_*.py'
.venv/bin/python3 -B tmp/canonical_contact_cpu_optimization_002/parity.py
.venv/bin/python3 -B tmp/canonical_contact_cpu_optimization_002/events_and_timing.py
```

`INPUT_BINDINGS.json` pins the exact source and consumed raw/derived corpus; `candidate_delta.patch` contains the sole classifier edit. `verify_bundle.py` verifies this frozen proposal's bytes. No candidate native execution is claimed. Any runtime adoption must be a separately versioned source with root's review and the required `STATUS.md`, plan, registry and new `site/updates/` record under `docs/PROJECT_SITE.md`; root must preserve all prior failures and native admission requirements.
