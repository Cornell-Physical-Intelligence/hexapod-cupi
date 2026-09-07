# Complete capture comparison: portable staging v2

V2 fixes one fixture deployment error in the v1 bundle. Python eagerly evaluated
`HERE.parents[2]` as a `dict.get` default even when `source_dir` was supplied.
A bundle at `/workspace/complete_metrics` has only two parents, causing an
`IndexError` before any metric comparison. V2 selects the provided source path
with a conditional expression, evaluating the fallback only when needed.

Only `test_complete_metrics.py` differs from the seven executed v1 files. All
v1 evidence and code remain unchanged. Baseline, selected conservative runtime
candidate, standalone comparator and exact installed SDK functions are unchanged.
No runtime file, threshold, numerical operation or release manifest changed.

## Shallow staging proof

The seven-file bundle was actually staged at `/private/tmp` (two parents), with
an explicit absolute external source directory. `shallow_cpu_comparison_001.json`
passes all 8 tests: 800 complete 18-field rows at 32 and 512 environments are
byte-identical, all field differences are zero, and the source remains unchanged.
The imported timestep was 0.000625 s with decimation 32. No CUDA context was
initialized. `shallow_staging_proof_001.json` records the actual shallow path,
executed hashes, unchanged v1 hashes and sole changed file; its command output
is preserved in `shallow_cpu_comparison_001.log`. Temporary staging sources were
removed only after verifying their bytes still matched this bundle.

The original CUDA v1 failure was fixture setup, with no arithmetic comparisons.
A complete-row CUDA comparison of v2 is still required. This CPU evidence does
not constitute physical qualification or native sensor/performance evidence.
The full patch rationale, SDK provenance, coverage and limitations remain in
`../runtime_efficiency_integration_v1/README.md`.

## Exact bundle and command

Stage these seven files together; no repository test modules are required.

| File | SHA-256 |
| --- | --- |
| `compare_complete_metrics.py` | `7fe26439b66645929c1ebbb5e87c8eedb6d510f2461ca9ced2449ef6adb53ca4` |
| `test_complete_metrics.py` | `b10bca5f0f2a605271a4349a41ea790f6ba534a2e5f3b5d1cf488dfb9e7bbba5` |
| `baseline_validate_mkii_fourbar.py.txt` | `2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c` |
| `proposed_validate_mkii_fourbar.py.txt` | `811726701377ec9cc55672ac5df0f417bcb22eb112258bda259feed177becdb3` |
| `installed_isaaclab_math.py.txt` | `4ae6fe19d9b18c2747bf6546dc72f75cfe497ee2ba6046e95fabbeb33d958d33` |
| `installed_matrix_from_quat.py` | `48eca4847b0ac648711b9fd61fab554315c488265385bcbdbee4f0be3f017380` |
| `sdk_math_provenance.json` | `083188f4f1d29fb4f0ae84e9ff79ba3a80454ae9ad095c684ed4370da4a61c28` |

```sh
python /absolute/v2bundle/compare_complete_metrics.py \
  --source-dir /absolute/frozen-source \
  --device cuda:0 \
  --report /absolute/new-cuda-report.json
```

Omit `--device` for CPU. The existing report parent must exist, and reports cannot
be overwritten. CUDA is selected only explicitly; this tool does not create
remote jobs, leases or shared compute reservations. Use the existing bounded
GPU coordination flow. Source identity and imported timing are recorded from
the explicit external source and checked for changes before the result passes.
