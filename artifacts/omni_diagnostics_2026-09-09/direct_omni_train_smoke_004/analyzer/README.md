# Native005 smoke and extended-run analyzer004

CPU analyzer bound to root's actual source004 build. No smoke004 or 500-update outcome is claimed here. Physical scorers, formal 0.04-radian comparison, quiet gates, raw checkpoint checks and separate SDK-rate versus interval-angle evidence are unchanged. All 13 inherited scorer/input/training function ASTs and their gate/metric constants are identical to analyzer003.

| Binding | SHA-256 |
|---|---|
| Native005 (28 payloads) | `0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f` |
| Source004 (599 files) | `aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e` |
| Plan | `b2286828148c89fb34624d38beaf84b389584f7bda4d2648512e017da4b50acb` |
| Origin | `850aa811e702dcfd0284862c0c27f475e82a5335aab06b96e0ea21ea51e6f8b7` |

Schema: `direct315_extended_native_v4`. Smoke remains quiet-priority32×24×2; pilot remains1024×24×50. Extended is one explicitly selected1024×24×500 branch. Caps and quiet_priority stay separate choices; analysis cannot choose, queue or admit either. The original checkpoint remains `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.

## Use

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tmp/direct_omni_train_analysis_004/analyze.py \
  --campaign /path/to/run \
  --cold /path/to/direct_omni_cold_results_001/run \
  --output /fresh/path/to/analysis
```

Runtime imports only NumPy and the standard library. It writes `REPORT.md`, `report.json`, and `INPUTS_SHA256.json` outside immutable inputs and rechecks every consumed input. Exit zero means a forensic report was produced, not that learning or physical qualification succeeded. Inspect `errors`, `evidence_verified`, original host receipts and the separate terminal audit. `Stage2_complete`, `automatic_continuation` and optimizer-based admission always stay false.

Required curated files are the same: campaign/phase/job receipts, training receipt/state/initialization/raw trace/joint trace/events, final and explicit decision checkpoints, and available initial/final constant/stop diagnostics. Intermediate native `model_*.pt` files are **not read locally**. For extended, keep all 500 ordinary autosave names/sizes/hashes in the separate remote terminal audit and explicitly list local omissions. The analyzer checks seven local decision hashes/relative iterations and the source-bound producer/host readback; it does not claim to have reconstructed the omitted ordinary inventory. A failed run can preserve its native prefix without manufacturing a final or completed decision receipt.

The extended complete receipt must declare500 updates,12,000 controls,1024 replicas,12,288,000 transitions,10,000 minibatches,14 sparse gradient rows, native indices1847..2346, and decisions1/10/25/50/100/250/500. Selected/source/host counts and the exact extended readback must agree. Missing, failed, stale or malformed evidence yields a forensic report, not a formatter exception or admission. Prior-source smoke is rejected.

## Preserved interpretation

All old smoke/pilot summary calculations and formatting remain identical for the same optimizer data. The only summary change selects additional gradient milestones100/250/500 when allocation is explicitly `extended`. Every minibatch's actual before/after learning rate and KL is retained; first/last loss windows remain disjoint windows of at most10 updates (491..500 for a completed extended run). Norms/cosines remain sparse measurements, not every optimizer step or Adam's parameter displacement. Pair counts are repeated optimizer presentations, not unique environment transitions. Aggregate temporal MSE cannot substitute for quiet target-step p95 or motor/contact evidence.

Cold formal004 remains the constant-case fallback for smoke; it has no moving-to-stop baseline. Initial/final stop comparisons require the same pilot's recorded initial stop run. Constant traces contain one replica per four-replica direction summary; sampled training joint detail covers only its declared subset. Neither becomes full1024 joint coverage. SDK velocity and adjacent-angle interval averages remain separate. Any trial reset invalidates quiet acceptance; planar excursion from a window containing a reset is not continuous physical drift. No threshold is changed.

## Verification

20 distinct CPU tests passed:13 NumPy/stdlib successor and inherited orchestration tests in `tests_successor_001.log`, six unchanged raw/scorer fixtures in `tests_physical_001.log`, and one complete extended phase-selection/failed-budget forensic check in `tests_extended_campaign_001.log`. The six physical-scoring fixtures use frozen CPU Torch/TensorDict dependencies only to generate synthetic inputs; the analyzer runtime has no Torch/RSL/Isaac/CUDA imports.

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_train_analysis_004 \
.venv/bin/python3 -m unittest discover \
  -s tmp/direct_omni_train_analysis_004 -p 'test_*.py' -v
```

Tests compare extended diagnostic counts to the frozen native005 stdlib validator, preserve exact parent scorer ASTs and old summary outputs, verify14 scheduled rows among10,000 minibatches, and reject missing final gradients, bad budgets, changed decision files and absent/failed receipts. No GPU throughput, actual extended learning or success claim is derived from synthetic fixtures.

`SOURCE_BINDING.json`, `SCORER_PARITY.json`, `CPU_READINESS.json`, `DEPENDENCIES.json` and the parent freeze bind this preparation. Root owns source/GPU/terminal audit and publication. Eventual tracked integration must follow `docs/PROJECT_SITE.md`; the supplied central-framework proposal covers only new evidence/plan entries and preserves all frozen inputs.
