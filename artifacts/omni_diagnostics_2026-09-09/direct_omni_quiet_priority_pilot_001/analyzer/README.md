# Quiet-priority direct PPO analyzer003

Prepared CPU analyzer for native004/source003. No native004 simulator outcome is claimed here. The existing physical scorers, formal 0.04-radian target slew comparison, quiet bounds, checkpoint checks, and separate SDK-rate versus interval-angle evidence remain unchanged. `SCORER_PARITY.json` binds exact AST equality for all 13 inherited scoring/input functions and classes; the tests also compare the gate/metric constants.

Bindings:

- Native00443 freeze: `1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20`.
- Source003599 map: `ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62`.
- Plan: `eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2`.
- Source origin: `3b89ac6451e85020381994dce073b2b61af9caa08c79c12c887aff0cb41fb0c2`.
- Schema: `direct315_quiet_priority_native_v3`. Smoke requires `quiet_priority`; explicitly selected same-source pilot branches retain their declared selections. Original checkpoint remains `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.

The source values came from root's completed remote build and are independently consistent with the frozen host003 `FINAL_BINDINGS.json`. This analyzer does not replace root's full source/input/cleanup audit. Previous analyzer002 and all native inputs remain immutable.

## Use

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tmp/direct_omni_train_analysis_003/analyze.py \
  --campaign /path/to/completed-or-failed-quiet-priority/run \
  --cold /path/to/direct_omni_cold_results_001/run \
  --output /fresh/path/to/review
```

The runtime requires only Python's standard library and NumPy, with no Torch, RSL, Isaac, CUDA or network calls. It reads JSON/NPZ/checkpoints and hashes every read input before rechecking it at the end. It creates `REPORT.md`, `report.json` and `INPUTS_SHA256.json` in a fresh output directory outside the inputs. CLI exit zero means a forensic report was written, **not** successful training or admission; inspect `errors`, `evidence_verified` and the original host receipts. `Stage2_complete` and `automatic_continuation` always remain false.

Keep the actual full raw hash inventory and explicit local omissions with any publication. Analysis needs train receipt/state/events/raw traces/joint trace/initialization, final and decision checkpoints for a completed run, phase state/job receipts and available initial/final evaluations. Intermediate native `model_*.pt` autosaves are not read. Never infer unrecorded 1024-replica joint details from the sampled joint trace. Constant cases retain four-replica summaries and one traced replica per direction; stop traces cover all48.

The formal historical cold004 constant cases remain the fallback initial comparison for smoke. Historical 0.03 cases remain separate and are rejected by the unchanged formal scorer. Cold004 has no moving-to-stop baseline; a final stop screen cannot be described as improvement over that missing baseline. For pilots, the same-pilot initial constant/stop data provide paired comparisons. New source/plan pins do not authorize substituting different physics into historical evidence.

## Optimizer summary

`optimizer_summary.py` reads `direct315_actor_gradients_v1` without training imports. It preserves every recorded minibatch KL and before/after learning rate, rather than inferring the path from update endpoints. It reports the existing 1e-5 floor's exact recorded count; no new performance threshold is introduced. Update-end values remain separately available.

Quiet, moving, aggregate temporal, spatial and weighted loss summaries use disjoint first/last windows of up to10 updates. Window statistics are descriptive means of per-update values; exact update IDs and counts are retained. Conditional quiet/moving losses have distinct pair counts, and zero loss with zero quiet pairs is not evidence of quiet behavior. Counts are optimizer presentations including repeated PPO epochs, not unique simulation transitions. The all-valid objective denominator remains a property of the frozen native code.

Sparse rows remain exactly the recorded first/last minibatches of updates1/10/25/50: two rows for a two-update smoke, eight for a50-update pilot. These include the weighted PPO actor, quiet temporal, moving temporal and spatial component norms; PPO–quiet cosine; component-sum norm; combined actor norms before/after clipping. Null cosine is preserved as undefined/unavailable, including zero-norm components. Sparse gradients do not describe all optimizer steps or Adam's actual parameter displacement. Aggregate temporal MSE cannot substitute for the physical quiet target-step p95, saturation or contact metrics. No gradient sign, loss trend or learning-rate statistic admits or extends training.

## Failed and incomplete evidence

Missing campaign/phase/receipt files, null reload, missing optimizer fields and malformed/nonfinite diagnostic values yield an explicit forensic report. Producer-reported update counts are retained separately when missing raw files prevent independent replay. Completed claims still require the original strict checkpoint/raw/receipt tests; failure never turns into acceptance. Nonfinite values in the report copy become null with an error and retain the original immutable input hash. The actual failed smoke001 is retained under `actual_failed_smoke001_forensic/`: two updates replayed, reload unavailable, old source rejected, final evaluations absent. This is a historical formatter fixture, not a native004 result.

## Tests and reproduction

Eight new NumPy/stdlib tests run without training dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/direct_omni_train_analysis_003 -p 'test_optimizer_summary.py' -v
```

All14 tests, including the six unchanged inherited tests, pass in `tests_final.log`. The inherited full stop evaluator fixture uses the already frozen CPU Torch/TensorDict dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_train_analysis_003 \
.venv/bin/python3 -m unittest discover \
  -s tmp/direct_omni_train_analysis_003 -p 'test_*.py' -v
```

The first full system-Python test attempt lacked `tensordict`; its failure log is preserved as `tests_all_001.log`. No runtime edit was needed for that environment issue. Other logs remain chronological; `tests_final.log` is authoritative. Source-bound synthetic tests prove the summary/admission seams and preserve scorers, not new physical success. `DEPENDENCIES.json` binds retained local historical fixtures without duplicating large raw arrays.

Publication must follow `docs/PROJECT_SITE.md`. `FRAMEWORK_UPDATE_PROPOSAL.json` and `PLAN_NEXT_RUNS_PROPOSAL.md` are proposed edits only; root owns integration, source audits, GPU dispatch and final milestone claims.
