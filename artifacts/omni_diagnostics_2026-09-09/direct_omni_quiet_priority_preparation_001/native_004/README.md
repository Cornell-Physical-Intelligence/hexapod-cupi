# Native004: one bounded quiet-priority PPO ablation

This successor changes the direct315 temporal objective for consecutive exact-zero-command pairs and records optimizer diagnostics. It preserves native003's actuator, action/observation, curriculum, adaptive optimizer, standing and cold evaluation behavior. **No actual native004 GPU run or physical acceptance is claimed.** Root owns source construction, guarded smoke and any pilot dispatch.

## Exact experiment

`--direct-branch quiet_priority` uses the same original `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8` actor/critic/normalizers, std reset .1, optimizer reset and initial LR 5e-5 with unchanged adaptive .01 KL target. The actor/critic remain 315/318. Physics remains the original direct C-study setup, applied cap 1.6 N·m, .04 rad/20 ms slew, observation noise 1 and filter disabled. Source009 supplies the already used container ownership supervisor, not this robot's physical configuration.

- Smoke: 32 replicas × 24 controls × 2 updates, **quiet_priority**, then final constant and moving-to-stop diagnostics.
- Pilot: 1024 × 24 × 50 = 1,228,800 transitions, fresh original checkpoint; matching initial/final 12-case constant and 48-replica moving-to-stop evaluations; immutable decisions 10/25/50 and final strict reload.
- The same-source completed smoke campaign, accepted phase identities and terminal input integrity remain required. Historical smoke002/native003 cannot silently admit this successor.
- `curriculum` and `caps` remain explicit comparison selectors in the source, with their original coefficients. No extra run or automatic continuation is created by retaining them.

The existing valid mask excludes reset/termination crossings and changed latest command slices. Quiet requires both latest command triples exactly zero. For per-action mean square `d_i`:

```text
sum(valid_i * (1.0 if quiet_i else .1) * d_i) / max(sum(valid_i), 1)
+ .1 * spatial_MSE
```

The implementation retains the old operation ordering when quiet and moving coefficients are equal. At quiet priority it adds `.9 * sum(valid_i * quiet_i * d_i) / max(sum(valid_i),1)` to the original temporal/spatial loss. Both policy-mean passes retain gradients. No target clamp is inserted into the auxiliary objective, no actor output is forced to zero, and no observation history channel is replaced. The coefficient is an experimental choice, not a convergence promise. Both previous 50-update pilots still fail all 48 quiet trials.

## Optimizer diagnostics

Every minibatch records its relative update/minibatch index, existing KL, LR before/after the unchanged adaptive rule, and actual valid/quiet/moving pair counts. Mode-specific temporal means are count-weighted across optimizer minibatches; the historical aggregate temporal/spatial/weighted keys retain their original arithmetic.

Only minibatches **1 and 20** of updates **1, 10, 25, 50** compute actor-gradient decomposition: PPO surrogate minus entropy, weighted quiet temporal, weighted moving temporal and spatial terms. Actor and critic parameter sets must be disjoint. The helper uses the already constructed graph and `autograd.grad(..., retain_graph=True, allow_unused=True)`; it makes no model forward, random draw, normalizer update, `.grad` assignment or optimizer step. Zero quiet gradients have zero norm and a null cosine. It records the component norms, PPO/quiet cosine, component-sum norm and actual combined actor norm before/after the existing **actor-only** clipping. The existing critic clipping stays separate. Algebraically separated components can differ by floating-point roundoff from the combined backward norm; no exact summation or causal conclusion is claimed.

The fixed allocation yields 20 minibatches/update, 40/1000 rows and 2/8 sparse gradient rows for smoke/pilot. The receipt field is `optimizer_diagnostics_schema="direct315_actor_gradients_v1"`; each `optimizer_updates` element has `minibatches`. `validate_optimizer_diagnostics` is stdlib-only and rejects missing/reordered/nonfinite rows, inconsistent counts, wrong endpoint rate and missing/extra sparse samples. `validate_result` returns a compact `optimizer_diagnostics` summary for the host. Diagnostics are always enabled by the admitted `configure` path; the disabling switch exists for equivalence tests, not a CLI allocation option.

## Verification

Final command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_recovery_001/native_004 .venv/bin/python3 -m unittest discover -s tmp/direct_omni_recovery_001/native_004 -p 'test_*.py'
```

**23 tests pass** in `tests_successor_002.log`. Meaningful new coverage includes:

- exact parent regularizer loss, actor gradients, normalizers and both RNGs when coefficients match;
- the quiet-only .9 increment with the unchanged all-valid denominator, both-command/missing/reset masks, unchanged moving and spatial behavior;
- preservation of preexisting parameter `.grad` buffers, model state and RNG by gradient measurement;
- two actual RSL CPU updates from the real original checkpoint, diagnostics off versus on: bit-exact actor/critic/normalizer/Adam states, global and private noise RNG, observations/history, LR and 48 synthetic controls; actual generated diagnostic rows pass the stdlib validator;
- inherited strict reload/inference-buffer tests and exact native physical-entry/stop behavior, plus negative host receipt tests.

This establishes mathematical/no-mutation equivalence in the tested CPU implementation. It is not a claim of bitwise CUDA-kernel equivalence or actual GPU readiness. The existing bounded smoke resolves first-run integration. The first full discovery attempt (`tests_successor_001.log`) failed because an inherited test fixture imported and cached historical `training/caps.py`; the successor test now pins its own CAPS module before loading the historical fixture. That failure is preserved. Earlier inherited logs retain their original scopes and are not summed into the final test count.

## Build and host seam

`build_source.py` still takes the exact cold589 parent (manifest `4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b`) and a fresh output directory. Root uses the same verified cold parent as native003. The builder adds `gradient_diagnostics.py` to its explicit runtime overlays, declares schema `direct315_quiet_priority_native_v3` and the new selector, and preserves the physical entry functions/assignments. It refuses changed parent input or an existing output. The expected new count is 599, to be measured by root rather than assumed.

`verify_inputs`, `runtime_arguments`, `validate_result` signatures and all phase folders/checkpoint mount paths remain unchanged. Host must admit branch `quiet_priority`, use its new source/native pins, and require quiet_priority for the same-source smoke. No broader physics sweep is part of this successor.

`FABLE_REVIEW_DISPOSITION.json` records the completed tools-disabled Fable5.1/max consultation (497-word answer) and source-checked findings. Useful caution led to early **and late** sparse minibatch sampling. Incorrect shared-clipping, LR-ceiling, RMS-to-p95 and convergence claims were not adopted. The exact consultation inputs/result are preserved under `review/`.

`FRAMEWORK_UPDATE_PROPOSAL.json` gives the required central registry and append-only update fields. `PLAN_NEXT_RUNS_PROPOSAL.md` supplies concrete edits for `docs/PLAN.md` and `docs/NEXT_RUNS.md`. Root applies these with any tracked publication under `docs/PROJECT_SITE.md`; this preparation itself changes no tracked file or frozen input.
