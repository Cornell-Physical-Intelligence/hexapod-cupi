# Prepared paired BC loss-weighting contrast

Status: **prepared and validated; neither candidate has been fitted**. Root approved these precise comparison semantics after reporting that both matched startup trials failed forward tracking. The latest source publication repair and root's execution review are still separate from this artifact. No maintained file, learner byte, checkpoint, dataset, native state, or qualification gate changed here.

`PROTOCOL.json` binds the unchanged source018 learner, migrated BC fit004 parent `dfb6d3…`, original 3,760-row BC dataset `f016af…`, unchanged native AMP prior `22f7b0…`, physical source files, parent adoption, independent source/data audit, and actual Fable review. It also binds the helper and both complete float32 weight vectors. The 23 declared rows are raw control200, source_kind1, and a nonzero command; they were selected from teacher-input fitting diagnostics, not successful policy qualification windows.

## Exact comparison

Both candidates start from the same migrated BC checkpoint, full Config and RNG, with 1,000 existing BC updates and zero PPO. Both run 1,000 additional CPU BC updates, batch512, with the original separate fresh Adam learning rate3e-4, gradient clipping1.0, velocity coefficient1, and BC-only velocity detach. Each fresh BC optimizer is discarded afterward. Normalizer buffers are loaded and never updated. The helper uses the maintained learner classes, ordinary strict loader, and ordinary strict checkpoint writer; it does not bypass source validation or migrate metadata.

The two arms use identical uniform draws with replacement. They are **loss weighting**, not changed sampling or duplicated data:

- `uniform`: original `mean((action-target)^2)` expression plus the original unweighted velocity MSE, in the same addition order.
- `onset_weight20`: multiply each sampled row's mean squared action error by20 for the 23 onset rows and1 otherwise, average the batch, then divide by the **fixed full-dataset mean weight4197/3760**. Add the same unweighted velocity MSE. The denominator is not recomputed from each minibatch.

This normalization keeps the expected average action weight at1. It does not amplify every action loss twentyfold. Velocity receives no row weighting. Shared global gradient clipping can still couple gradient magnitudes; that is existing BC behavior preserved in both arms, not a claim of identical estimator updates.

Each arm strictly restores the same parent RNG before fitting. The CPU Torch RNG advances through1,000 calls to `torch.randint(3760,(512,))`. Minibatch index hashes, per-row sample counts, and final Torch RNG hashes must agree between arms. Python/NumPy RNG and the parent's empty CUDA RNG list remain unchanged. There is no GPU RNG exercise or simulator-continuation claim.

The helper checks that actor/critic normalizers, critic weights, `log_std`, the entire AMP model and normalizer, both pre-existing PPO/D optimizer states, full Config, existing PPO metrics, and every non-BC counter remain exact. Only policy/memory/estimator weights, `bc_steps`1000→2000, and the expected CPU sampling RNG may change. Candidates remain update0/PPO0 with the original checkpoint schema and learner source hash.

## Prepared verification

Five synthetic tests verify row selection and full weight sums, rejection of an altered selection, exact original uniform loss and gradients, action-only weighting with the fixed denominator, and absence of RNG consumption in objective construction. These tests do not execute an optimizer or fit an actor. The read-only preflight validates the actual pinned checkpoint, data, velocity mapping, Config, source origin, onset row identities, and weight hashes. Full candidate-loop execution and its strict reload checks remain untested until the authorized fit runs; preparation is not an execution result.

The future execution records all-row, steady, first-onset, first-five-onset, and zero-prefix errors before/after fitting. It preserves pre-update sampled loss/gradient logs, sample counts, ordinary checkpoint save/load receipts, and partial failure evidence. Full-dataset deterministic actor and estimator outputs must reproduce exactly after ordinary strict reload. Failure does not trigger a retry or change the recipe.

## Interpretation registered before fitting

Compare each candidate against its exact starting checkpoint and against the equal-budget uniform control. Report requested-target errors and velocity errors per group; low pooled error is insufficient. Verify that any onset improvement does not conceal worse steady fitting, altered normalization, parameter corruption, or changed motor semantics. No winner or qualification threshold is declared automatically by this helper.

An improved fit is a candidate for the same separately bound native cold/warm evaluation, followed by all required unchanged walking, quiet, stopping, motor, contact, and support screens. It is not a Stage2 pass. Continued drift after improved onset fitting would not logically rule out an onset contribution, and neither outcome validates nearest-neighbor recovery labels. No AMP change is included.

## Proposed integration and execution

After root's source-publication repair and review, copy the helper and tests into the maintained experiment namespace without changing learner bytes, add their inventory entries, and bind the resulting helper hash in a fresh protocol if its bytes change. The standalone helper imports only the maintained learner. Production code need not import this preparation artifact.

Read-only preflight (already exercised):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B artifacts/restart_2026-09-14/paper_bc_refit_preparation_001/refit_bc.py --repo-root /Users/andreboufama/Documents/hex/hexapod-cupi --protocol artifacts/restart_2026-09-14/paper_bc_refit_preparation_001/PROTOCOL.json
```

Proposed future explicit execution, **not run here** (use the integrated helper and corresponding protocol if root first relocates it):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B artifacts/restart_2026-09-14/paper_bc_refit_preparation_001/refit_bc.py --repo-root /Users/andreboufama/Documents/hex/hexapod-cupi --protocol artifacts/restart_2026-09-14/paper_bc_refit_preparation_001/PROTOCOL.json --execute --output artifacts/paper_bc_refit_001
```

Expected outputs are `uniform/` and `onset_weight20/`, each with its actual candidate checkpoint and receipts, plus a paired `RESULT.json`. Existing outputs are rejected. Native execution remains root-owned and is not part of this helper.
