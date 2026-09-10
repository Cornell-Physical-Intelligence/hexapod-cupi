# Proposed publication updates — no tracked edits made

Suggested artifact path: `artifacts/omni_diagnostics_2026-09-09/direct_omni_quiet_priority_analysis_003/`.

`docs/PLAN.md`: add this source-bound analyzer as the review path for the single quiet-priority objective ablation. Physical scoring remains identical to the previous formal0.04 comparisons. Optimizer diagnostics provide actual minibatch KL/LR and sparse weighted gradient evidence, with conditional quiet/moving loss counts; none constitutes physical admission.

`docs/NEXT_RUNS.md`: after the separately guarded native004 smoke or pilot terminates, root first audits full source/input/phase/checkpoint/ownership receipts, then runs analyzer003 against the retained raw diagnostics. A missing/failed receipt produces forensic evidence and no continuation. Compare same-pilot initial/final directions and moving→stop against the unchanged gates; compare preserved matched pilot evidence separately. Do not interpret a two-update smoke as convergence or extend a50-update run automatically.

`STATUS.md` and `site/project.json`: mark analyzer preparation verified on CPU, while retaining the latest actual checkpoint/recording and failed quiet verdicts until new actual evidence exists. This package alone changes no execution or benchmark status.

Append a record under `site/updates/` covering all published paths. Root must run `tools/project_site.py check --base <integration-base>` and `build`. This ignored preparation does not mutate integrated `tools/*.py` or a current runtime lineage; if publication placement changes that coverage, follow the existing lineage checks too.
