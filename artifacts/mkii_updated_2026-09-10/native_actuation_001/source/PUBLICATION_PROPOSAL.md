# Required central framework record — root integration only

Suggested immutable path: `artifacts/mkii_updated_2026-09-10/native_actuation_preparation_001/`.

Update `STATUS.md` to distinguish completed native003 import from the prepared coordinate/effort diagnostic. Update `docs/PLAN.md` and `docs/NEXT_RUNS.md` with this bounded next experiment, then provisional servo/support qualification and a fresh32×24×2 scratch smoke if their own gates pass. Keep hardware, standing and Stage2 acceptance incomplete. Update `site/project.json` to point to the new canonical preparation and actual native003 evidence, retaining all historical asset labels.

Add a new bounded `site/updates/<UTC>_canonical_actuation_preparation.json` record covering those Markdown/registry changes and this artifact directory; do not edit old records. Suggested summary: “Exact canonical coordinate/reset and400Hz external-effort checks are prepared after native import; no servo, standing or PPO is admitted.”

Run `tools/project_site.py check --base <integration-base>` and `tools/project_site.py build`; run current runtime-lineage checks if relevant tracked runtime coverage changes. Root owns tracked integration, publication and every GPU dispatch. This preparation modified only its new tmp directory and preserves native003 and draft design001 unchanged.
