# Progress page contributor contract

The static GitHub Pages site is the team's visual progress record against
[ARCHITECTURE.md](../ARCHITECTURE.md). It is built from `site/project.json`.
`python3 tools/project_site.py status` writes an untracked `STATUS.md` from that registry, and each
Pages build publishes the same text. Git does not track `STATUS.md`.

## Update progress

1. Change a roadmap marker or recorded fact only when its evidence changes.
   Keep the stable IDs `walking`, `stage2`, `stage3`, `mission`.
2. Distinguish capability state (`needs_definition`, `ready`, `active`, `blocked`,
   `accepted`) from attempt result (`passed`, `failed`, `interrupted`,
   `inconclusive`). A failed investigation can close its issue while the
   capability stays blocked.
3. Keep backend, model scope, source reference and measurement window explicit.
   Numerical summary fields must match the referenced JSON payload. An accepted
   marker requires a named human acceptance, its scope and evidence. Ready/active
   work requires accepted dependencies and a complete next-step definition.
4. Define the next smallest step with the program lead and the relevant lead before adding
   it: outcome, owner, reviewer, issue, acceptance and source/fixture baseline.
   Until then `next_step` stays `null`. Do not invent assignments or dates.
5. Generate the text view with `python3 tools/project_site.py status`.

The registry's evidence timestamp and source revision describe measured results;
the build revision describes the published page. Neither is live GPU telemetry.
Current compute instructions remain in `docs/SPARK_COMPUTE_COORDINATION.md`.
Historical walking, canonical standing, CAD animation and physical qualification
must remain distinguishable. Do not infer progress percentages from commits,
files, test totals, training budgets or videos.

## Every repository change

Every push must include a corresponding paper/Pages update. This is the program lead's
explicit hard rule, reiterated on 14 September 2026. Validate the complete push
diff, build the paper and exercise the renderer before pushing. Evidence paths
must match on-disk capitalization so local macOS checks agree with Linux CI.
After pushing, verify the exact revision's checks and served Pages version;
repair failed publication before declaring it complete.

Add a new `site/updates/YYYYMMDDTHHMMSS_name.json` record with:

- `id` matching its filename and ISO-8601 UTC `date`;
- plain-English `title`, `summary`, affected `areas`, and the next decision/action;
- `changes`: exact repository-relative paths or bounded globs covering the diff;
- `evidence`: existing repository paths or explicit public HTTPS references;
- boolean `no_project_impact`; if true, a specific `reason`.

Records are append-only. Do not rewrite an old record or a frozen result to make
validation pass. A later record may declare `evidence_reference_corrections`: each
entry pins the older record ID/hash, exact original reference, canonical
replacement path and its hash. Only that named historical link is resolved for
presentation; original bytes and provenance remain accessible. Corrections never
waive exact-case checks for current references. A presentation update is not a new experimental result.

```sh
python3 tools/project_site.py status
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
```

The check validates change coverage, source references, model-labelled media,
checkpoint/video identity, progress dependencies, accepted/ready definitions,
and measured summary values. Review evidence semantics
as well as these mechanical checks. Run relevant tests for changed behavior.

Source changes covered by the integrated release still require a new versioned
manifest, matching checker default and CI selection. Published manifests remain
unchanged. See [PIPELINE_LINEAGES.md](PIPELINE_LINEAGES.md).

## Publishing and presentation

`.github/workflows/pages.yml` validates pushes/PRs and deploys main to
`https://cornell-physical-intelligence.github.io/hexapod-cupi/`.
The builder verifies selected files under `site/assets/` and resolves archived
references through `configs/archive.json` without loading old Git objects.
Only explicitly selected public files enter `site/dist`; generated output is
ignored by Git. Verify the deployed revision after publishing the integrated
change. Preserve the existing GitHub Pages destination and evidence permissions.

Keep each roadmap stage to a title, status, optional recorded media with one short
caption, one summary line and a Details link. Full criteria, ownership and next-task
context remain in the linked status/design documents. Keep M1 to a single row
linking its test specification. Do not add explanatory paragraphs between these
items.

Stages without uploaded progress media show no image, illustration or placeholder.
The program lead supplies their visuals when progress is available. Keep the interactive
system graph expanded and existing walking videos playable. Preserve historical
recordings and their model labels; never present a concept image as progress.

Write for a teammate opening the page for the first time. Use concrete results
and actions, explain necessary technical terms, and remove repeated descriptions.
Keep keyboard access, mobile layout and reduced-motion support.
