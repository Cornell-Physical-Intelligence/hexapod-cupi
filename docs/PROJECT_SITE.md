# Progress page contributor contract

The static GitHub Pages site is the team's visual progress record against
[ARCHITECTURE.md](../ARCHITECTURE.md). It is built from `site/project.json`.
`STATUS.md` is generated from that same registry and must never be edited by hand.

Research is paused for [James’s handoff](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/JAMES_HANDOFF.md). Publication and
repository checks do not authorize native research or training to restart.

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
4. Define the next smallest step with James and the relevant lead before adding
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

Add a new `site/updates/YYYYMMDDTHHMMSS_name.json` record with:

- `id` matching its filename and ISO-8601 UTC `date`;
- plain-English `title`, `summary`, affected `areas`, and the next decision/action;
- `changes`: exact repository-relative paths or bounded globs covering the diff;
- `evidence`: existing repository paths or explicit public HTTPS references;
- boolean `no_project_impact`; if true, a specific `reason`.

Records are append-only. Do not rewrite an old record or a frozen result to make
validation pass. A presentation update is not a new experimental result.

```sh
python3 tools/project_site.py status
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
```

The check validates change coverage, source references, model-labelled media,
checkpoint/video identity, progress dependencies, accepted/ready definitions,
measured summary values and exact generated STATUS. Review evidence semantics
as well as these mechanical checks. Run relevant tests for changed behavior.

Source changes covered by the integrated release still require a new versioned
manifest, matching checker default and CI selection. Published manifests remain
unchanged. See [PIPELINE_LINEAGES.md](PIPELINE_LINEAGES.md).

## Publishing and presentation

`.github/workflows/pages.yml` validates pushes/PRs and deploys main to
`https://cornell-physical-intelligence.github.io/hexapod-cupi/`.
Only explicitly selected public files enter `site/dist`; generated output is
ignored by Git. Verify the deployed revision after publishing the integrated
change. Preserve the existing GitHub Pages destination and evidence permissions.

Lead with a concise mission and four visual checkpoint cards. Keep actual
walking videos visible beside their stage and scope, the interactive system
dependency graph expanded, and the approved CAD video directly accessible.
Give planned outcomes clearly labelled conceptual illustrations. Selectable
nodes expose inputs, outputs, existing source and the relevant qualification.
Show approved increment definitions, including M1, underneath their parent
capability without treating approved scope as accepted implementation.
Keep ownership, acceptance and long evidence notes in optional detail panels.
Historical recordings and figures remain visible in a separate gallery.
Keep a readable white scientific-poster layout, restrained rules, meaningful
labels, keyboard-accessible disclosures, mobile layout and reduced-motion support.
Never animate invented robot behavior as experimental evidence.

Write for a teammate opening the page for the first time. Name the robot model,
the behavior tested, the measured result and the work remaining. Explain technical
terms where they appear. Avoid internal shorthand such as “canonical admission”,
“causal envelope” and “qualification boundary”. Keep videos and the system graph
visible; use expandable sections for detailed test conditions and source links.
