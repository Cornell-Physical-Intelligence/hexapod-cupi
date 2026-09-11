---
name: Integrated capability
about: A sublead-owned outcome demonstrated across its component work packets
title: ""
labels: ""
assignees: ""
---

## Outcome and owner

Describe the observable capability and name its accountable sublead, backup,
and acceptance reviewer. State contributor availability and any access needs.

## Requirements and design

Link R/Q IDs in ARCHITECTURE.md §1, relevant ARCHITECTURE.md sections, the target
poster milestone ID and applicable G0–G6 gate, and the current STATUS.md context. Link decisions this depends on.

## Scope and dependencies

State the supported operating case, boundaries, and prerequisites. Distinguish
software fixtures, walking simulation, restrained hardware and field results.

## Acceptance demonstration

Specify input, expected behavior, failure scenarios, V IDs, measurement profile,
and artifacts that prove the capability. Existing gates remain unchanged.

## Work packets

Link bounded child issues with owners and dependencies. A design/experiment
packet must close unresolved interfaces before dependent implementation merges.

## Integration and completion

Record the shared integration command/procedure, result and artifact links.
Child PRs merged does not mean this capability passed. Close with the accepted
demonstration or a clearly labelled failed/withdrawn outcome; update STATUS.md
only with verified project-level progress.

## Roadmap and publication

Name the poster milestone (`walking`, `stage2`, `stage3`, or `mission`). Link the
new `site/updates/` record, evidence, and next action; update `site/project.json` and regenerate STATUS
when the displayed project changes. Record the contributor check and build
results required by `docs/PROJECT_SITE.md`.
