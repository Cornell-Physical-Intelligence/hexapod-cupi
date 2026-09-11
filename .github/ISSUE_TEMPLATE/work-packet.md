---
name: Contributor work packet
about: One bounded behavior or evidence-producing experiment
title: ""
labels: ""
assignees: ""
---

## Outcome

State one observable behavior or experiment decision. Link the parent
capability and name the owner and human reviewer.

## Required context

Link the specific requirement IDs, design sections, contracts, package
instructions and input fixtures needed to do this work. Record the source
commit and selected asset/motor/observation contract when applicable; identify
the existing implementation to reuse.

## Scope and approach

Name the affected boundary/files and what the packet ends with. State open
questions and dependencies. For unfamiliar or consequential work, write a
short approach for the lead to check before substantial implementation.

## Acceptance and verification

List expected outputs, relevant boundary/failure cases and requirement IDs. Give the
focused test command or experiment procedure and the evidence to attach.
Declare measurement limits before the experiment. Preserve existing tests
and gates. Synthetic and hardware evidence must be labelled separately.

## Availability and blocking conditions

State expected work sessions/access, dependencies and the next unblock action.
If the packet exceeds the owner's available sessions or mixes independent
outcomes, agree on a split with the lead before expanding it.

## Completion evidence

Link the PR, checks actually run, artifact/configuration hashes where needed,
remaining limitations and review result. A failed experiment records its
result; it does not establish a successful capability.

## Roadmap and publication

Name the poster milestone (`walking`, `stage2`, `stage3`, or `mission`). Link the
new `site/updates/` record, evidence, and next action; update `site/project.json` and regenerate STATUS
when the displayed project changes. Record the contributor check and build
results required by `docs/PROJECT_SITE.md`.
