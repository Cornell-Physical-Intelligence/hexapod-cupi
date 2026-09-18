# Agent instructions

Keep replies under 200 words. Give no time estimates. Preserve other contributors'
work. `AGENTS.md` links to this file; edit this source and retain the symlink.

## Context and ownership

Read [README](README.md), then the affected package guide. Use
[ARCHITECTURE](ARCHITECTURE.md) for requirements and [CONTRIBUTING](CONTRIBUTING.md)
for checks and publication. Follow the source boundaries in README; navigation
and mission must not import simulation.

Use explicit search paths. Read old records to answer a named historical question.
Keep procedures in existing guides, assignments in GitHub issues, and design
review in PRs. Do not add another plan, handoff, decision log or task queue.
Edit `site/project.json` for progress and generate STATUS from it. Preserve
execution history in Git and large new results outside the source checkout.

## Model and evidence

Preserve the approved model and motor corrections in `robot/active_model.json`.
Use the detailed 19-body, 18-joint direct-drive robot after matching native
admission. Preserve the 0.040 rad / 20 ms comparison limiter and numerical gates.
Do not change tests or thresholds to admit a result.

Preserve checkpoint bytes, published manifests, raw captures, failed attempts,
incident/recovery records and the probe ledger. Use fresh result names and exact
model/source/input hashes. Retain historical labels and source revisions.
Stage 2 requires direction, transition, stop, torque, contact and visual
acceptance. Native evaluations must record ground-contact force and motor torque
before another optimizer/PPO sequence; these measurements add no acceptance limits.

James removed the Fable review requirement. Do not start Fable consultations.
Delegate bounded independent work only when the user authorizes delegation.

## Compute and publication

Read [OPERATIONS](docs/OPERATIONS.md) and
[compute policy](docs/SPARK_COMPUTE_COORDINATION.md) before dispatch. One lead
owns Spark; preserve competing workloads' recovery state before stopping them.
Keep both GPU locks, exact-container cleanup and the retained reservation.
Preserve SSH, networking and host services. Do not expose `.env.base` or secrets.
Recheck live state before each run. The mirror has no Git metadata. Keep the
historical heartbeat paused; cleanup checks do not resume the research sequence.

Run focused checks while editing and the required checks in CONTRIBUTING before
publication. Pair dependency changes with `uv.lock`. Follow
[PROJECT_SITE](docs/PROJECT_SITE.md) for each push and verify exact-revision CI
and deployed Pages. Covered source changes need a fresh manifest.

The standing publication authorization permits verified changes on a
`[netid]/[task]` branch, a PR to `main`, review and merge after green CI.
Use forward commits and preserve teammates' work. Do not push to `main` directly.
