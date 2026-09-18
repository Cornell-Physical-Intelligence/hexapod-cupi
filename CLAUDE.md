# Contributor guide

Keep replies under 200 words. Give no time estimates. Preserve other contributors' work.
`AGENTS.md` is a symlink to this file. Edit this source for shared instructions.

## Read only the context you need

Start with [README](README.md) and [locomotion](locomotion/README.md) for walking
work. [ARCHITECTURE](ARCHITECTURE.md) owns mission requirements and system
boundaries. Read the affected package guide for its historical contract scope.
Package guides do not define the approved robot's pending hardware runtime.

Use explicit search paths. Read raw CAD, frozen source or old decision records
only to answer a named question. Do not create another plan, handoff, decision
log or task queue. Use GitHub issues for assignments and PRs for design review.
Keep durable procedures in the existing reference documents. Preserve historical
context in Git instead of copying it into new prompts or maintained guides.

## Source boundaries

- `locomotion/` owns the canonical simulator, reward, stock PPO, evaluation and
  guarded Spark launcher. `robot/active_model.json` selects its approved robot.
- `experiments/trajectory_optimization/` owns the optional optimizer and example
  comparison. Its native packers call the kernel.
- `packages/` retains shared contracts and historical runtime/navigation code.
  Core uses stdlib; runtime depends on core and stdlib; navigation depends on
  core and must not import simulation or runtime.
- `tools/` owns repository and asset commands. `configs/source_inventory.json`
  records ownership and the Git identities of retired source and context.
- `artifacts/` holds results and replay inputs. Production code must not import
  its frozen source copies. Historical task IDs retain their original semantics.
- `site/project.json` owns current progress. Generate `STATUS.md` from it.
  Retain decision-relevant findings; use Git history for the execution journal.

## Model and result invariants

Preserve the approved model and motor corrections in `robot/active_model.json`.
New work uses the detailed 19-body, 18-joint direct-drive robot after matching
native admission. Preserve the 0.040 rad / 20 ms comparison limiter and existing
numerical gates. Do not change tests or thresholds to admit a result.

Preserve checkpoint bytes, published manifests, raw captures, failed attempts,
incident/recovery records and the probe ledger. New results need fresh names
and exact model/source/input hashes. Keep original labels and source revisions.
A checkpoint save, video or source release does not qualify walking. Stage 2
still requires direction, transition, stop, torque, contact and visual acceptance.
Native evaluations must record ground-contact force and motor torque before
another optimizer/PPO sequence. These measurements add no acceptance limits.

James removed the Fable review requirement. Do not start Fable consultations.
Use bounded independent worker tasks only when the user authorizes delegation.

## Compute

Read [OPERATIONS](docs/OPERATIONS.md) and [compute coordination](docs/SPARK_COMPUTE_COORDINATION.md)
before dispatch. One lead owns Spark. James authorized full compute ownership;
identify competing workloads and preserve recovery state before stopping them.
Keep both GPU locks, exact-container cleanup and the retained reservation.
Preserve SSH, networking and host services. Do not expose `.env.base` or secrets.
The Spark mirror is not a Git repository. Recheck live state before each run.
The historical heartbeat remains paused. Cleanup regression checks do not
restart the open-ended Stage 2 research sequence.

## Verification and publication

Run focused checks while editing, then the required checks in
[CONTRIBUTING](CONTRIBUTING.md). Keep dependency changes paired with `uv.lock`.

```sh
uv sync --locked
uv run python -m unittest discover -s locomotion/tests
uv run python -m unittest discover -s isaaclab/tests
uv run python -m unittest discover -s robot/tests
uv run python -m unittest discover -s experiments/trajectory_optimization/tests
python3 tools/source_inventory.py check
python3 tools/check_pipeline_lineages.py historical
python3 tools/check_pipeline_lineages.py current
```

Every push needs a corresponding paper/Pages update under
[PROJECT_SITE](docs/PROJECT_SITE.md): one append-only change record, generated
STATUS, exact-diff validation, build and renderer check. Source changes need a
new manifest. Published manifests remain unchanged. Verify the exact pushed
revision's CI and deployed Pages revision.

The standing publication authorization permits verified changes on a
`[netid]/[task]` branch, a PR to `main`, review and merge after green CI. Use
forward commits and preserve teammates' work. Do not push to `main` directly.
