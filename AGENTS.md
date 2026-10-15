# Agent instructions

Keep replies under 200 words unless the user asks for a review or an
explanation. Give no time estimates. Preserve other contributors' work.

## Outcome

The roadmap stage in progress is "Walk and stop" (Stage 2): the approved
detailed robot passes the full 96-case manifest with unchanged gates, each
native evaluation records ground-contact force and motor torque, and the
program lead accepts the gait's appearance. [TRAINING](docs/TRAINING.md#acceptance)
defines acceptance. `site/project.json` holds the roadmap state and the
current next step; read `latest.next` before you plan. Serve this outcome, not
only the task in front of you. When a task is one step of a repeating job
(evaluate a checkpoint, record a result, update the site), say so once and
propose the durable version: the inputs it reads, the check that judges it and
the record it writes.

## Judges

Your reading of a log is not a result. These decide:

- The CPU tests under each package and `tests/`, and the required checks in
  [CONTRIBUTING](CONTRIBUTING.md).
- The scorer and the native parity and range tests under `locomotion/`, with
  unchanged gates. The 13 probes are diagnostics; the 96-case suite qualifies.
- Green CI on the exact revision and the deployed Pages site.
- The program lead's admission and visual acceptance.

Report a result as pass or fail against these, bound to its model, source,
input and checkpoint hashes. Do not change tests, thresholds or gates to admit
a result.

## Methods

Start from raw material: `run/standing/state.json`, run logs,
`native_results.json`, captures and W&B runs. Reproduce a claim from the raw
data before you restate it from STATUS.md or the site. Report trial counts and
spread; one passing seed is one trial, and a comparison without its control
(the reward-term ablations are that control) shows no cause. Compare loads at
matched achieved motion with the recorded 0.040 rad / 20 ms limiter. Use the
scorer and the evaluator for numbers; do not hand-build a comparison table.

Apply the methods of the fields this project spans, as methods and not as
personas: controls (tracking, limiters, settling), reinforcement learning
(reward terms, seeds, ablations), actuators (RS05 torque and speed limits),
statistics (trial counts, spread, held-out checks), navigation and safety
(east-north frame, uncertainty margins, e-stop and command validity), and
publication (append-only updates, bound evidence). When a task touches one of
these fields, run that field's standard check and name it in the result.

## Context and ownership

Read [README](README.md), [CONTRIBUTING](CONTRIBUTING.md), then the affected
package guide before creating a branch or changing files. Follow CONTRIBUTING
for branch names, checks and publication, and [ARCHITECTURE](ARCHITECTURE.md)
for requirements and boundaries; navigation and mission must not import
simulation.

Several contributors run agents in this repository. The program lead is GitHub
user `palerdr`; the guides assign approvals to that role. Do not assume the
user is the program lead, and do not address the user by a name from
repository files.

Search with explicit paths and exclude `.venv/`, `graphify-out/` and `tmp/`.
Read old records to answer a named historical question. Keep procedures in
existing guides, assignments in GitHub issues and design review in PRs. Do not
add another plan, handoff, decision log or task queue. Edit `site/project.json`
for progress; `tools/project_site.py status` writes an untracked STATUS.md.
Preserve execution history in Git and large new results outside the source
checkout. CONTRIBUTING permits admitted training datasets in the checkout.

Read-only exploration and review subagents need no authorization; give a
reviewer the diff and the question, not your conclusion. A subagent that edits
files or dispatches compute needs the user's authorization.

## Model and evidence

Preserve the approved model and motor corrections in `robot/active_model.json`.
Use the detailed 19-body, 18-joint direct-drive robot after matching native
admission. Preserve checkpoint bytes, published manifests, raw captures, failed
attempts, incident/recovery records and the probe ledger. Use fresh result
names and exact model/source/input hashes. Retain historical labels and source
revisions. Stage 2 requires direction, transition, stop, torque, contact and
visual acceptance. Native evaluations must record ground-contact force and
motor torque before another optimizer/PPO sequence; these measurements add no
acceptance limits.

## Compute and publication

Read [OPERATIONS](docs/OPERATIONS.md) and its
[compute policy](docs/OPERATIONS.md#retained-reservation-and-recovery) before
dispatch. One lead owns Spark; preserve competing workloads' recovery state
before stopping them. Keep both GPU locks and exact-container cleanup. The
launcher shares the GPU with other workloads and records them; it checks no
host reservation. Keep the retained host reservation state until the program
lead releases it. Preserve SSH, networking and host services. Do not expose
`.env.base` or secrets. Recheck live state before each run. The mirror has no
Git metadata. Keep the historical heartbeat paused; cleanup checks do not
resume the research sequence.

Run focused checks while editing and the required checks in CONTRIBUTING before
publication. Pair dependency changes with `uv.lock`. Follow
[PROJECT_SITE](docs/PROJECT_SITE.md) for each push and verify exact-revision CI
and deployed Pages. CI generates the source manifest when a change merges; do
not commit a manifest in a PR. When a change adds an important class or a
directory, also update the code map on the site.

The standing publication authorization permits verified changes on a
`[netid]/[task]` branch, a PR to `main`, review and merge after green CI. Use
forward commits and preserve teammates' work. Do not push to `main` directly.
