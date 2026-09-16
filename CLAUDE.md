# Contributor guide

Keep chat responses to 200 words or fewer.
Use direct language. Preserve other contributors' work.

Edit this file for shared agent instructions. `AGENTS.md` is a symlink to
`CLAUDE.md`, so both names read and edit the same source.

## Context and authority

Read the assigned issue, its input fixtures and the relevant section of
[ARCHITECTURE.md](ARCHITECTURE.md). It owns agreed requirements and boundaries.
`site/project.json` owns progress; `STATUS.md` is its generated text view.
GitHub issues own assignments. Define the next smallest increment with James
and the responsible lead before treating a roadmap marker as ready work.
Do not create another plan, progress document or handoff-based task queue.

During the design review, discuss one decision or milestone at a time. Read the
team lead's existing design and evidence before asking James to choose. Use that
design as the default and propose its smallest viable version, explicitly
separating existing decisions, unimplemented proposals and new simplifications.
Ask only about consequential gaps or tradeoffs. James's later decisions take
precedence; historical instructions never resume research or override the
approved robot model.

Read the affected package's `CLAUDE.md` for local ownership. Those package guides
are part of the frozen historical release: their 66-value layouts, mock timing
and gate claims apply to their original models. They do not define the canonical
robot's pending runtime. Preserve those bytes; this root guide and ARCHITECTURE
make that scope explicit. Use explicit paths
for source searches; default search excludes frozen evidence, historical notes,
raw CAD and generated files. Open evidence intentionally when assessing a result.

## Maintained boundaries

- `packages/`: shared contracts, environment, training, evaluation, runtime and navigation.
- `tools/`: repository commands; `tools/assets/` owns asset utilities.
  `configs/source_inventory.json` classifies every tool and records its owner.
- `ops/` and `isaaclab/deploy/`: existing guarded operations and release manifests.
- `experiments/c_length_study/tools/`: historical study runners and analysis.
- `experiments/terrain/tools/`: terrain/perception prototypes; qualification is separate.
- `experiments/paper_walk/`: maintained canonical PPO/motion-prior prototype;
  its top-level modules are inventoried, while `tests/` contains CPU checks.
- `experiments/trajectory_optimization/`: model-bound forward-cycle optimizer
  and native replay entry; its tests check dynamics and recorded inputs.
- `robot/`: approved models, importers and model-specific conventions.
- `site/`: the progress registry and static GitHub Pages presentation.
- `artifacts/`: immutable results and exact replay inputs; never import these copies into new production code.
- `docs/`: [maintained reference index](docs/README.md); `docs/archive/` holds superseded prose.
  Historical prose is not current progress.

## Invariants

The approved model and motor corrections in `robot/active_model.json` are ground
truth. New training uses the detailed 19-body/18-joint direct-drive robot after
matching admission. Historical mock/C-study/four-bar tasks, models and policies
keep their own semantics. Never repoint a frozen task ID or assume an old
observation layout defines the canonical runtime.

Checkpoint bytes, published SHA-256 manifests, failed attempts, incident records
and the probe ledger are immutable. Preserve their original labels and source
commits. New results and releases get new identities. Never change gates or tests
to admit a result. Formal comparisons use their common recorded limiter and
matching configuration. A code merge or video alone cannot qualify behavior.

Core remains stdlib-only; runtime depends on core and stdlib; navigation depends
on core and must not import simulation or runtime. Model-specific package rules
retain the exact compatibility constraints and tests.

## Operations

**Current dispatcher paused for handoff on 15 September 2026 UTC.** James said:
“take a pause for now and let someone else continue work, push all non commited
changes”. This supersedes the current dispatcher's instruction to continue
research. Publish the preserved work, then relinquish execution. Another
designated lead may continue the existing authorized goal without another user
permission; no successor is named here. Do not automatically resume this
dispatcher or the historical heartbeat, which is already PAUSED. Do not create
a separate handoff task queue.

The [pause receipt](artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
records no running native jobs, containers or GPU compute apps at 01:58:15 UTC.
The exclusive reservation and queue lock remain retained. The shared Spark
coordination file now has SHA-256
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
A successor must recheck live ownership/resources and create a fresh guard and
launch binding to those coordination bytes; preserve every old source and
binding unchanged. Read the completed startup comparison and unexecuted refit
scope in [TRAINING](docs/TRAINING.md) before deciding the next increment.

**James authorized the canonical qualification restart on 14 September 2026.**
He confirmed the displayed mass-corrected model and the sequence standing →
walking/stopping → terrain → survey. Use the URDF selected by
`robot/active_model.json`, SHA-256
`9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78`.
That authorization superseded the earlier research pause for that sequence; it does not qualify a
stage, change admission gates or release the Spark reservation. Keep historical
results and the [JAMES_HANDOFF](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/JAMES_HANDOFF.md)
pause receipts intact. Fresh attempts need new identities bound to this model.

**The active user goal is Stage 2 on this confirmed model.** James explicitly
directed continuing until the existing Stage 2 requirements are satisfied,
including actual paper-informed PPO training and a video of the trained policy
running in Isaac Sim. The successor's remaining scope includes matching native admission, learning and
the full existing direction, transition, quiet-stop, torque, contact and visual
acceptance requirements. Diagnostics, prepared code, checkpoint saves and video
recording alone do not finish the goal. Preserve the 0.040 rad / 20 ms comparison
limiter and every existing numerical gate; never transfer historical acceptance
or substitute prior animation for an actual policy rollout.

**James also granted Codex full Spark compute ownership on 14 September 2026:**
"you can stop all other processes on the spark" and "take full ownership of the
spark, you have my permissions". The lead dispatcher may stop or suspend
competing user compute and its restart triggers without another approval.
Identify each workload and preserve its source, outputs and recovery state;
keep SSH, networking, operating-system services and host health available.
Only the lead dispatcher allocates shared compute or changes its controls.

The Spark mirror `/home/orionh/HEXAPOD` is not a Git repository. Authorization is
not proof that a job or continuation automation is running. Before dispatch read
[OPERATIONS](docs/OPERATIONS.md) and the current
[compute coordination rules](docs/SPARK_COMPUTE_COORDINATION.md), including the
user's existing HEXAPOD reservation and external-automation blocks. Keep the GPU
lock, exact process/container ownership, bounded recovery and post-exit checks.
Preserve reservation controls between allocations; never act on unidentified jobs.
Never expose secrets, especially the Spark container's `.env.base`.

## Verification and publication

James requires locomotion load tracking before the next optimizer/PPO sequence
(16 September 2026). Native evaluations must include `force_metrics.json` with
contact-normal loads and motor torque. Check that the summary is available
before the next experiment; preserve failed or incomplete captures. These are
descriptive measurements, not new numerical acceptance limits. See TRAINING.

```sh
uv sync --locked
uv run python -m unittest discover -s isaaclab/tests
uv run python -m unittest discover -s robot/tests
python3 tools/source_inventory.py check
python3 tools/project_site.py status
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
python3 tools/check_pipeline_lineages.py historical
python3 tools/check_pipeline_lineages.py current
```

Run focused checks while editing and the required suites before integration.
Keep dependency changes paired with the lockfile. No new test may merely mirror
an implementation or replace a failing gate.

**Hard rule: every push includes an update to the public research paper/Pages.**
James reiterated this on 14 September 2026. Include the corresponding central
paper update, validate the exact push diff, build the paper and exercise its
renderer before pushing. Verify the exact pushed revision's CI and Pages
deployment; a failed build or an older served revision is not completed publication.

Every change follows [docs/PROJECT_SITE.md](docs/PROJECT_SITE.md): update the
registry for changed progress, regenerate STATUS and add an append-only change
record. CI checks coverage, evidence, generated status and release identity.
Covered source changes require a new manifest and matching CI selection;
published manifests stay unchanged. Source snapshots do not admit hardware.

The user's standing publication authorization applies to verified changes:
commit on a `[netid]/[task]` branch, open a pull request to `main`, merge after
green CI and review, then verify the deployed revision. Direct pushes to `main`
are not allowed; see [CONTRIBUTING.md](CONTRIBUTING.md). Use forward commits,
preserve teammates' main changes and never rewrite published history. Subleads
own ordinary review; James accepts mission changes.

## Review and parallel work

James removed the Fable review requirement on 16 September 2026. Do not start
Fable consultations under the former instruction. Preserve prior consultation
records as historical evidence.

Give workers bounded work areas and concrete deliverables. Keep one owner for
Spark execution and integration. Review proposed code and run relevant checks
before adoption. Record the evidence and engineering decision.
