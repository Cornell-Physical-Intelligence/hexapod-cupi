# Contributor guide

Keep chat responses under 100 words. Begin visible responses with `James,`.
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

**Research remains paused for James.** The latest main handoff pauses native
validation, training, recording and unattended continuation. This user's current
authorization covers repository cleanup and its checks/publication. It does not
resume research or release the Spark reservation. Keep the pause until James
explicitly chooses to resume. [JAMES_HANDOFF](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/JAMES_HANDOFF.md) preserves
prepared, unexecuted sources and the actual pause receipts.

The Spark mirror `/home/orionh/HEXAPOD` is not a Git repository. This local cleanup
does not launch training or modify shared compute. For an authorized run read
[OPERATIONS](docs/OPERATIONS.md) and the current
[compute coordination rules](docs/SPARK_COMPUTE_COORDINATION.md), including the
user's existing HEXAPOD reservation and external-automation blocks. Keep the GPU
lock, exact process/container ownership, bounded recovery and post-exit checks.
Preserve reservation controls during cleanup; never act on unidentified jobs.
Never expose secrets, especially the Spark container's `.env.base`.

## Verification and publication

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

Every change follows [docs/PROJECT_SITE.md](docs/PROJECT_SITE.md): update the
registry for changed progress, regenerate STATUS and add an append-only change
record. CI checks coverage, evidence, generated status and release identity.
Covered source changes require a new manifest and matching CI selection;
published manifests stay unchanged. Source snapshots do not admit hardware.

The user's standing publication authorization applies to verified changes:
commit and push the integrated result with its context and verify the deployed
revision. Use forward commits, preserve teammates' main changes and never rewrite
published history. Subleads own ordinary review; James accepts mission changes.

## Architecture partnership and parallel work

The user requests Claude Code with Fable 5.1 (`claude-fable-5-1`) with maximum reasoning (`--effort max`) as a partner for major architecture decisions and substantial parallel implementation/review work. Consult it on controller/action architecture, learning objectives, terrain/perception interfaces and consequential experiment changes. Give workers bounded work areas and concrete deliverables; keep one owner for Spark execution and integration. Preserve the actual model/session, inputs, findings, disagreements and the resulting engineering decision. Independently review proposed code and run relevant checks before adoption. Report unavailable authentication/model access honestly; do not silently substitute a model or claim a consultation that failed. Routine verified repairs and ongoing authorized jobs can continue while reviews run.
