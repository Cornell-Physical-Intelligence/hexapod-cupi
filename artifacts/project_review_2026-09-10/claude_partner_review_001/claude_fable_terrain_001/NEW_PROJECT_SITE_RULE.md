# Live research poster: required contributor contract

This is a hard user requirement for every repository-changing agent and human contributor, including Claude Code workers. Pass it into delegated tasks. The poster should explain the project visually in clear English: the mission, full system, robot geometry, actual findings, media, unresolved issues and next experiments. PPO Stage 2 remains the immediate execution priority; site work runs alongside training.

## Central framework

- `site/project.json` is the presentation registry: system nodes/edges, milestone meanings, geometry, evidence and curated media. Keep its IDs stable. It points to authoritative repository documents and immutable artifacts.
- `STATUS.md` is the execution snapshot. The build extracts its opening statement and selected execution text; never claim that a static push snapshot is a live Spark probe.
- `site/updates/YYYYMMDDTHHMMSS_name.json` is an append-only change record. Every repository change requires a new record. It includes `id`, ISO-8601 UTC `date`, `title`, `summary`, `areas`, `changes` (repository-relative paths or bounded glob patterns), `evidence`, `next`, and boolean `no_project_impact`. A no-impact entry also needs a concrete `reason`.
- `tools/project_site.py` validates changes, resolves real evidence and selected media, and builds the public artifact in `site/dist`. Generated output is not committed. The build includes the exact Git revision and timestamp.
- `.github/workflows/pages.yml` validates every push/PR and deploys main with GitHub Pages. Deployment reads only explicitly selected public files, not the entire checkout.

## Workflow for every change

1. Read this contract and the affected project/evidence documents. Tell delegated agents the rule before they edit.
2. Make the code/data/documentation change. Preserve frozen failures and checkpoint bytes.
3. Update `STATUS.md` for a changed execution state and the relevant plan/contract for design changes. Update `site/project.json` when its presented implementation, geometry, milestones, results or media changed.
4. Add a new `site/updates/` JSON record. Cover every changed path except the new update record itself. Use bounded directory globs for an immutable evidence bundle; do not use a catch-all `**`.
5. Run `python3 tools/project_site.py check --base <integration-base-SHA>` and `python3 tools/project_site.py build`, plus the tests relevant to the code change. The check requires new records relative to the base; editing an old entry cannot satisfy it.
6. Commit and push with the relevant Markdown. Verify the GitHub Pages deployment and remote revision. The poster is a public research snapshot from that push, not an unverified progress claim.

Example entry:

```json
{
  "id": "20260910T160000_ppo_smoke",
  "date": "2026-09-10T16:00:00Z",
  "title": "Checkpoint reload now works",
  "summary": "The learner completed its small integration run. Quiet walking still fails.",
  "areas": ["locomotion"],
  "changes": ["STATUS.md", "artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_002/**"],
  "evidence": ["artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_002/README.md"],
  "next": "Compare the two independently initialized training pilots.",
  "no_project_impact": false
}
```

A no-impact entry explains why the change affects neither presented behavior nor evidence, such as formatting an internal test helper. It still names the affected files and workstream. The validator is a required check, not a mechanism for waiving research gates.

## Evidence and media rules

Keep latest saved checkpoint, latest actual recording and accepted benchmark separate. Every selected local media item has its path, SHA-256, type, caption and evidence reference. A video must identify the controller/checkpoint actually rendered, playback speed if known, and whether it is qualified. Never animate an invented robot rollout or use a schematic as experimental evidence. The flowchart may animate information flow; it describes architecture, not live sensor packets.

Each milestone states what exists, what remains, and what passes it. Preserve C-study versus physical four-bar lineage and software integration versus real-robot qualification. Do not invent completion percentages. Plain English comes first; technical detail, code, papers and hashes remain one click away. Respect reduced motion and usable mobile/keyboard navigation.

## Hosting

The public target is GitHub Pages for `Cornell-Physical-Intelligence/hexapod-cupi`. The user explicitly requested this destination; no separate Sites-hosted deployment is created. Use GitHub's official `configure-pages`, `upload-pages-artifact` and `deploy-pages` actions. The repository is already public; this workflow does not change its visibility or publish arbitrary workspace files. All linked evidence was already authorized for repository publication.
