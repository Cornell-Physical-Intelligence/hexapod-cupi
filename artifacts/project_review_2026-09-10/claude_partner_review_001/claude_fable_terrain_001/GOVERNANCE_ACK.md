Acknowledged. This session stays read-only; root publishes the review with the site update record.

For any future implementation of the terrain backlog, each change must:

- **Register presentation changes** in `site/project.json` when implementation, geometry, milestones, results or media change, keeping IDs stable, and update `STATUS.md` for execution state or the plan/contract for design changes.
- **Add an append-only record** at `site/updates/YYYYMMDDTHHMMSS_name.json` with `id`, UTC `date`, `title`, `summary`, `areas`, `changes`, `evidence`, `next` and `no_project_impact`, plus a concrete `reason` for no-impact entries. Cover every changed path with bounded globs, never a catch-all.
- **Run the checks** before commit:

```sh
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
uv run python -m unittest discover -s isaaclab/tests
```

Then commit, push, and verify the Pages deployment and remote revision.
