# Quiet-priority native004 preparation and smoke003 dispatch

This bundle preserves the reviewed preparation for a new quiet-priority PPO smoke and one bounded dispatch snapshot. It does not establish a quality improvement or Stage2 completion. The original policy checkpoint, robot geometry, physics, observation widths (315/318), target limits and evaluation gates remain bound to the prior direct-policy lineage.

The new `direct315_quiet_priority_native_v3` objective weights quiet temporal CAPS at 1.0, moving temporal CAPS at 0.1 and spatial CAPS at 0.1. Added optimizer observations record KL, learning-rate changes, pair counts and sparse actor-gradient diagnostics. They do not alter the physical gates. The explicit smoke allocation is 32 replicas × 24 controls × 2 updates; any later pilot requires its own selection and an exact same-source completed quiet-priority smoke receipt.

| Preserved bundle | Payloads | Evidence |
|---|---:|---|
| [native004](native_004/README.md) | 43 | 23 CPU tests, including actual RSL instrumentation equivalence |
| [host003](host_003/README.md) | 20 | 28 focused tests; original ownership, cleanup and phase bounds |
| [guard004](guard_004/README.md) | 31 | 31 focused tests; unchanged embedded restorer and exact ancestry |
| [root readiness](root_readiness/README.md) | 10 | Independent remote 599-file audit and actual no-GPU host preflight |
| [dispatch snapshot](dispatch/README.md) | 6 | Exact launch receipt and one explicitly nonterminal snapshot |

Actual source003 has 599 payloads, manifest `ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62`; plan `eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2`. The original checkpoint remains `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. The root receipt contains the full source inventory and comparison with the exact 589-file cold parent, rather than another complete source copy. Its subsequent build verification is explicitly distinguished from original builder stdout. Existing CPU logs and accurately labeled independent summaries are retained; no unit tests were repeated for packaging.

The separate dispatch record reports launch at 2026-09-10 17:53:36 UTC, invocation `0882531a12724963a64901721d87dc96`, under pause059. At its bounded snapshot, the unit was active, standing was admitted, and training had been attempted with zero completed updates reported. These are snapshot facts, not a current-status or terminal-result claim. The weather replay had naturally completed before dispatch; preparation alone does not grant a GPU slot. Subsequent training, reload, evaluation, cleanup and restoration evidence belongs in a new immutable terminal wrapper.

Every payload and original manifest in the five selected bundles is copied unchanged. [PUBLICATION_SELECTION.json](PUBLICATION_SELECTION.json) identifies the exact selection; no selected payload is excluded. Historical draft and parent receipts retain their labels. No internal Claude JSONL reasoning event streams are included.

Verify this portable bundle without importing runtime or contacting Spark:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B verify_bundle.py
```

The publisher must follow `docs/PROJECT_SITE.md`: append a bounded `site/updates/` record, update the current `STATUS.md` snapshot and relevant Markdown, change `site/project.json` if presentation changes, and run the required site check/build. This staging step changes only ignored tmp files. Public milestone framing remains: first walking benchmark achieved; omni in progress; terrain/perception next; autonomous survey final. It does not assert that all Stage1 physical gates passed.
