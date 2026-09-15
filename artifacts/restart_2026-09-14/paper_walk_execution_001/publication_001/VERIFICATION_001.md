# Publication verification for 75c9a137

GitHub `main` points to `75c9a13770e816db203221e739778448477b3c24`.
The push succeeded; both exact-commit workflows completed with failure.

- [Research poster, run 34916202273](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/actions/runs/34916202273)
  fails its contributor/evidence test. The build, upload and deployment are
  skipped. Its 25-test run has one error.
- [tests, run 34916202284](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/actions/runs/34916202284)
  runs 981 IsaacLab tests with one error. Later robot/prototype/lineage steps
  are skipped and cannot be claimed to have passed remotely.

Both errors are the same filename-case mismatch:
`visual_review_004/review.json` is referenced by the registry, while immutable
Git evidence is named `visual_review_004/REVIEW.json`. The local macOS filesystem
resolves both spellings, masking the problem in previous local checks. The
exact Git-tree audit checks 332 registry and maintained-document references and
finds this one mismatch in three consumers: `site/project.json`, generated
`STATUS.md`, and `docs/TRAINING.md`.

The public `version.json`, fetched with a cache-busting query and no-cache
request, still reports `e9d5e775b3de76e9836c0139b557e2fcc4af047f`, built
2026-09-11T04:38:57.863936+00:00. The latest Pages deployment6386563297 names the
same old commit and has a successful September11 deployment status. The Pages
configuration's generic `built` state therefore does not establish deployment
of75c9a137.

The repair updates only maintained reference consumers, preserving the original
evidence bytes. The new progress snapshot separately records the verified014
quiet/recovery gates, failed forward tracking, completed train008 and actual015
dispatch; it does not assert Stage2 acceptance. Local contributor tests pass
25/25, the site builds, and its renderer test passes. An artifact-only strict
case check validates98 unique registry references and rejects a deliberately
reintroduced lowercase reference. This adds no production compatibility path.

The first whole-worktree coverage check fails because concurrently created BC
review/migration and evaluation-source changes lack a new covering update at
that instant. Execution-evidence coverage is added by the integration record;
the other owners must cover their changes before root publishes the follow-up.
This receipt is not a successful publication verification for that future
commit. Recheck both exact-commit CI and the served Pages revision after root's
forward fix commit.

`exact_path_audit_002.json` supersedes only the empty missing-path field of001:
the first helper defined but did not invoke traversal. Exact-case proof attempts
001/002 preserve their local helper tuple-unpacking errors;003 uses the actual
three-value registry API and passes both positive and negative fixtures. Those
artifact-helper failures are not additional production defects. All original
request/push receipts are preserved. Timestamped API/command outputs accompany
this report; `VERIFICATION_SHA256_001.json` seals this observation set without
preventing append-only follow-up verification.
