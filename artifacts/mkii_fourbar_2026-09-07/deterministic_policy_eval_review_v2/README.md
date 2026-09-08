# Independent deterministic-evaluator v2 delta review

No material issue found in the two v1→v2 fixes. Both immutable bundle manifests verified without mismatch; `evaluate_policy.py` is byte-identical. Nineteen CPU tests passed independently; exact command, reviewed file hashes and raw output are recorded in `review.json` and `tests.log`.

The new `verify_source_before_import` checks the fixed deployment-manifest hash and every listed source member using only the standard library, before the frozen capture helper may execute `source_api`. Resolved members must remain inside the source directory. The subsequent original functional identity and checkpoint/admission/runtime checks still run. Tests reject altered source code before its sentinel can execute, reject a rewritten manifest and reject an escaping member. The manifest hash itself is pinned, so replacing source and regenerating the manifest cannot silently authorize new code.

Log retrieval now has its own error handler inside exact-container cleanup. Log-file open failures, command timeouts, nonzero exits and OS errors mark the evaluation failed, then continue to re-inspect the same container ID/ownership, require it stopped, remove that exact ID and finally release the project and read-only shared locks. Missing logs cannot become an accepted result. CPU lifecycle tests exercise these failures and verify removal precedes both lock releases. The unchanged unsafe/foreign/running-container checks remain conservative.

This review changed neither frozen bundle nor simulator/runtime source and launched no GPU work. It does not establish native CUDA readiness or policy skill. The v2 descriptor still reports execution/integrity only; walking, navigation, terrain and hardware acceptance are not inferred.
