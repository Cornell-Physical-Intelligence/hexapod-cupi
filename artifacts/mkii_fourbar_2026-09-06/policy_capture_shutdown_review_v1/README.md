# Policy capture shutdown review

The immutable `policy_capture_tools_v2` already persists its successful outputs
before Isaac's native shutdown. No v3 lifecycle change is needed for the reviewed
hard process-exit risk. The six source/evidence files still match the original
SHA256SUMS, whose digest is
`d76e6601c9327f6c04bde05fb6012d399c8db1463b1ffe68c5f57b17c42959d3`.
No GPU job, native simulator or live encoder was started for this audit.

The installed `launch_simulation` context calls its `close_fn()` in `finally`;
the observed Kit path can exit the process there before control returns to code
after the context. The capture's successful path does not require that return:

| Operation in record_admitted_policy.py | Line | Shutdown relationship |
| --- | --- | --- |
| Encoder close; then check encoder process return code | 249 | Inside simulator context |
| Validate states and write compressed NPZ | 256 | Inside simulator context |
| Write metadata and hash all three artifacts | 277 | Inside simulator context |
| Revalidate input/tool identities; finish success report | 285 | Inside simulator context |
| Restore guard and close any unfinished writer | 293 | Inner finally |
| Close environment | 299 | After persisted success report |
| Exit launch_simulation context | 299 | All required recorder writes already complete |

The host `capture_policy.py` remains outside the Kit process. It verifies the
exact container identity and final exit status, removes that owned container,
then validates the primary report and artifact hashes (line 236), runs actual
ffprobe frame decoding/counting (239), checks dimensions/rate/duration/codec
(242), and persists supervisor success. A zero container exit with a failed,
missing or corrupted primary report is rejected; the existing focused tests
exercise that condition.

`review_001.json` records verified source hashes, AST-derived containment/order
locations, and the focused suite result: 21 tests passed. This proves the
reviewed source ordering and existing mocked supervisor behavior, not that live
RTX rendering/encoding has already succeeded. The first guarded capture remains
the live check. No frozen artifact or runtime source was edited.
