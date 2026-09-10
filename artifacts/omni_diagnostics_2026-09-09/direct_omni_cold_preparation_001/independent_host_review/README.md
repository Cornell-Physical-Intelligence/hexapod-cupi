# Independent cold-host execution review

Five independent CPU tests pass against the final host. They execute its actual inherited `run_owned` implementation, with fake process, container, resource and lock boundaries; they do not substitute a copied control loop. No native Isaac application or GPU was started by this review.

The exercised paths are successful standing, successful baseline with the exact checkpoint metadata, a missing AppReady marker beyond 90 seconds, a mismatching completed runtime identity, and a Docker client that has already exited while its exact owned container remains running. Both lock descriptors close and the exact owned container is stopped and checked in every exercised path.

The host's real command builder, cold-source verification override, job-metadata wrapper and parent container inspection execute in these tests. The command mounts the cold source and checkpoint read-only; baseline additionally shadows the completed standing directory with a read-only mount. This verifies the explicit generated CLI mounts, not every possible inherited Docker Compose alias. Root separately performed actual Spark source and host preflight.

Read-only adapter inspection found two metadata insertions around the unchanged legacy entrypoint: the AppReady marker follows successful `AppLauncher(args).app` construction, and the `state.json` writer verifies and records the exact 16-file imported runtime. Existing control functions and actor loading remain intact. The old entrypoint's traceback timer is 90 seconds, while the inherited startup rejection deadline is also 90 seconds; this review does not claim a 45-second traceback. The baseline uses the original 315/318 checkpoint contract, not the newer reference actor.

No concrete first-run blocker was found in these paths. This is launcher preparation evidence; it is neither a physical standing verdict nor an all-direction walking qualification. Native failures and actual cleanup remain the dispatched job's responsibility.

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_omni_cold_host_independent_review_001 -p 'test_*.py'
```

`REVIEW.json` binds the reviewed final host, metadata adapter, parent supervisor and exact test payload. `FREEZE_SHA256.json` covers this immutable review bundle.
