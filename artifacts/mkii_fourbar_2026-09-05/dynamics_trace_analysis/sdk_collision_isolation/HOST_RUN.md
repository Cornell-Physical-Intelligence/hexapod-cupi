# Bounded external collision diagnostic

`host_run_overlap.py` imports the selected frozen source's existing host
supervisor for resource checks, source identity, archiving, container labels and
ownership verification. It has not been run against Docker or a GPU. The local
tests exercise command composition, rejected reports, exact-ID cleanup, GPU
resource vetoes, the monitored case loop and shared-note interruption with mocks.

Stage this artifact directory separately from the frozen production snapshot.
Run the copied host script in that exact staged directory; it hashes itself and
every other fixture file. Do not put outputs inside either input snapshot.

```sh
python3 /ABSOLUTE/FIXTURE/host_run_overlap.py \
  --source-dir /ABSOLUTE/FROZEN_SOURCE \
  --fixture-dir /ABSOLUTE/FIXTURE \
  --source-commit d863663 \
  --output-root /ABSOLUTE/OUTPUTS \
  --case all --physics-steps 256 --solver-multiplier 2 \
  --timeout-seconds 600
```

This prints the planned commands without launching. Add `--execute` only for
the authorized live run. `--case filtered` and `--case unfiltered_negative`
also work separately. `all` runs a fresh Kit container for each case, proceeding
to the negative case only if the first case completes and meets its control
expectation. It then invokes the fixture's CPU-only report/trace comparison.

Each case is limited to 600 seconds, followed by bounded exact-ID stop/log/remove
cleanup. Both phases share the usual hexapod host lock. The shared coordination
note must remain in its unambiguous `NONE` state with unchanged bytes; a change
stops the owned diagnostic at the next resource check, normally within five
seconds. Unrelated GPU activity also stops it. The supervisor requires an
observed GPU PID verified as a descendant of its own immutable container ID.

The production source is mounted read-only at `/workspace/hexapod`, the external
fixture read-only at `/workspace/overlap_fixture`, and each case's output is the
only writable task bind mount. The production source and external fixture are
archived separately and checked before admission and during/after execution.
The case directories preserve argv, native logs, source/fixture identities,
resource evidence, the fixture report and trace. A successful case has supervisor
status `diagnostic_complete`; all pass, training-admission and hardware-admission
flags remain false. These controls establish the measured contact response of
the diagnostic pair, not whole-robot or mission readiness.
