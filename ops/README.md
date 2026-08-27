# ops/ — operator entry points

```text
ops/hexctl      The single operator CLI: compose, doctor, probe, screen
ops/attic/      Retired deploy scripts, kept verbatim for provenance
```

## hexctl

`ops/hexctl` is a stub. It resolves the repository root from its own path, puts
`packages/hexapod_train` and `packages/hexapod_env` on `sys.path`, and calls
`hexapod_train.cli.main()`. There is nothing to install and no dependency beyond
the standard library.

```sh
# Print the exact launcher argv for a named experiment plus an intervention.
# Works anywhere, including a laptop; no side effects.
ops/hexctl compose \
  --experiment configs/experiment/stage2c_accel.yaml \
  --intervention configs/intervention/probe21_bilateral.yaml \
  --seed 99 --label accel_probe21_bilateral_ref1800_seed99_20260827T090000Z

# Read-only resource gates: GPU, producer scripts and their descendants,
# containers, the training service, and the shared GPU lock.
ops/hexctl doctor

# Gate, compose, launch the probe launcher as a supervised child, record the
# attempt outcome and the bounded retry decision.
ops/hexctl probe --experiment ... --intervention ... --seed 99 --label ...

# Supervise the formal sharded screen.
ops/hexctl screen --batch-label ... --run-name ... --parent-run ... \
  --parent-checkpoint model_2.pt --parent-sha256 ... --output-label ... \
  --checkpoint 0 --checkpoint 1 ...
```

`compose` runs anywhere. `doctor`, `probe`, and `screen` fail closed on any host
that is not the Spark: they print what is missing and exit non-zero rather than
attempting a launch or raising a traceback.

## Wrap, do not replace

The bash launchers under `isaaclab/deploy/` are battle-tested and stay the
executors. They hold `/tmp/hexapod-isaac-gpu.lock` for the whole logical run,
gate on Docker and the systemd service, verify the immutable parent by SHA-256,
publish their artifact leaf atomically with a non-reusing `mkdir`, refuse
unreviewed Hydra overrides, and clean up by exact immutable container ID — a
cleanup path whose parser bug has already been found, fixed, and validated in
vivo. Rewriting that in Python would throw away every one of those proofs and
re-open failure modes the repository has already paid for.

So `hexctl` composes the launcher's argv and supervises the launcher as a child
process. It adds only what the bash cannot see from inside one
`timeout`-wrapped `docker compose run`:

- milestone timing — `Loading user config` by 45 s, the AppLauncher completion
  marker by 90 s, a 420 s overall cap;
- an atomic diagnostics capture after 30 s without progress;
- a contention recheck after container creation and again before PPO starts,
  which is the race a one-shot prelaunch gate structurally cannot catch;
- the bounded retry decision — at most one identical retry, under a new
  immutable label, only for a proven pre-AppReady stall.

What it never does: bypass a launcher gate, delete a cache, a Docker volume, or
a container beyond the launcher's own exact-ID cleanup, signal a process it did
not create, reuse a label, or retry more than once.

## In-vivo validation is still pending

None of `doctor`, `probe`, or `screen` has been run against the Spark. The
package is unit-tested against fakes — `isaaclab/tests/test_hexctl_compose_contract.py`,
`test_run_contract_supervisor.py`, and `test_labels_and_gates.py` — and every
module says so in its docstring: *implemented to spec; not yet validated in vivo
on the Spark*. Treat the first real run as a supervised experiment, keep its
`attempts/` evidence, and record the result in `STATUS.md`.

One spec item is knowingly partial. `docs/OPERATIONS.md` §6 item 10 requires
shard B to launch only after shard A reaches AppReady, but
`screen-stage2c-probe-sharded` starts both shards itself, and enforcing the
order means editing that launcher. `hexctl screen` therefore *verifies* the
ordering from the shard logs and refuses to certify a screen that violated it,
rather than imposing the order. Hardening the shard launcher itself is separate
work.
