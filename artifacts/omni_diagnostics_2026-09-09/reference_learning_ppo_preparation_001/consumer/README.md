# Recoverable moving PPO consumer003

Read [PROTOCOL.md](PROTOCOL.md) for the complete admission, evidence and host
contract. This source has not run on Spark. It is a separately versioned learning
infrastructure candidate; predecessor profiles and every physical evaluation
verdict remain unchanged.

The admission group is fresh standing → quiet calibration →
`learning_recovery_32` (200 startup + 512 zero-residual controls). It must prove
an actual finite event and completed 200-control recovery with correct observed
history, reset clocks and critic evidence, then stop without PPO updates.
An external root review is required before the separate ten-update group and
its unchanged cold forward/stop and quiet screens.

Thirty CPU tests pass in `tests_all003.log`, including actual RSL collection and
reload, exact final same-episode timeout values, an actual-class 712-control
recovery-ledger run, reset inference-mode/int32 semantics, selected history
isolation, no-automatic-allocation/receipt failures and standard-library-only
host imports. These are synthetic CPU fixtures; actual GPU reset/recovery and
physical admission remain untested for this source. The first aggregate run's
two test-fixture errors are preserved in `tests_all003_initial.log`: canonical
macOS paths and float32 GEMM batch-shape rounding. Runtime bounds were not relaxed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps \
  .venv/bin/python -m unittest discover -s tmp/reference_moving_ppo_003 -p 'test_*.py'
```

The frozen parent002 description/receipt is preserved under `PARENT002_*` as
history, not this protocol's admission authority. Earlier `tests_*.log` files
are likewise retained as provenance; `tests_all003.log` is the final aggregate.
Source009, observation005 and device001 are bound inputs, not copied or altered.
No main, production-model, frozen-predecessor or GPU changes were made here.
