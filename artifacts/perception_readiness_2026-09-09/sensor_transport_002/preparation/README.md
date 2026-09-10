# Phase3 sensor transport v2: bounded CPU successor

This isolated successor fixes three independently reproduced transport failures. It has **15 passing CPU tests**, including the original failing behaviors and exact nominal numerical parity. It is not connected to an actor, terrain task, real sensor, or new GPU job. The original source remains at `original/sensor_model.py`, SHA `101f6a59488f702117e93da87354b0d68c3d475e50421f445a12d9caf1491336`.

## Corrections and explicit behavior

1. **Newest valid capture per environment.** The queue still releases by modeled delivery time. Each environment now accepts only a lexicographically newer `(capture_time_s, enqueue_sequence)` than its held valid sample. A later-enqueued packet wins an equal-time tie only in rows where it is valid. An invalid newer packet neither replaces a held value nor renews its age; a delayed older valid packet can still fill a row that has no newer valid value. Ordering uses float64 timestamps and integer sequence numbers independently of the unchanged public float32 timestamp/age tensors. Default cadence/jitter combinations did not exhibit this reorder; the reproduced bug occurs under a legal custom latency ablation.

2. **Writable persistent state.** Transport enqueue/read/reset and sensor capture/reset run their data operations with inference mode disabled and gradients disabled. Queued values/masks, held values/order metadata and IMU bias state therefore remain ordinary tensors even when called from an inference rollout. Selective reset outside inference mode now succeeds both during latency warmup and with delivered/pending frames together. Unselected held values, pending validity, selected timestamps and IMU bias state are preserved at reset. The original shared RNG semantics are retained; this is not a promise of independent future random draws when different subsets reset.

3. **Clock rejection before mutation.** Transport timing configuration, capture/read clocks and IMU integration dt must be finite. Simulation clocks are nonnegative; capture and read timestamps are independently nondecreasing per stream. Equal timestamps remain legal. Backlogged capture time may precede the latest read, provided that stream's capture sequence does not rewind. Invalid public capture time is rejected before noise draws or IMU bias updates. Model read validates all three stream clocks before any stream drains. A readiness tolerance never exposes a sample captured after `now`. No negative age is valid.

`reset(None)` clears all queues/values and starts a new clock epoch. `reset(env_ids)` invalidates only those environments' queued/held data and IMU initialization while retaining stream clocks, even when `env_ids` happens to contain every environment. The caller must use the full reset API to restart simulation time; resetting one robot must not rewind time for its peers. Already queued pre-reset rows are invalidated. There is no new external packet-epoch API for accepting late packets injected by another transport.

Payload fields, shapes, dtypes and devices are fixed within an epoch and checked before enqueue mutation. The original latest-frame public API and defaults remain unchanged. Returned values are read-only references owned by the sensor model, as before. Age/staleness remains the original float32 representation; this patch does not claim long-duration timestamp precision has been improved. Sensor clocks use simulation seconds, not epoch/Unix timestamps.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/phase3_sensor_transport_002 -p 'test_*.py' -v
```

`test_output.txt` records the passing run. The tests reproduce old overwrite, inference-reset exception and negative-age behaviors before exercising the fixes. Further coverage includes per-row validity, equal capture-time arrival reversal, same-read queue draining, distinct timestamps that round to the same float32 value, pending/delivered mixed-row reset, IMU bias reset, invalid clock/RNG-state isolation, staleness after dropout, full versus partial clock reset, cross-stream atomic clock validation, and nonfinite transport/dt inputs.

The seeded nominal parity test runs the original and successor camera/LiDAR/IMU models for 401 reads over two seconds at their configured cadences, including a selective reset and a final stale read. It compares every emitted value, mask, public timestamp and age **bitwise**. No configured noise, latency, dropout, material, support threshold, 250 ms map lease, or physical gate changes.

## Integration proposal

`INTEGRATION.json` gives exact file mapping and checks. `integration/sensor_model.patch` is the runtime-only patch against the preserved original. Proposed tracked destinations are:

- `isaaclab/hexapod_phase3/sensor_model.py` — the reviewed successor runtime.
- `isaaclab/tests/test_phase3_sensor_transport.py` — the portable tests, with repository paths already adapted in the supplied integration copy.
- `isaaclab/tests/fixtures/phase3_sensor_transport_v1.py` — the exact historical source fixture for failing-before evidence.
- `isaaclab/hexapod_phase3/README.md` — document v2 ordering, clock and reset semantics.

Root owns any adoption. Follow the mandatory `docs/PROJECT_SITE.md` contract: publish immutable review evidence, update corresponding progress/evidence/roadmap entries or an explicit checked no-impact declaration, add a new bounded `site/updates/` record, and run site validation/build plus the integration tests. This work does not select an actor or claim terrain/perception completion. No optional released-frame/drain API was added; causal mapping still needs a separately specified bridge if every intermediate packet is required.
