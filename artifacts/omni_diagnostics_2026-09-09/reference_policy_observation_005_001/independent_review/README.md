# Independent CPU review: qualified-flight tensor and observation005

**No concrete correctness blocker was found in the frozen CPU bundles for the next bounded device-bridge smoke.** This is not CUDA, Isaac device integration, PPO, deployment, or new physical admission. No frozen source, main checkout, simulator state, or GPU job was changed.

The reviewed tensor freeze is `5f46b6f99c172bb037004e758f4f3715d41da42725abd10bfd313ee2b84fe286` (86 files). The observation freeze is `22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63` (160 files). Every file and the absence of unlisted files were independently verified. The observation's entire copied reference tree matches the tensor bundle. Its actual009 inputs match the separately audited raw trace and reference-state bytes.

## What was checked

The independent rerun passed all **13 tensor tests** and **18 observation tests**. Logs are preserved in [tensor_tests.log](tensor_tests.log) and [observation_tests.log](observation_tests.log). The source009 full replay covers 2,200 post-settle controls with 11 confirmed steps and final reference quiet hold. The two-sample RR micrometre rebound remains unqualified unloading. The earlier003 insufficient-clearance evidence remains rejected at the new explicit apex deadline; earlier004 still fails the 12 mm landing bound. These CPU outcomes do not replace the original physical verdicts or add a tensor/GPU physical result.

The contact state machine keeps raw off-contact sample/run counts distinct from qualified flight. An unqualified return resets the contiguous run and baseline/peak; separated rebounds cannot accumulate clearance. Qualification still requires two consecutive off-contact samples and 2 mm actual lift before the finite apex deadline. After qualification, return contact retains apex/descent, speed, original endpoint, excursion, finite reacquisition, three-contact confirmation, support, drift, and executable P/V/A checks. Stop suppresses new liftoffs while completing the current swing and reaches finite reference quiet. Selected fresh resets preserve other rows; invalid rows latch failure and cannot emit finite executable targets. The explicit 198-step float32 position-rounding recurrence remains part of source compatibility.

The live observation schema independently reconstructs **846 actor values and 849 critic values**, with contiguous field ranges and hash `26dfa8604379a396d477683b530c3c0b0aa576aa646c90ccca94eb660a7a8d7e` for the fixture's named order. It includes the ninth unloading mode, all raw-run/unqualified-return counters, optional return time/lift/presence, original vertical and horizontal polynomials, provisional landing state, anchor/preload state, command filter, desired pose, stop/quiet state, and residual controller state. It is explicitly incompatible with prior checkpoints.

History has five 63-value frames plus five validity flags. A new episode begins with only its current frame valid; an interval-derived rate is invalid until two consecutive accepted samples exist. Identical duplicate reads do not append history. Changed duplicate, skipped/stale step, invalid row, and old episode inputs fail closed; a selected fresh reset clears old history without changing unaffected rows. Joint order is named and checked, including reversed-order tests. Body/world frame handling and raw XYZW conversion are explicit.

Raw SDK joint rates remain in history and returned diagnostics. The additional rate channel is a **20 ms interval-average of consecutive joint positions**, with explicit interval and validity, not an instantaneous or physically validated replacement. The biased-rate regression preserves the SDK bias while the stationary angle-derived rate stays near zero. No quiet metric, torque gate, or simulator velocity property is changed. Contact points retain a raw NaN/validity pair and use a masked encoded zero sentinel; a missing point cannot become a fabricated foot anchor. Sensor ages and pre-reset flags must be supplied by the integration layer.

The three dynamic implementation files have no `.cpu()`, `.numpy()`, `.item()`, or `.tolist()` calls. Static geometry/source construction and fixed loops remain visible. This source inspection and CPU execution establish no GPU synchronization cost or throughput. [verification.json](verification.json) contains exact source hashes, schema fields, initial-history evidence, and manifest checks; [verify_review.py](verify_review.py) reproduces the bounded inspection.

## Device-bridge boundary

The PPO owner is separately preparing an installed-sensor timestamp and pre-reset bridge. Its planned contract uses real lazy sensor updates and device timestamp receipts, checks all required sensors, and calls the original done predicate once before reset. The short proposed hold smoke is not a new full standing or walking admission. The reviewed frozen encoder correctly requires actual freshness and measurement validity rather than inventing them.

A brief read of the in-progress bridge ordering found the settled history seed at 4.0 s, then consecutive 20 ms sample indices, consistent with the frozen encoder. Invalid measured/freshness/terminal rows are rejected before their use by the next reference step. That coordination note is not a freeze or launch review of the still-changing bridge. The existing source009 observer and raw SDK/angle-derived channels remain separate. Installed CUDA APIs, real timestamp cadence, device packing parity, and actual execution cost still require the owner/root's bounded physical smoke.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_tensor_wave_005_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_policy_observation_005_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_wave005_observation_independent_review_001/verify_review.py
```

The test logs retain a scalar-fixture warning about constructing a tensor from a list of NumPy arrays. It occurs in the CPU oracle, not a newly added dynamic device transfer or a failed check.
