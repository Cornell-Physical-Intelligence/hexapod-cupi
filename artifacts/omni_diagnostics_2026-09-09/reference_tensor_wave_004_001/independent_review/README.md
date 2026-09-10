# Independent CPU review of the wave004 tensor prototype

**Verdict:** no concrete blocker found for the stated CPU parity prototype. All 43 manifest entries match, and all ten owner tests pass independently. This receipt does not admit a physics launch, a PPO observation schema, GPU throughput, or useful walking.

Reviewed source: `../reference_tensor_wave_004_001`. Its freeze manifest SHA-256 is `2c258dfccbae306d0a0e6c9231b744c7ef365cd059bf04553d47756ab7cdb0c3`; the implementation SHA-256 is `b3abc891eb2416e44c4269ef955769b0f046682e57ec4515c0cfdbd3ddebad8f`. This review did not modify that source, main, or any prior artifact.

The ten-test run includes 790 controls over six asynchronous rows, different command/stop times, a selected-row reset while other rows continue, named joint order and preload, provisional landing/contact loss, confirmed touchdown, latched failures, malformed inputs, and fresh episode identities. The independent output is in [tests.log](tests.log). A benign scalar-oracle tensor-construction performance warning is preserved.

The actual004 measured trace is retained at SHA-256 `0b18528e8d537cce7c3591d407970bf81348fa34b09e95080d30c2b6c7c813ee`. Both the frozen new scalar reference and tensor port reject sample 624, time 12.50 s, with the unchanged 12 mm landing-bound error (tensor code 10). These are unchanged measured inputs from an older physical execution. Agreement is a counterfactual replay check, not evidence that the new horizontal timing has passed physics.

The float32 position flag is executable state. The controller rounds body position after each actual update and each of the 198 prediction updates, matching the scalar oracle's in-place NumPy recurrence. The two `.float().to(self.dtype)` conversions remain on the selected device. The independent 24-case mixed-precision prediction check agrees with the oracle to 1.39e-17 m in position and 7.78e-16 in rotation. Casting only the final prediction changes all 12 float32 cases, by up to 1.1921e-7 m. That optimization would therefore change semantics and requires a new comparison.

[checks.json](checks.json) records the manifest check, cases and source audit. Neither `batch_wave.py` nor its production tensor geometry kernel contains `.cpu()`, `.numpy()`, `.item()` or `.tolist()` calls. Inspected dynamic branches use Python metadata or fixed operations, not per-replica host decisions. The 198-step loop is fixed and batches every row. Constructor file hashing and offline fixture packing occur on the host as documented. This is static inspection plus CPU execution, not a proof of the future Isaac/CUDA adapter's behavior.

One nonblocking documentation nit remains in the frozen source: `_predict` first says there is no 198-step loop, immediately before the later comment and actual fixed 198-step loop. The README describes the implemented recurrence correctly. Preserve the frozen source; correct that comment in a later revision if one is made.

Reproduce the independent checks with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tmp/reference_tensor_wave_004_independent_review_001/review_checks.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/reference_tensor_wave_004_001 -p test_batch_wave.py
```

`harness_initial_error.log` preserves a setup error in the extra review harness: it initially moved the synthetic floor-relative reset height, so the fixture correctly rejected missing support. The final harness changes horizontal position only. This was not a controller failure.
