# Initial 1600 Hz release preparation

The first complete suite ran 935 tests and failed only the new regression
fixture’s assertion that the production validator was still the old, unapplied
version. The conservative measurement patch had already been installed. Its
actual CUDA comparison passed all eight tests and 400 complete rows; no
numerical mismatch caused this local integration-test failure.

The next fixture revision explicitly distinguishes the unapplied-proposal check
from comparison against the installed candidate. Existing physical thresholds,
full-row equality, and guard/reset checks remain unchanged. This preparation
snapshot is not a physics admission or training release.
