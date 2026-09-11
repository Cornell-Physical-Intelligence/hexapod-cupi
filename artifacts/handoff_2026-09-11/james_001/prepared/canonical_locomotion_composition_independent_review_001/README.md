# Independent CPU composition review

The frozen 27-file composition proposal has a concrete clock-validation defect that should be corrected before adoption. It is separate from the active standing source005 and pending canonical PPO smoke, and this review changes neither native source nor physical gates.

The exact owner freeze is `f6b239605e84389a9c3a9b292932b64304c54a853a28c85e83267234b1e637b9`. `replay_clock_boundaries.py` verifies all owner bytes before and after running five small cases from the owner fixture. A single NaN substep timestamp, NaN pre-hold timestamp, or fractional counters/substep indices each produced an otherwise accepted reward of 0.14. The neutral-context helper also accepted NaN time. The finite wrong-time control correctly rejected. Original source and outputs remain preserved.

`abs(NaN - expected) > tolerance` is false, and `int()` silently truncates the fractional values. A fresh successor should require finite scalar timestamps and exact integer-valued, non-boolean counters and indices before arithmetic; the same rule should cover reward packing, neutral context and patch/pose identity boundaries. Keep the original 2.5 ms/20 ms relationships and tolerances, and add old-fail/new-pass regression cases. This finding concerns malformed evidence admission, not a measured clock failure in a native run.

The bounded read found no other concrete issue in the command/history order, root COM-to-origin velocity transform, or material-point slip calculation. That is a limited code review, not native qualification. The proposed packet accessor and moving contact rule remain explicitly unadopted. No GPU, remote action, tracked-file edit, or frozen-owner edit was performed.

Replay with Python and NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B replay_clock_boundaries.py --source /absolute/path/to/canonical_locomotion_training_001
python3 -B -S verify_bundle.py
```

Root owns eventual publication under the hard `docs/PROJECT_SITE.md` framework. This receipt requires an explicit central update record if adopted or published; it makes no Stage 2 or PPO admission claim.
