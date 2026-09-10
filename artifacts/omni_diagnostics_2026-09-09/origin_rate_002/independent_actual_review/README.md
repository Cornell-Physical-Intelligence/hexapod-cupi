# Independent review of completed origin002

All five matched origin cases pass the unchanged physical and quiet-standing gates. Moving the robot's global XY coordinates **perturbs but does not resolve** the angle-versus-reported-rate discrepancy. The repeated origin case reproduces every recorded control and substep array exactly. No native cause or velocity-fidelity qualification follows from this diagnostic.

## Physical and quiet results

Each case completed 1,000 controls (20 s), with the original 4 s startup/settling excluded from scoring and 16 s of quiet data. The fresh preceding 32-replica standing run also passes every quiet and physical gate when recomputed. All cases retain six distal supports, zero forbidden-contact steps, zero termination/truncation, zero postsettle saturation and zero reference-to-executable target lag. Commands and residual actions are zero; executed joint target trajectories are exactly equal across all five cases.

| Case | Global XY (m) | Physical / quiet | 400 Hz peak requested torque (N·m) | Maximum 50 Hz reported joint RMS (rad/s) |
|---|---|---|---:|---:|
| origin_a | (0.0, 0.0) | pass / pass | 1.478385 | 0.025507 |
| near | (3.0, -5.0) | pass / pass | 1.483903 | 0.025609 |
| far | (30.0, -50.0) | pass / pass | 1.453999 | 0.024837 |
| near_opposite | (-3.0, 5.0) | pass / pass | 1.482163 | 0.025597 |
| origin_repeat | (0.0, 0.0) | pass / pass | 1.478385 | 0.025507 |

The table reports postsettle torque. Startup contains the usual applied-torque cap; it is not hidden by labelling the entire recording unsaturated. The full-run and postsettle windows remain distinct in the original traces/gates. The independent 400 Hz peaks above differ slightly from the existing 50 Hz gate samples; neither channel is substituted for the other.

## Rate versus angle

For the identical 4.00–20.00 s interval, I independently integrated all 18 reported joint-rate channels using left, right and trapezoidal quadrature, then compared the results with their actual endpoint angle changes. The same named runtime joint, `revolute_2_1`, has the largest absolute integrated discrepancy in all five cases.

| Case | Actual joint angle change (rad) | Trapezoidal reported-rate integral (rad) | Integral minus angle change (rad) | Root-link integral/displacement error (mm) |
|---|---:|---:|---:|---:|
| origin_a | -0.0004253 | +0.4080621 | +0.4084874 | 0.834318 |
| near | -0.0006971 | +0.4096996 | +0.4103968 | 0.870177 |
| far | -0.0002267 | +0.3974250 | +0.3976517 | 1.536228 |
| near_opposite | -0.0006809 | +0.4094852 | +0.4101662 | 0.826420 |
| origin_repeat | -0.0004253 | +0.4080621 | +0.4084874 | 0.834318 |

The far translation reduces the largest joint discrepancy by about 2.65%; the two near translations increase it by about 0.4–0.5%. Across all joints, the largest paired changes from origin are 0.009722 rad (near), 0.024685 rad (far), and 0.009641 rad (near opposite). These effects exceed the zero observed difference between the two origin recordings, but only one cold run was performed at each translated coordinate. This is not a statistical characterization or evidence of a specific native solver mechanism.

The root-link discrepancy remains 0.826–1.536 mm and is largest at the far translation. COM position/velocity and world angular increments/rates were calculated separately as well; their full values are in [report.json](report.json). Quaternion increments use explicit raw SDK XYZW and world-frame composition, independently checked with SciPy Rotation. The sum of local rotation vectors is a diagnostic, not a replacement for finite rotation composition. Finite-difference channels are retained as evidence and do not replace the SDK rates or relax any existing quiet gate.

## Origin and initial-state equivalence

All generalized q, qdot, target, root Z/quaternion and COM velocity inputs are exactly equal to the frozen selected009 cold-state payload and across cases; only the declared root XY coordinates differ. The initial 400 Hz observation equals those actual readbacks, and every control endpoint matches its corresponding actual substep joint positions/rates and root pose. The selected cold state is replica 6 from the prior009 discrepancy study, not five randomly selected robots. Two reset injections are recorded before the first observed physics step; no body pose is written during the trial.

The source reads and verifies persistent terrain and scene origin views separately. They are **different storage**, with recorded strides `(3, 1)` and `(7, 1)` respectively; they are not interchangeable aliases. Both stores retain the declared XY after writing and re-reading, with Z and the scene default-pose quaternion preserved. All initial storage/view checks pass. These are source-bound recorded initialization checks; this offline review cannot inspect now-destroyed live GPU pointers.

All cases report exactly one external collision surface: an enabled infinite plane at Z=0 with normal +Z. Consequently global XY translation preserves the declared collision geometry. The finite visible ground grid is not treated as physical support. Derived 19-body local positions are not claimed bit-exact: maximum difference after subtracting the translation is 0.507 µm for either near case and 3.070 µm for the far case; it is zero for the repeated origin. Raw float32 positions remain retained, and float64 offline subtraction cannot recover lost precision. These derived differences do not establish the cause of the rate discrepancy.

## Receipt and reproducibility

[review.py](review.py) verified all 96 fetched payloads against root's remote audit and all 933 local source files against source manifest `c028664ae782b053f0e126ace1e7a24e6eb8a89cf8dbaa9fd8b5628d28560160`. It checks exact initial conditions, origin metadata, infinite-plane readbacks, sample index/cadence, control/substep endpoint equality, all 32 preceding quiet verdicts, all five case verdicts and the independent quadratures. It runs the unchanged quiet function/thresholds from the pinned source while recomputing the other numeric comparisons directly from raw arrays. Every assertion passes; see [review.log](review.log).

Root's separately recorded remote audit reports all 550 admitted assets unchanged, exact owned containers absent, and forecasting restored at Unix time 1789027564.5990298. This review verified the fetched audit/payload identities; it did not repeat a live SSH/container/restoration inspection. Source/preparation/raw-result bytes are unchanged. This review used no GPU and grants no new walking, PPO, production or velocity-fidelity admission.

Run from the repository root with NumPy and SciPy:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_origin_actual002_independent_review/review.py
```

Use a copy to preserve frozen review outputs. Raw inputs remain under `tmp/reference_origin_results_002`; the reusable script resolves them relative to this receipt's parent directory.
