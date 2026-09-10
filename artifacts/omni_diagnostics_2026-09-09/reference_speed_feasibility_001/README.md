# Faster single-leg wave timing: CPU rejection evidence

All 12 declared faster combinations exceed the existing executable reference velocity or acceleration budget. The unchanged 2 s swing / 0.005 m/s baseline completes this synthetic test. No faster setting is selected for a physical screen or omnidirectional follow-up.

This result concerns the **exact current swing template and target budget**. It is not a measured motor speed limit, a torque test, or proof that faster walking is impossible. In particular, a different support sequence can shorten the interval between a leg's placements without compressing this same swing curve.

## Method and source

[PLAN.json](PLAN.json) was saved before running the matrix. [study.py](study.py) imports the six byte-identical frozen wave005 geometry/controller/fixture files under [oracle](oracle); their original hashes and the actual009 named soft-limit input hash are in the plan. The wave005 freeze is `5b6c076cb03426ac8e24cc6df7e8684e48862aa78a495bd61ae49773a7826814`. The exact C URDF is `e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c`. This is the user-selected length-study C geometry, not a new production CAD revision.

The only configuration changes are `swing_s` and `max_translation_mps`. The latter equals each requested speed, so the old 0.005 m/s cap does not silently reduce a faster request. Every case records its complete configuration, requested/admitted twist and successful prefix; the admission factor remains 1. The command filter, foothold-horizon algorithm, 7 mm lift, 80% horizontal timing, 0.3 s contact hold, 2 mm qualified flight, five-foot support, landing bounds and finite stop logic remain unchanged.

Each case has 2 s initial quiet, 24 s requested forward motion and 12 s stop. An ideal fixture perfectly follows the desired body and emitted joint positions, with synthetic flat-plane contact. It has no dynamics, ground forces, PD preload, collision geometry or sensor noise. Contact flags and support-margin calculations exercise the controller but do not qualify physical support. Actual009 is used only for the exact named soft limits; its motion has not been time-warped or transplanted.

The formal total target budget is 2 rad/s and 8 rad/s² (0.04 rad per 20 ms comparison), reserving 0.25 rad/s and 2 rad/s² for a future bounded residual. The reference therefore has 1.75 rad/s, 6 rad/s² and 0.02 rad joint margin. These are executable target constraints, not measured motor specifications. The 1.6 N·m torque ceiling is unchanged and untested here.

## Results

The table below is generated from [report.json](report.json) and [failure_classification.json](failure_classification.json). Times are measured from command onset, including the existing command ramp. Values are **the first rejected knot**, not the peak demand of an unexecuted remainder. No failed target is emitted or clipped.

| Swing (s) | Request (m/s) | Prior synthetic touchdowns | First crossing after command (s) | Active leg / phase | First exceeded bound | Joint margin (rad) |
|---:|---:|---:|---:|---|---|---:|
| 1.5 | 0.01 | 5 | 10.12 | RM / swing | 1.7697 > 1.75 rad/s | 0.2943 |
| 1.5 | 0.02 | 1 | 2.44 | RR / swing | 1.7799 > 1.75 rad/s | 0.2760 |
| 1.5 | 0.04 | 0 | 0.36 | LF / swing | 6.6159 > 6 rad/s² | 0.3700 |
| 1 | 0.01 | 3 | 4.44 | RF / unloading | 6.1227 > 6 rad/s² | 0.2907 |
| 1 | 0.02 | 1 | 1.56 | RR / unloading | 6.3157 > 6 rad/s² | 0.3321 |
| 1 | 0.04 | 0 | 0.08 | LF / unloading | 6.0959 > 6 rad/s² | 0.3730 |
| 0.75 | 0.01 | 2 | 2.50 | LM / unloading | 6.5453 > 6 rad/s² | 0.3277 |
| 0.75 | 0.02 | 0 | 0.12 | LF / unloading | 6.2219 > 6 rad/s² | 0.3690 |
| 0.75 | 0.04 | 0 | 0.06 | LF / unloading | 7.3967 > 6 rad/s² | 0.3730 |
| 0.5 | 0.01 | 0 | 0.06 | LF / unloading | 8.1108 > 6 rad/s² | 0.3686 |
| 0.5 | 0.02 | 0 | 0.06 | LF / unloading | 7.9198 > 6 rad/s² | 0.3697 |
| 0.5 | 0.04 | 0 | 0.04 | LF / unloading | 8.6032 > 6 rad/s² | 0.3726 |

The baseline completes 10 synthetic touchdowns, stops without new liftoffs, and reaches exact reference quiet 5.54 s after the stop request. Its valid target peaks are 0.9363 rad/s and 1.8581 rad/s², with at least 0.2399 rad joint margin. This is a synthetic regression, not a second physical admission or a claim that all faster settings were comprehensively optimized.

All 12 first-crossing violations belong to the active swing leg: three during qualified swing and nine during unqualified unloading. None is a landing or planted-stance first crossing. Every rejected candidate has valid FK/IK, at least 0.27596 rad joint margin and minimum Jacobian singular value at least 0.04546 m, versus the 0.002 m guard. Thus proximity to the joint limit or IK singularity does not explain these first failures. The analysis does not assert that stance or landing would remain feasible later; the test ends immediately on the first rejection.

The strongest completed faster prefix is 1.5 s / 0.01 m/s: five synthetic touchdowns before the RM swing crosses 1.75 rad/s at 10.10 s. Passing several individual steps is insufficient to promote that setting. Per the declared selection rule, no reverse/strafe/arc matrix was run because no faster forward case completed.

## Reuse and limits

[traces](traces) contains each successful target prefix and its first failure, including the attempted q/v/a vectors, exact configuration, active leg, synthetic events and timestamps. [classify.py](classify.py) derives joint/phase/IK diagnostics from those saved knots without continuing a failed rollout. The small target-bound audit subclass records candidate arrays and then calls the unmodified parent check; its target/verdict parity is tested.

For another physical route, the opposing-pair load-transfer diagnostic can test whether passive PD distributes load across four feet with useful torque margin. That is complementary evidence: it could permit a different support schedule and shorter foothold horizon, rather than accelerating this rejected single-leg template. No pair gait, controller replacement, training run or new constraint has been adopted by this study.

Changed swing durations/speed limits would require new explicit scalar, tensor and observation identities. The existing frozen wave005 batch and 846/849 observation bundles must not silently load these configuration changes. No actor or checkpoint was loaded, no GPU was used, and no frozen source, geometry or gate was edited.

## Reproduction

From the repository root, with NumPy, SciPy and CPU Torch available:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_speed_feasibility_001/study.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_speed_feasibility_001/classify.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tmp/reference_speed_feasibility_001 -p test_study.py -v
```

Run in a copy if preserving the published frozen outputs. The five tests pass; see [tests.log](tests.log). They cover allowed configuration changes, actual named soft-limit intersection, parent/audit parity, no emitted/clipped rejected target with a latched failure, and finite rejection when no measured flight occurs. One development test initially treated the geometry's already-soft limits as hard URDF limits; the test was corrected to parse the actual hard URDF bounds and compare the exact intersection. Runtime limits and study results were not changed.
