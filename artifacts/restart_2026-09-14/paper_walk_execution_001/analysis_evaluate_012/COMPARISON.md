# Native evaluation012 compared with011

All three012 trials completed their full requested recordings. Every numeric verdict
remains failed; this analysis does not complete any of the96 required Stage2 cases.
The robot, physics, ordered cases, duration and velocity measurement points match011.
The checkpoints and BC training configuration differ, so the comparison is
descriptive and does not isolate one change causally.

## Estimator calibration and motion

Values below use **whole trials**, including reset settling. Estimator errors compare
the recorded pre-policy prediction with the same-time native **root-origin** navigation
velocity in the critic, in forward/left/up order. COM tracking is a separate channel.

| Case |011 estimator RMSE(m/s) |012 estimator RMSE(m/s) |
|---|---|---|
| Forward0.05,20s |0.28330 /0.27470 /1.59837 |0.02979 /0.05183 /0.02203 |
| Zero,20s |1.19682 /0.76471 /1.24645 |0.00629 /0.01687 /0.02011 |
| Forward8s, then zero13s |0.52278 /0.90064 /1.36631 |0.06955 /0.04680 /0.03374 |

Calibration improved substantially in these recordings. It did not produce the
requested forward motion. The20s forward trial's mean signed COM speed was
−0.000443m/s in011 and−0.000576m/s in012 for a+0.05m/s command. Within the unchanged
2s-settle static scoring window it was−0.000060 and−0.000047m/s. The full-trial planar
tracking RMSE increased from0.05223 to0.05787m/s. The stop trial's first8s moving phase
likewise averaged−0.001080 and−0.001470m/s.

## Quiet windows are distinct from whole-trial behavior

The20s zero trial uses the unchanged controls200–999(4s settle,16s quiet). The stop
trial uses controls500–1049(8s moving,2s settle,11s quiet). No suffix was selected.

| Fixed quiet-window metric |Zero011→012 |Stop011→012 |
|---|---|---|
| Worst joint RMS at50Hz(rad/s) |0.180679→0.049253 |0.822116→0.480717 |
| Worst joint RMS at400Hz(rad/s) |0.278105→0.056659 |1.118302→1.827122 |
| Maximum joint target-step p95(rad/20ms) |0.006415→0.002582 |0.018372→0.007147 |
| Native samples missing six-toe support |30→0 |411→38 |

The zero trial improved, but still fails the original0.03rad/s quiet joint RMS and
0.002rad target-step gates. The stop trial remains above both, exceeds the0.02rad
joint-range gate, loses six-toe support38times in its fixed quiet window, and has
the native speed violation described below. These are counts of400Hz samples,
not necessarily38separate contact-loss episodes.

Across the complete012 trials, missing-six-toe samples were676,253 and1238. Each
had12initial no-toe-support samples; none remained in the fixed scoring windows.
No active nonfoot contact was recorded. Saved shaft-category patch tuples are
retained in the JSON; category alone is not an above-threshold nonfoot-contact event.

## Exact reason for the stop trial's native physical rejection

One SDK-reported speed violates the native joint-speed bound:

- At10.0275s, sequence4010/control501/substep2, `rf_tibia_pitch` reports
  −114.842803955rad/s against50.265483856rad/s plus the unchanged2e−6 tolerance.
- The joint position is0.557851315rad, within limits. Root height is0.093481481m.
  There is no fall, endpoint termination, truncation, reset, nonfoot event, clearance
  failure or torque-cap breach. All1050controls and8400physics steps are retained.
- The angle difference over that interval gives+1.992630959rad/s. This differs from
  the SDK velocity; the next interval is−27.484679222rad/s. The observed discrepancy
  is preserved, and its underlying simulator/contact cause is not established.
- Applied torque during the violating step is−0.180959761Nm. On the next step the
  recorded speed enters PD/derating: demand12.775898933Nm, allowed/applied torque0.
  At the control endpoint, speed is9.689039230rad/s and termination is false. The
  full400Hz physical screen correctly catches an event that endpoint-only checks miss.

`NUMERICAL_COMPARISON.json` preserves neighboring motor/state samples and contact
packets. The single speed event, not a fall claim, explains
`native_motor_and_joint_checks_pass=false`.

## Targets, spectra and recording integrity

Targets are normalized as(held_target−neutral)/0.35rad. Full-trial peak absolute
normalized targets in012 were1.000,0.877 and1.000; action outputs beyond±1 were
0.0389%,0% and0.0370%. The exact emitted-target recurrence, native motor recurrence,
applied-input readback, consecutive pre/post states and all50Hz endpoints were
reproduced for all48,800substeps across011/012. Recorded previous-target features
agree within2e−7CPU/CUDA float32 arithmetic tolerance; physical gates are unchanged.
Strict saved-actor reconstruction matched all3050recorded012actions within1e−5;
maximum absolute CPU error was4.18e−7.

Full-trial012 joint-velocity Hann spectra peak at3.55,4.30 and1.762Hz. Fractions of
AC spectral power above5Hz are53.0%,5.0% and81.7%. The stop fixed-window spectrum
has1.58%above5Hz even though its400Hz RMS is worse: the large speed spike lies close
to the window's beginning and a Hann window attenuates boundary samples. The raw RMS,
event audit and gate failure retain it. Spectral power fractions are descriptive,
not a substitute for native physical bounds.

The maintained analyzer verified file manifests and all contiguous native arrays.
The additional artifact-only verifier streamed all48,800contact packets containing
1,124,861saved patches across both evaluations, checking packet/counter order.
It does not claim to independently regenerate mesh contact classifications.
All original inputs and failed verdicts are unchanged. The exact input/output hashes,
window metrics, estimator biases and per-joint normalized-target metrics are in
`NUMERICAL_COMPARISON.json` and the final manifest.
