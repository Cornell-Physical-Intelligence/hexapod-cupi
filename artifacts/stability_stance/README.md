# Lower-stance A/B evidence

These are deterministic, nominal-model replays of the immutable Stage-2
`model_25.pt` checkpoint.  They isolate the reset/default leg geometry; they
are not evidence that the old policy is ready for the deeper stance on every
joystick command.

## Selected forward-only adaptation stance

- Reset root height: `0.185 m`
- Femur (second joint / first pitch joint): `0.6000 rad`
- Tibia: `2.2335 rad`
- Measured moving deck height: `0.1808--0.1815 m`

Against the half-lowered `femur=0.50 rad`, `tibia=2.170 rad` stance at the
same 0.20 and 0.30 m/s anatomical-forward commands, the selected pose reduced
the mean of the two command-local metrics by:

| Deck-motion metric | Reduction |
| --- | ---: |
| Height standard deviation | 8.7% |
| Vertical-velocity RMS | 6.5% |
| Roll/pitch-rate RMS | 4.1% |
| Tilt RMS | 7.3% |

Both forward commands were fall-free.  The worst raw computed-demand peak was
`4.094 N*m` at 0.30 m/s in the two-command replay (`4.220 N*m` in the separate
eight-command replay), below the `5.5 N*m` simulation termination threshold.
The deeper pose does increase motor demand, so Stage 2C still grades all 18
RS05 joints and rejects sustained demand above the `1.6 N*m` continuous rating.

## Admission boundary

The old checkpoint is unsafe as a mixed-axis deployment in this pose.  In the
eight-command diagnostic it ignored pure lateral commands, produced falls on
negative lateral and both oblique commands, and reached `5.610--6.224 N*m` on
the oblique pair.  Therefore the deeper stance is admitted only to the
forward-only Stage 2C adaptation.  Later Stage 2D/2E checkpoints must pass
command-local fall, deck-stability, tracking, and RS05 gates before the stance
is accepted for joystick use.

Source reports:

- `20260825_half_qf0p50_qt2p170/eval_forward.json`
- `20260825_deep_qf0p60_qt2p2335/eval_forward.json`
- `20260825_deep_qf0p60_qt2p2335/eval_8cmd.json`
