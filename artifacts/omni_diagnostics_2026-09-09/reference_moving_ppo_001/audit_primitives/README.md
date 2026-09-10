# Independent moving-pilot raw audit

This CPU-only tool reads completed or rejected outputs from consumer001, pinned
to `dd49fd3e3639e453b067b9d4d67e5465db20865200c6a17616dd6936b527b842`.
It issues no physical admission and never starts, stops or changes a job.

`audit.py` reconstructs the command schedule and episode ages independently from
the raw initial 200-control startup and subsequent row outcomes. It checks all
eight physical samples per control, exact endpoint equality, named joint shape,
motor cap, per-row requested torque events, native/contact/reference terminal
causes, exactly 200 excluded recovery controls, reset-only clock epochs and
consecutive float32 clock advancement for all 14 sensors. It reconciles raw
learnable/terminal/timeout counts with every completed 256-control PPO update and
checks that the event penalty appears exactly once. Partial field masks remain
explicit; incomplete evidence does not receive a complete replay result.

The current moving consumer does not export actor packets, encoder history-valid
arrays or critic bootstrap tensors. This audit cannot directly prove those GPU
tensors correct. Their reset isolation and final pre-reset bootstrap remain
covered by the frozen consumer's executable CPU tests. Episode and clock replay
are useful evidence, but do not replace those missing raw channels.

`retention.py` compares cold initial/final forward-to-stop outputs. It independently
recomputes measured speed, body displacement, the unchanged position/integral
comparison, quiet joint range and target motion, with all actual substeps and
sensor clocks retained. Raw SDK joint rates and angle-difference interval averages
are reported separately. Existing physical gate results remain unchanged.

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_moving_ppo_audit_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_moving_ppo_audit_001/audit.py --phase RUN/train_10 --out NEW/audit.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_moving_ppo_audit_001/retention.py \
  --initial RUN/evaluate_initial --final RUN/evaluate_010 --out NEW/comparison.json
```

Three focused tests pass, covering a mixed 32-row recovery, a requested-torque
spike hidden between control endpoints, and corrupt episode/count/penalty/clock
evidence. The two `actual_parent_*_replay.json` receipts additionally exercise
the numerical readers against byte-bound, already published actual parent
evaluations. These are existing stand-only smoke results, not new moving-pilot
admission or evidence that training improved walking. No moving-pilot result is
included in this preparation bundle.
