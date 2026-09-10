# Wave003:7mm swing lift after the measured RF clearance failure

2026-09-10. A separately versioned, one-parameter successor to frozen wave002. **The physical2mm minimum-clearance gate is unchanged.** The runtime code changes only nominal swing lift from5mm to7mm. The landing state, target/PVA limits, body/foot tracking bounds, torque/support/contact gates and finite stop behavior remain unchanged. No new physics result is claimed.

## Actual failure being addressed

The reference003 physical screen passed standing and confirmed three measured steps in orderLF,RR,LM. RF then spent32consecutive samples below the support-force threshold, including sustained zero-force flight, and returned2.0676N contact at12.34s. Planned apex had passed and actual descent was observed. The failed predicate was **measured toe rise1.445mm<2mm**, not force-threshold flicker or missing flight.

At the12.06sRF apex, virtual targetZ was+4.460mm. Transforming the exact commanded joints through the **actual measured body pose** places that target at+1.487mm; the actual toe was+1.090mm. Thus approximately2.974mm of target-height shortfall came from actual-versus-desired body pose and0.397mm from joint deflection. Initial RFtoe height immediately before flight was−0.355mm. The late first-cycle RFstride was about69.80mm, versus34.50mm for the firstLFstride, so the first successful step did not establish all-leg clearance.

`actual_failure_report.json` and the exact copied trace/final rejected reference preserve these findings. Raising virtual lift is a small test of clearance reserve, not a correction to measured body pose or a promise of improved physical lift.

## Why7mm rather than8mm

A full23-case synthetic study at8mm passed22cases. Pure right strafing hit the right-middle tibia's required residual joint margin at control829:0.019899rad available versus0.020rad required. The same5mmcase passed. We preserved that rejection and did not relax limits.

Focused7mm and7.5mm right-strafe probes both completed10synthetic steps and finite quiet hold. Their minimum joint margins were0.024713rad and0.021166rad respectively.7mm provides more remaining geometric margin while adding2mm nominal lift. The final full23-case7mm study is recorded separately from the rejected8mm study. These are ideal measured-motion/contact fixtures; physical body response, torque, slip and clearance must pass the next full-robot screen.

## Verification

`cpu_report.json` / `cpu_study.log` record the final7mm study: stand,16translation bearings, both yaw directions and four mixed-twist arcs, each with4sstand,24smotion and12sstop. Every motion case must complete at least six synthetic touchdowns, no new liftoff after stop and finite reference quiet hold. `cpu_report_8mm.json` / `cpu_study_8mm.log`, `right_strafe_limit_comparison.json` and `intermediate_height_probe.json` preserve the rejected alternative and narrower probes.

Sixteen unit tests cover geometry/limits/contacts and the unchanged bounded-landing transition. The historical actual-reference002 prefix tests explicitly configure5mm to verify that original recorded trajectory; they are not mislabeled as7mm physical replay. The current default7mm is exercised by the full all-leg study and retained forward-wave/stop test. No old checkpoint or policy observation schema is transferred.

The target/reference state structure is unchanged. The exact config/source identity still changes and must be rebound explicitly by any future observation builder, physical adapter or checkpoint. Runtime files remain `wave_reference.py`, `wave_math.py`, `serial_geometry.py`, `geometry/candidate_c_reference.json`, `geometry/f050_t060.urdf`; only the first differs from wave002.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_reference_wave_003 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 tmp/omni_reference_wave_003/run_cpu_study.py
```

Fresh physics still must prove all six actual steps, unchanged measured progress and stopping, no torque/contact failures and the final quiet window. Three successful earlier steps remain evidence, not completion of Stage2.
