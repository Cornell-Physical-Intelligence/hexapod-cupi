# One opposing-pair candidate for faster reference motion

A planned opposing-pair crawl at **0.01 m/s** completes the CPU target study within the existing executable bounds: **0.72925 rad/s velocity, 1.72746 rad/s² acceleration**, zero reference/executable lag, and at least **0.23891 rad joint margin**. Its minimum planned projected support margin is **104.18 mm**, above the separate four-support diagnostic's proposed 50 mm bound.

This is useful target feasibility, **not a working faster gait**. The study schedules ideal foot endpoints; it does not simulate forces or run the measured-contact landing state machine. Real unloading, body tracking, foot slip and event-triggered landing corrections can change the outcome. It leaves directional002, wave005, all frozen evidence, production CAD and PPO untouched.

## The one selected change

Move opposing pairs in the order **LM/RM → LF/RR → LR/RF**, keeping the current 2 s swing, 7 mm vertical curve, 80% horizontal timing and 0.3 s planned handoff. The nominal per-leg cycle becomes 6.9 s instead of 13.8 s. The same placement rule then uses a 4.45 s lookahead instead of 7.9 s. The explicit proposed forward command is doubled from 0.005 to 0.01 m/s; there is no hidden old speed-cap derating.

This changes support sequencing rather than compressing the rejected swing curve. The previous frozen `reference_speed_feasibility_001` study rejected all 12 faster single-leg duration/speed combinations at a reference P/V/A bound. Those results remain valid. This new planned sequence is not a controlled proof that pairing alone cures those failures: the old study exercised a full synthetic contact FSM, whereas this bounded study only tests nominal planned endpoint targets. A contact-aware pair controller remains to be built and admitted separately.

`PLAN.json` was written before the run. There is one candidate and no automatic search. The source is the exact five frozen wave005 geometry/math/controller files plus the unchanged bounded residual core. `inputs/actual_pair001_startup_trace.npz` is a byte copy of the audited real startup trace. Its final sample seeds actual named q0 targets, soft limits, body pose and recorded joint/foot preload; no measured q is substituted for the loaded target. Virtual desired body motion is kept distinct from real body motion and is never written to physics.

## Target evidence

The 2,200 accepted controls cover 2 s hold, 24 s forward request and 18 s zero request. Every accepted q knot is passed through the unchanged formal zero-residual target core. Bounds are checked from consecutive executable q/v differences, including the first moving knot, swing transitions, finite stop and final hold. Unreachable or over-budget knots fail before emission; there is no clipping or continuation after failure. Raw SDK joint rates are not used to infer these target derivatives.

| Quantity | Result | Existing target limit or separate proposal |
|---|---:|---:|
| Executable target velocity | 0.729245 rad/s | 1.75 rad/s reference |
| Executable target acceleration | 1.727463 rad/s² | 6 rad/s² reference |
| Residual target lag | exactly 0 | exactly 0 at zero residual |
| Minimum joint margin | 0.238914 rad | ≥0.02 rad reserve |
| LM/RM planned support margin | 184.006 mm | ≥50 mm proposed |
| LF/RR planned support margin | 104.185 mm | ≥50 mm proposed |
| LR/RF planned support margin | 106.115 mm | ≥50 mm proposed |

The total controller limits remain 2 rad/s and 8 rad/s², with 0.25 rad/s and 2 rad/s² reserved for a future bounded residual. The same 1.6 N m torque ceiling is retained but cannot be evaluated by this kinematic study.

There are 11 scheduled paired swings and 11 scheduled endpoint completions, covering all three pair groups. These are **zero measured contact confirmations**. No new swing starts after the stop request. The command filter reaches exact reference quiet after 5.90 s; the final stationary target interval lasts 12.10 s, leaving 10.10 s after the same 2 s quiet settling allowance. This is reference quiet, not measured physical quiet. Desired displacement is approximately 240 mm forward over the complete ramp/stop trajectory; no actual robot progress is asserted.

Projected support margins use the exact mass/COM geometry, the virtual desired body and commanded joints, with four planned planted anchors during a paired swing. Initial recorded preload offsets are retained separately when deriving hypothetical contact centres. The real initial snapshot's corresponding geometric margins are 185.772, 112.554 and 116.388 mm. None of these hypothetical four-foot margins proves that those feet will carry load or avoid slipping.

## What actual evidence supports—and what is next

The earlier `pair_load_transfer_001` physically unloaded **LM and RM together for 2.02 s**. Measured lift was 5.082/5.513 mm, all four corner contacts were retained, and complete 400 Hz requested torque peaked at **1.401371717 N m**, leaving **0.198628283 N m** below the unchanged ceiling. Those unequal diagonal load measurements support trying another four-support sequence. They do not establish torque margin for either diagonal pair, faster transitions or translation.

The next smallest physical discriminator is two separately declared static transfers of **LF/RR** and **LR/RF**, using the same previously measured 3 s raise / 2 s hold / 3 s return shape. Each starts from fresh canonical hold after exact-source 32×1,000 physical-plus-quiet admission and retains the original four-support diagnostic rules: named remaining supports, ≥50 mm measured projected COM margin, real ≥2 mm paired flight and ≥1 s simultaneous unloading, no forbidden contact/reset, all six contacts confirmed on return, every diagnostic 400 Hz requested torque ≤1.6 N m, bounded body motion/support drift/slip, and ≥10 s measured quiet after settling. This tests the currently unknown load distribution without changing the target speed at the same time.

Only if both remaining pairs pass should a new measured-contact pair controller attempt the proposed 0.01 m/s forward/stop trace. It must require both swing feet to qualify real flight and complete bounded individual landing reconciliation, retain all four remaining measured supports and the same torque/target limits, and prevent the next pair from lifting until both prior feet have stable confirmed contact. A missing/early/inconsistent landing must stop or reject within declared bounds, never be replaced by this study's scheduled endpoint completion. Independent actual position/rate progress, raw 50/400 Hz evidence and measured final quiet remain necessary. The unresolved reported-rate bias must remain visible.

The original five-support wave gates are unchanged. Four-support criteria apply only to a new named diagnostic and cannot admit the old wave/PPO task. A pair policy would also require a new explicit state/observation schema for two active legs and both landing states; the current 846/849 scalar-wave contract cannot silently represent it. Terrain would need current support/height evidence and reacquisition of each landing footprint; this multi-second flat lookahead does not extend a sensor-map lease.

## Reproduction and files

`planned_targets.npz` contains initial and all accepted q/v/a, exact times, desired pose/commands, planned foot points, support masks/margins and active pair IDs. `report.json` preserves source identity and all scheduled events. `tests.log` records six passing checks: exact initial/final derivative continuity, every-knot bounds, planned support partition/stop, runtime joint-name permutation, fail-closed unreachable target, and strict source/JSON lineage.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_opposing_pair_feasibility_001 -p 'test_study.py' -v
```

The study is CPU-only. `study.py` rejects overwriting an existing report; reproduce it in a fresh copy. There is no Isaac launcher, physics adapter, training job or hardware admission in this bundle.
