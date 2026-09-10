# Contact-aware opposing-pair motion: CPU prototype 001

The 0.01 m/s forward candidate completes the 44 s ideal-contact fixture with 10 completed pairs, 20 independently qualified synthetic landings and no new liftoffs after the stop request. Its peak executable reference velocity is 0.759201 rad/s and acceleration 1.786553 rad/s²; zero-residual target lag is exactly zero. This is evidence for a bounded first physics trial, not a measured walking speed or PPO admission.

The new support sequence is LM/RM → LF/RR → LR/RF, retaining the original 2 s swing, 7 mm lift and 0.30 s confirmed-contact hold. The planned cycle is 6.9 s, compared with 13.8 s for the scalar six-leg wave. Independent measured landing delays can lengthen the cycle. All three static pair partitions have passed separate actual Isaac transfer tests; those tests do not establish moving contact or torque feasibility. Exact geometry, scalar wave005 and bounded residual sources are retained in `oracle/`, with identities in `SOURCE_INPUTS.json`.

## Current evidence

`report_002.json`, `baseline_targets_002.npz` and `STATE_SCHEMA_002.json` bind the current runtime. Fourteen owner tests pass in `tests_002.log`; an independent review additionally exercises early obstacle contact, a landing bounce longer than 100 ms, and interruption after only one foot has confirmed its landing.

| Contact-aware ideal fixture | Result | Peak reference velocity | Peak reference acceleration | Minimum support margin |
|---|---|---:|---:|---:|
| 0.01 m/s | All 2,200 controls; 10 pairs | 0.759201 rad/s | 1.786553 rad/s² | 102.828 mm |
| 0.015 m/s | Retained support absent at 21.84 s; 8 pairs | 1.169579 rad/s | 2.727707 rad/s² | 96.201 mm |
| 0.02 m/s | Retained support absent at 22.98 s; 8 pairs | 1.594225 rad/s | 3.657882 rad/s² | 89.046 mm |

The higher-speed extrema cover only accepted prefixes. `ideal_fixture.py` explicitly assumes that the measured body follows desired motion and classifies contact from FK against a fixed, sharp per-foot plane with a 10 µm tolerance. Its illustrative forces do not model dynamics or compliance. The higher-speed contact failures remain failures of this fixture; they do not prove those speeds physically impossible. A separate fixed-joint-deflection stress fixture also loses retained contact at 2.98 s and remains rejected.

`target_speed_comparison.json` separately evaluates the exact old **planned-endpoint** oracle: 0.015 and 0.02 m/s each complete 2,200 target knots with respective velocity/acceleration peaks 1.120962/2.645713 and 1.528083/3.549741. This comparison changes only declared speed and candidate identity. It has no measured landing qualification and does not overrule either contact-aware rejection. The contact-aware 0.01 m/s candidate remains the proposed first physical test.

The 0.01 m/s reference becomes exactly quiet at 31.90 s, 5.90 s after the stop request. This leaves 12.10 s of reference hold, or 10.10 s after a further 2 s settle. Actual body/leg quiet and stopping distance must be measured in physics. Target P/V/A are independent of the unresolved SDK angle/rate-integral bias; this prototype neither fixes that bias nor substitutes finite differences for existing physical metrics.

## Controller and caller responsibilities

`PairContactReference` accepts a single measured snapshot and an explicit `[forward, left, yaw]` request. This version accepts nonnegative forward or zero only. It emits named-runtime-order reference P/V/A and complete new controller state. It never writes robot pose or simulator joint state. Reset preserves the last executed target and its PD preload; measured FK/contact anchors and virtual reference anchors stay distinct.

Each active foot retains its own flight baseline, force-free run, measured lift, apex/descent, landing curve, contact confirmations and deadlines. Two force-free samples alone do not qualify flight: each foot must independently reach 2 mm measured lift by its planned apex. Landing requires the inherited speed, position, correction, excursion and three-sample confirmation rules. A first-confirmed foot must retain contact while its partner finishes. New pair liftoff requires all six supports. A missing partner suppresses new motion and fails within the inherited finite deadline; a timer cannot complete a pair.

`distal_contact` and `contact_point_valid` are authoritative **caller-supplied classifications**. The CPU generator checks their Boolean/finite-point consistency; it does not read force thresholds, sensor timestamps, contact-buffer overflow, or torque clocks. A real adapter must prove the existing named distal ≥1 N classification, current sensor evidence, complete contact buffers and pre-reset terminal flags before supplying the snapshot. It must independently preserve complete 400 Hz torque, pose, joint and control-boundary evidence. Synthetic contact points and illustrative forces are not sensor or physical support admission.

The paired branch proposes four retained contacts and a measured articulated-COM support margin of at least 50 mm. The original scalar five-support wave gates remain unchanged. Moving body tracking (35 mm) and retained-toe drift (20 mm) are inherited from the moving-wave controller; the static transfer's absolute-body-displacement bound is not silently reused for a translating robot. All inherited landing/stop thresholds and target limits are constructor-locked. Only the explicitly declared forward cap may vary from 0.01 through 0.02 m/s for CPU comparisons. Total executable limits remain 2 rad/s and 8 rad/s², including the existing residual reserve of 0.25 rad/s and 2 rad/s²; reference limits remain 1.75 rad/s and 6 rad/s², with a 0.02 rad joint reserve.

## State, history and interruption

`pair_contact_motion_v001` is a new state version. `state_packet.py` defines `pair_contact_motion_fragment_v001`: **1,014 float64 CPU features**, schema SHA `da863625e026f8c96297ae6e6a28f84a008cf7956c946dc314d95980db038326`. It covers the shared desired motion, filter, anchors, preload, bounds and reference history plus six fixed per-foot slots with independent qualification state and complete swing/landing coefficients. Controller slots use named LF/LM/LR/RF/RM/RR order; emitted joint targets use the recorded articulation order.

This is only a controller-state fragment. It is not a full actor/critic observation composition, normalized device packet, training registration or checkpoint migration. The existing scalar 846/849 observations remain unchanged and incompatible. `FragmentHistory` requires contiguous 20 ms samples within an explicit reset epoch and clears all slots on a new epoch. `checkpoint.py` checks source/configuration/runtime identity and restores controller, executable reference/residual state and history only at a matching measured time/target boundary. It does not restore physical state or authorize resuming on a different robot pose. Tests replay interruption during a partial landing exactly.

## Preserved development history

`tests_initial.log` records the initial failing expectation that combined name-order permutation with a fixed-deflection contact fixture. The correction separates permutation correctness from that contact stress fixture; no tolerance was enlarged. The 13-test predecessor, its reports/NPZ and exact runtime/test bytes in `history_before_handoff_guard/` remain intact. A subsequent review added the confirmed-foot support guard and its regression. The authoritative 14-test evidence is suffixed `_002`; earlier reports are historical, not claims against the final runtime. `verify_bundle.py` checks both lineages and final payload hashes.

## Next actual admission

`CONTRACT.json` is an explicit **proposed** paired-motion contract. Before motion, require fresh source-bound 32 × 1,000 standing and all existing per-replica quiet gates. Then run one independent zero-residual 0.01 m/s forward/stop case, retaining the exact asset/actuator/solver settings, canonical startup, target limits and recorded named frames. The new paired test needs four valid retained supports, independent flight/landing proof for both moving feet, actual progress and stopping, no nonfoot contact/reset, complete substep torque evidence and the existing quiet/motion consistency reports. Preserve original 50 Hz metrics alongside 400 Hz diagnostics; do not choose a favorable replacement. No actual speed, policy, terrain, sensor or Stage 2 qualification is granted by this bundle. No GPU adapter or launch is included.

Run focused CPU tests from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_pair_motion_001 -p 'test_pair_motion.py'
python3 tmp/reference_pair_motion_001/verify_bundle.py
```

Evidence generators write outputs beside their source; rerun them only in a fresh copy of this frozen directory.
