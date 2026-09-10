# Wave reference004: earlier horizontal completion

10 September 2026. **CPU-tested proposal; no successor physics result.** This changes the timing of the horizontal foot path in the frozen 7 mm wave003 reference. It preserves the requested body twist, original foothold, vertical arc, geometry, stance, measured-contact gates and 12 mm landing bound.

Actual source004 physics completed three measured steps, then RF returned after 45 flight samples with 3.44176 mm lift. It failed the original-endpoint consistency check at 12.23450 mm. At that moment the horizontal target still had 10.49231 mm to travel. The separate [actual failure diagnosis](actual_failure_report.json) preserves the measured failure and explicitly marks the timing counterfactual as unverified.

## Equations and executable behavior

For a new swing of duration T = 2 s and horizontal displacement d, horizontal position is `p0_xy + d_xy S(u)`, where `S(u) = 10u³ − 15u⁴ + 6u⁵` and `u = clamp((t − t0)/(0.80T), 0, 1)`. Its analytic velocity and acceleration use the same shortened duration. After 0.80T they are exactly zero. The vertical curve is byte-equivalent to the previous 2 s quintic endpoint interpolation plus its 7 mm C2 lift bump. The original endpoint is never replaced by this timing change.

`AdvancedHorizontalSwing` is used only at new liftoff, with zero endpoint velocity and acceleration. The existing `Swing` class and the contact-triggered landing blend remain unchanged. Measured flight of at least two consecutive samples, actual lift of at least 2 mm, passed apex, measured descent, bounded touchdown speed, five-foot support, full 3D landing consistency, stable measured touchdown, finite stopping, joint margins and executable P/V/A checks all remain in place.

The API remains `reset(snapshot)` and `step(snapshot, requested_forward_left_yaw, dt=.02)`, returning named `[1,18]` `q_ref`, `v_ref`, `a_ref` and `valid`. Measured pose remains explicit XYZW plus a checked rotation matrix. The generator never prescribes the robot's actual body pose. Discrete reference derivatives and the exact zero-residual core remain the executable budget authority.

**State schema changed:** exported `swing` now includes `horizontal_duration_fraction` and `horizontal_coefficients`. Consumers reconstructing foot phase or derivatives must bind this source/configuration and use the shortened horizontal timing. Existing actor/checkpoint compatibility is not claimed. No PPO actor is trained or injected by this bundle.

## CPU evidence and remaining physical question

All 23 synthetic cases pass: standing, 16 translation bearings at 0.005 m/s, both yaw directions at 0.015 rad/s, and four translating/turning arcs. Every moving case completes ten synthetic touchdowns, suppresses new liftoffs after stopping and reaches finite quiet reference hold. Worst reference velocity is 1.4065 rad/s and acceleration is 3.2388 rad/s², below the formal reference allocations of 1.75 rad/s and 6 rad/s². The surrounding residual controller reserves 0.25 rad/s and 2 rad/s², for combined bounds of 2 rad/s and 8 rad/s². These are engineering experiment settings, not measured motor speed limits. Some cases exceed the separate 1.25 rad/s reference allocation corresponding to a 0.03 rad/20 ms diagnostic; no universal pass under that diagnostic is claimed.

Twenty tests cover the earlier regressions plus unchanged vertical motion and endpoint, C2 horizontal completion, analytic derivatives against finite differences, and continued rejection of the actual frozen 12.23450 mm contact. Historical source002 prefix tests explicitly request its old 5 mm lift and full-duration horizontal path; they do not claim to replay new physics.

The remaining question is whether the actual robot tracks the earlier horizontal motion with acceptable torque, contact and body deflection. Synthetic contact follows geometry and cannot answer it. The faster horizontal trajectory may change loading and touchdown timing. Source004's raw position/velocity integration discrepancy is a separate unresolved measurement issue; this timing revision does not fix it or relax its physical gate.

## Reproduce

Run from the bundle directory with Python, NumPy, SciPy and PyTorch. Copy the frozen bundle before regenerating reports.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 run_cpu_study.py
```

Runtime files are `wave_reference.py`, unchanged `wave_math.py`, unchanged `serial_geometry.py`, and the two unchanged `geometry/` inputs. The manifest records all files. [cpu_report.json](cpu_report.json), [tests.log](tests.log) and [validation_summary.json](validation_summary.json) are CPU evidence only. Physical admission, useful all-direction walking, smooth paths, quiet stop, terrain and sensor qualification remain outstanding.
