# Native mimic velocity residual — targeted solver experiment

The completed **`d6d5863` eight-environment, 128/1 TGS, Kp30/Kd0.30** standing
trace contains a large velocity-level residual despite small position error.
This supports testing **128/4** with all other physics, controller, geometry,
timing, targets and acceptance bounds held fixed. It does not establish that
four velocity iterations will solve the problem.

At physics sample **1928**, beginning at **2.410 s**, environment 7 at
(−2, 0) has:

| Quantity | Recorded / derived value |
| --- | ---: |
| Right-rear pushlever final velocity | −0.55521846 rad/s |
| Right-rear pushrod final velocity | +4.98058748 rad/s |
| Their required sum, ideally zero | **+4.42536902 rad/s** |
| Position relation error | −2.02656e−6 rad |
| Relation error derived from position changes / dt | −0.00548363 rad/s |
| C-pin relative velocity, hinge-local Y | **−0.36232284 m/s** |
| C-pin separation magnitude | 0.798013 µm |
| Right-rear vertical foot force | **76.85047 N** |

The final joint velocities are identical to the next sample's direct backend
readback. Independent body/link velocity fields also produce the C-pin
velocity discrepancy, so this is not explained by the previously suspected
cached-state mismatch. The following step has zero right-rear pad force and
the femur demand spike documented in `../batch_analysis/README.md`. That
following step also contains actual motion; not every joint-velocity excursion
is merely a reporting discrepancy.

NVIDIA describes TGS position iterations as internal substeps. Its final
velocity iterations solve the last substep's unbiased constraints; those
velocities carry into the next frame. Therefore extra final velocity passes
can target residual coupling among contacts and bilateral constraints without
reducing the internal timestep further. NVIDIA recommends one velocity
iteration by default, so four is a measured candidate, not a universal
recommendation. [PhysX solver documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#constraint-solver)

**Prediction:** under the same eight-world/200-control trace, 128/4 should
reduce peak/RMS `qdot_passive − multiplier*qdot_motor`, C-pin relative
velocity, and the subsequent derivative-demand spikes. Keep measuring actual
joint-position changes, foot impulses/support and original position/motor
bounds. Lower torque alone would not establish improved constraint accuracy.
If the velocity residual persists unchanged, the final-solve hypothesis is
weakened; increasing position iterations again is not the implied next step.
Any successful candidate still needs matched nominal/refined numerical
qualification and the full standing/driven campaign before PPO.

`result.json` records exact input hashes, full runtime identity, per-world
metrics and the event window. It is independent of the earlier frozen batch
comparison. `analyze.py` verifies the original NPZ hash and kinematic mapping
before deriving residuals; it launches no simulation and modifies no source.

From the repository root:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/velocity_constraint_residuals/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/batch8/hexapod-fourbar-diagnose-20260905T205225Z-8faf9917/report.json \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/kinematics.json \
  --out /tmp/hexapod_velocity_residual_new.json
```
