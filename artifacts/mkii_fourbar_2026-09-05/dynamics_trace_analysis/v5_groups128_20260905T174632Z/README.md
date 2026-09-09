# Native mimic v5, 128/1 group diagnostic

This diagnostic stayed inside the original closure, passive-coordinate and motor
envelope bounds. The unchanged physical gate failed on four brief samples with
no loaded foot. All 18 group direction responses pass when reconstructed using
the validator's measurement method. This is evidence for a controller experiment,
not admission of training or hardware.

Run `hexapod-fourbar-diagnose-20260905T174632Z-36232e95` completed eight environments
× 200 standing + 700 driven control steps (14,400 physics samples per environment).
The model uses native physical mimic constraints, TGS 128 position / 1 velocity
iteration, external forces each iteration, and 1.25 ms × 16 substeps per control
step. The genuine [report.json](report.json) is preserved unchanged. Its source
contract hash is `ee8785ffb1c50e89e06a78ae965a6d87f2faf3cda68b9eec99bb5795b247e6f1`.

| Measurement | Observed |
|---|---:|
| Largest C-pin gap | 21.0167 µm |
| Largest passive-coordinate error | 0.000254154 rad |
| Largest raw / applied torque | 5.35664 N·m |
| Non-foot contacts / resets | 0 / 0 |
| Minimum group direction, coxa / femur / lever | +0.0608325 / +0.0401865 / +0.0170670 rad |
| Minimum left-middle lever direction | +0.0180698 rad |

Group directions use the mean joint position after each of the last ten control
steps of each sign, then positive minus negative separately in every environment.
All exceed the original +0.005 rad direction threshold. This group-only,
8-environment, 128/1 run omits the full 32-environment qualification's preceding
individual-motor history and differs in iteration count. It cannot isolate why
that earlier full run had a negative left-middle group response.

## What happened at the support gap

All four support-loss observations occur at zero-based sample 11,216, environments
1, 5, 6 and 7. Each environment has exactly one sampled substep (1.25 ms) with no
foot force. This is the first physics update of the second negative tibia-lever
control step, covering 20.00–21.25 ms after that segment begins. All six foot
forces are exactly zero, so the result does not depend on rounding around the
1 N threshold.

Reconstructed sphere-pad bottoms lie only +0.058 to +4.924 µm above the ground
plane. Root vertical velocity is −0.06625 to −0.06657 m/s. C-pin gaps at those
samples are at most 2.079 µm. These poses show microscopic separation at the
sampled instant, rather than sustained ballistic flight or a broken linkage.
The reconstruction uses recorded float32 body poses, normalized native XYZW
quaternions, the hashed URDF's exact sphere origins/radii, and each environment's
ground origin. It does not establish a continuous flight duration or a hardware
contact tolerance. The support gate remains failed.

The largest closure gap is a separate, later event during zero-offset recovery:
sample 13,163, environment 0, right-rear leg. The right-rear foot reports 327.461 N
in world +Z; its lowest sphere is approximately 0.682 µm below the plane. Other
feet remain loaded. One substep later, right-rear femur demand is −5.35664 N·m:
P = −0.695189, D = −4.66145 N·m. Its pre-step velocity is 7.76908 rad/s.
Applied torque equals demand and remains inside the motor envelope. This records
the contact-transient/PD sequence without proving a unique cause.

Cached and direct pre-step joint positions/velocities match exactly over
3,456,000 joint-value samples each. PD reconstruction residual is at most
2.38419e-7 N·m. No cache mismatch or incorrect PD sum is visible in this run.

## Bounded controller experiment recommended by this evidence

Compare continuous position-target interpolation against the current zero-order
hold, preserving Kp = 30, Kd = 0.6, zero velocity target, motor limits, physics
timing, test amplitudes/durations and physical acceptance bounds. The present
0.04 rad target change causes a 1.2 N·m proportional-demand jump at a control
boundary. Sixteen equal increments spread that change into 0.0025 rad /
0.075 N·m increments per physics step. This reduces the setpoint discontinuity;
it does not guarantee lower actual torque or a passing full test.

Use the previous actually sent target as the start and the existing slew-limited
50 Hz target as the endpoint; interpolate before each force update and reset
the history with the robot. Keep desired velocity zero for this first comparison:
adding a 2 rad/s velocity target would itself introduce 1.2 N·m through Kd.
Log the instantaneous target separately from the endpoint. The same command
filter must later exist in the hardware control path. A single 1 Mbit/s CAN bus
cannot carry 18 motor commands at 800 Hz once frame overhead is included;
local interpolation or multiple motor buses are needed for that implementation.

Changing gains simultaneously would confound this test. Lowering damping alone
can hide a demand spike while removing damping. The earlier ideal closed-CAD
inertia analysis puts its limiting stance mode near critical damping with the
current gains; it does not model actual contact transitions. Neither interpolation
nor gain tuning substitutes for running the unchanged full qualification again.

## Reproduction and provenance

[summary.json](summary.json) contains verified global/segment metrics, named peak
states, readbacks and all 18 per-environment group responses.
[contact_geometry.json](contact_geometry.json) preserves every support-loss pose
and the peak-gap contact geometry. [summarize.py](summarize.py) reproduces the
compact result and matches the validator's control-boundary response sampling.
[SHA256SUMS](SHA256SUMS) pins these local artifacts and the analysis helpers.

Original report and eight NPZ traces remain on Spark at
`/home/orionh/HEXAPOD_runs/mkii_fourbar_physical_mimic_v1/diagnostics/hexapod-fourbar-diagnose-20260905T174632Z-36232e95/`.
Full analysis, including ±12-sample peak windows, remains under
`/home/orionh/HEXAPOD_runs/mkii_fourbar_physical_mimic_v1/analysis/v5_groups128_20260905T174632Z/`.
Only CPU NumPy analysis ran here; no GPU job, model, gate or frozen source changed.

All eight trace SHA-256 hashes, shapes, named columns, finite values and contiguous
sample ranges were verified by `../analyze.py`. `../inspect_contact_geometry.py`
also verifies the URDF hash against the primary report. Run those two helpers
with the original report/adjacent NPZ set, then:

```sh
python3 summarize.py /path/to/report.json /path/to/analysis.json \
  /path/to/contact_geometry.json /new/path/summary.json
```
