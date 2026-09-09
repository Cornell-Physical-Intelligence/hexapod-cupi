# Recorded policy motion analysis

This tool computes descriptive motion metrics from a completed, hash-verified policy capture. It checks the original report, state archive and metadata, excludes transitions containing automatic episode resets, and records all input hashes. It does not modify a checkpoint, run Isaac Sim or grant walking/terrain/navigation/hardware admission.

It estimates plate-origin velocity from recorded position differences and relative body rotation over each control interval, using the robot's anatomical convention: forward is -body Y, left is +body X, yaw is about +body Z. These finite-interval plate-motion estimates are not the instantaneous COM velocities used by the training reward. The exact installed SDK alias evidence is retained alongside this script.

Outputs include per-axis error, command and measured velocity, actual world-horizontal path length, commanded distance, reset counts, tilt, height and near-stationary occupancy during nonzero translation commands. The 0.02 m/s moving-command and 0.01 m/s near-stationary thresholds label descriptive statistics only; no success threshold has been established. A high reward or a passing finite-inference check does not demonstrate actual locomotion.

Usage after a real capture:

```sh
python analyze_rollout.py /absolute/path/to/completed/capture --output /absolute/path/to/motion_analysis.json
```

Six CPU tests pass: forward/left mapping, rotated and tilted body frames, positive local yaw, quaternion sign invariance, reset-teleport exclusion, stationary behavior, invalid data, and CLI provenance/hash tamper rejection. Independent review confirmed the calculations and added the explicit plate/COM distinction. No real learned-policy capture was available at preparation time; this is tested analysis tooling, not a policy result.
