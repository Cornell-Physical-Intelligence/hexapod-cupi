# Detailed updated CAD: provisional articulated inspection

Includes **1753 exact mesh instances, 19 inferred rigid bodies and 18 revolute joints**. Raw CAD mass **5.147603654203 kg**; no motor-mass correction. This is a new direct-drive candidate, not the old physical four-bar or active simplified C-study.

The user confirmed that the new robot has no four-bar and uses the direct-drive topology. The input had one fused link and no joints. Geometry recovery supplies provisional axis lines and complete inferred ownership. Bearing races and vendor motor internals need confirmation. The preserved exported pose can be restored with `cad_pose.json`; it is not a recommended reset pose.

## Default inspection pose and bounds

q=0 points coxae radially outward, femurs 20 degrees above horizontal and tibiae vertically down (relative knee −110 degrees). Positive yaw rotates about export +Z; positive pitch raises the outward segment. Every child frame has joint axis +Z. Bounds are ±25 degrees yaw, ±20 degrees shoulder and −3 to +20 degrees knee around this default: **finite visual sweep envelopes, not measured stops or clearance-certified motion**.

Root height is **0.081611091 m**, placing the actual lowest tibia mesh vertex 5 mm above ground. Lowest vertex over every part has 5.000000 mm clearance. Motion away from the default can intersect the ground or other parts and requires the separate sweep report.

## Checks

- Complete unique ownership of all 1,753 parts; 19 connected acyclic bodies, 18 unique joint names.
- All source visual transforms reconstruct at their CAD joint values with maximum matrix-entry difference 5.55e-16; parsed URDF XML difference 7.49e-14.
- Summed per-link mass error 0 kg; maximum aggregate COM error 8.33e-17 m and inertia error 3.33e-16 kg·m².
- All 19 link tensors are positive definite and satisfy inertia triangle inequalities. Full off-diagonal terms retained.
- All 18 finite-difference axis sign checks pass; 256 deterministically sampled poses retain finite rigid transforms.

## Remaining qualification

Motor CAD represents about 62.2 g per device versus the historical RS05 191 g nominal. Housing/output inertia allocation, mechanical stops, encoder zeros, contact geometry, cable clearance and actuator dynamics are unverified. Original per-part collision meshes are preserved but explicitly unprepared. URDF effort 0 disables physical actuation for inspection; speed 0.5 rad/s is a placeholder. These values are separate from the frozen C-study and are not recovered CAD or qualified hardware limits. **No Isaac physics admission, training compatibility or sim-to-real readiness is claimed.**

## Reproduction

```sh
python build_updated_urdf.py --source /path/to/export --mass-ledger /path/to/part_instances.json --geometry /path/to/joint_geometry_recovery.json --ownership /path/to/provisional_body_ownership.json --out /path/to/output
```

`model.json` provides every part, link, joint, source-CAD coordinate and default root height. `mesh_mapping.json` maps ASCII asset names to unchanged source mesh bytes and hashes. `build_report.json` and `part_clearances_default.json` preserve detailed checks.
