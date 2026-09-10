# Reviewed joint travel and coxa neutral successor

Revision: `20260910_user_travel_coxa_midpoint_v2`. All **1753 parts, 19 bodies, 18 joints**, mesh bytes and exact link mass/COM/inertia records are preserved. The prior evidence bundle remains frozen.

## User-specified pitch travel

Femur travel is **−120° to +80°** and tibia travel is **−5° to +180°**, both relative to the **unchanged previous viewer pitch zero**. No pitch origin, CAD coordinate, motor sign or link frame was changed. These are user-confirmed travel limits, **not a collision-free envelope**; some combinations intersect other robot parts or the ground.

## Coxa coordinate rebase

Each coxa zero is the midpoint of its buffered intrinsic coxa-versus-standoff interval. Endpoints use the independent geometry review’s **0.5 mm separation buffer**. This check excludes femur/tibia geometry and does not certify full-robot motion. Body azimuth uses +X(left)=0°, +Y=90°, forward(−Y)=−90° and positive counterclockwise about +Z. Absolute endpoints below form continuous, unwrapped intervals.

| Joint | Previous-zero shift (°) | Absolute neutral (°) | Absolute bounds (°) | New offsets (°) |
|---|---:|---:|---:|---:|
| lf_coxa_yaw | 17.483552 | -50.715000 | -125.010000 to 23.580000 | -74.295000 to 74.295000 |
| lm_coxa_yaw | -0.515039 | -0.515000 | -47.230000 to 46.200000 | -46.715000 to 46.715000 |
| lr_coxa_yaw | -18.058547 | 50.140000 | -24.610000 to 124.890000 | -74.750000 to 74.750000 |
| rf_coxa_yaw | -18.058637 | -129.860000 | -204.610000 to -55.110000 | -74.750000 to 74.750000 |
| rm_coxa_yaw | -0.514966 | 179.485000 | 132.770000 to 226.200000 | -46.715000 to 46.715000 |
| rr_coxa_yaw | 17.483627 | 129.285000 | 54.990000 to 203.580000 | -74.295000 to 74.295000 |

For each yaw joint, `J_new = J_old Rz(shift)` and `q_CAD,new = q_CAD,old − shift`; reported yaw limits subtract that same shift. Link frames remain the original CAD anchor frames, so part placements and link inertias remain unchanged. Hardware encoder zeros remain uncalibrated.

## Verification

Raw-model maximum CAD part-transform change: **4.44e-16**. All serialized URDF variants independently reconstruct every part below 1e−12 maximum matrix-entry change. Both raw and nominal-motor-mass-corrected aggregate mass, COM and full inertia remain invariant at the source CAD pose.

The revised default root height is **0.081611091013 m**, calculated from the actual tibia mesh extrema. The lowest default part has **5.000000 mm** ground clearance. This is a default-pose geometry check, not planted-foot support or a wide-travel contact test.

**The previous 133-pose collision result is historical and applies only to the earlier narrow inspection envelope.** It is not reused as validation of this wider user travel. New dynamics preparation must use this revision and retain its distinct checks. No Isaac admission or physical-stop calibration is claimed.

## Reproduce

```sh
python apply_joint_review.py --source-model-dir /path/to/previous/model --joint-review /path/to/joint_review.json --out /path/to/successor
```

`joint_review.json` records provenance; `joint_review_report.json` records source hashes, all coordinate changes and invariance checks. `previous_*` files preserve historical inputs and must not be read as current travel qualification.
