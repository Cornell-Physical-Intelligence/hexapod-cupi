# Successor USD for the reviewed joint travel

This directory contains a new preparation lineage. The earlier USD bundles and frozen `usd_001` evidence are unchanged. Status remains **offline prepared; native SDF cooking and physics pending**.

The requested pitch travel is encoded directly relative to the existing viewer pitch zeros: femur **[-120°,80°]**, tibia **[-5°,+180°]**. Yaw uses the per-leg standoff-clearance intervals and midpoint zeros supplied by the joint review. These scalar travel bounds are not a certificate that every combined pose avoids body, leg or ground collisions.

`prepare_updated_usd.py` retains the previous authoring behavior and adds one successor-only audit precision repair: after widening a stored float quaternion to double, normalize it before passing it to `Gf.Rotation`. No bounds, geometry or numeric tolerance is relaxed. The earlier script and outputs remain unchanged. The full-source SDF selection, mesh data, mass-lineage separation, floating-base model and absence of invented actuator gains remain unchanged. `audit_joint_review_usd.py` adds explicit user-limit checks, positive 180-degree preservation, midpoint-centered yaw bounds, actual joint body relationship checks and 45 endpoint/FK comparisons. Those samples are zero, all 36 individual endpoints, six tibia 179.999-degree states and two combined endpoint states. They test coordinate math only. `audit_cad_pose_preservation.py` independently checks that each model's recorded CAD coordinates reproduce all original part placements and world COM/full tensors after the yaw-coordinate rebase.

USD angular limits are stored as scalar degrees. They must not be normalized through Euler/quaternion principal-angle ranges: **+180° stays +180°** in the upper-limit attribute. PhysX reduced-coordinate revolute positions wrap at ±2π; the requested intervals are within that documented range. [OpenUSD angular units](https://openusd.org/release/api/usd_physics_page_front.html), [PhysX articulation coordinates](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html).

The wide-angle auditor has four independent regression checks using a temporary, clearly labeled predecessor fixture: correct wide limits pass, +180→−180 corruption fails, an incorrect USD joint body relationship fails, and the new -0.515-degree yaw frame passes despite normal float quaternion storage roundoff. The fixture is not a successor asset and is not retained as one.

Reproduction uses a new output directory for each raw/corrected model:

```sh
python prepare_updated_usd.py prepare \
  --source /path/to/joint_review_bundle \
  --model /path/to/joint_review_bundle/model_rs05_mass_corrected.json \
  --urdf /path/to/joint_review_bundle/urdf/hexapod_updated_rs05_mass_corrected.urdf \
  --output /path/to/new_corrected_usd

python prepare_updated_usd.py audit /path/to/new_corrected_usd
python audit_joint_review_usd.py /path/to/new_corrected_usd --output /path/to/new_wide_limit_audit.json
python audit_cad_pose_preservation.py /path/to/old_model.json /path/to/new_model.json --output /path/to/new_cad_pose_check.json
python test_prepared_usd.py /path/to/new_corrected_usd
```

For raw CAD mass properties, omit `--model` and select `hexapod_updated_rs05_raw_cad.urdf`; the default is the source directory's `model.json`. The raw and nominal RS05 mass-corrected variants must remain separate. The latter's added cylinder masses/inertias are provisional engineering estimates, not exported Onshape inertias.

## Verified successor output

The final assets are `raw_cad/robot.usda` and `rs05_mass_corrected/robot.usda`. Both pass the saved-USD audit with **19 bodies, 18 revolute joints, 1,753 visual instances, 153 selected SDF colliders and 59 source meshes**. Both use exactly the same byte-preserved geometry library as their predecessors. Raw mass is 5.147603654kg; corrected mass is 7.466088235kg. The mass-corrected additions remain provisional engineering estimates.

| Yaw joint | New local lower/upper | Absolute midpoint zero |
|---|---:|---:|
| LF | [-74.295°,74.295°] | -50.715° |
| LM | [-46.715°,46.715°] | -0.515° |
| LR | [-74.750°,74.750°] | 50.140° |
| RF | [-74.750°,74.750°] | -129.860° |
| RM | [-46.715°,46.715°] | 179.485° |
| RR | [-74.295°,74.295°] | 129.285° |

The absolute azimuth convention is body +X=0°, body +Y=90° and counterclockwise about +Z; the local limits use the new midpoint zero. The USD float limits agree with their source URDF within the existing 1e-5-degree tolerance. All six femur limits are [-120°,80°], and all six tibia limits are [-5°,+180°]. Neither the 180-degree bound nor the 200-degree femur span is wrapped or shortened.

`raw_joint_review_audit.json` and `corrected_joint_review_audit.json` each pass all 45 kinematic samples. `raw_cad_pose_preservation.json` and `corrected_cad_pose_preservation.json` independently confirm the unchanged original CAD pose: all 1,753 part positions agree within 1.11e-16m, all 19 world COMs within 8.33e-17m, and world tensor components within 4.88e-19kg·m². The saved USD full-tensor error is at most 1.844e-9kg·m²; zero-pose joint frames agree within 3.984e-8 in their homogeneous matrices. Source hashes and all generated bundle manifest hashes are verified in `verification_summary.json`.

The new zero-pose visual bounds are X[-0.261868501,0.261817619], Y[-0.329891083,0.330084498], Z[0.005,0.158482367] metres. Root height is 0.08161109101311204m. The six foot surfaces start approximately 5mm above the nominal ground plane in that pose. This placement does not demonstrate stable standing.

## Preserved audit failure and exact repair

`failed_attempt_001/` retains both initial failed reports and USD bundles. The new LM yaw zero exposed a near-identity quaternion precision issue in the old CPU comparison. Its stored float quaternion becomes a double quaternion of norm 1.0000000258181416. Passing that directly to Gf's angle representation yields 0.5143413° instead of 0.515°, creating an apparent 1.149609e-5 frame discrepancy. Normalizing in double before the rotation conversion reduces it to 2.00633e-10. The new regression explicitly exercises that case; all four wide-angle tests and five prior preparation tests pass.

Crucially, each final `robot.usda` is **byte-identical to its preserved initial attempt**. The repair changes only the CPU audit's interpretation of normal float storage roundoff. Geometry, joint frames, all requested limits, inertias and the original tolerances are unchanged. No physics acceptance gate was weakened.

Integrate the final raw/corrected directories, the CLI and added audit scripts/tests, their JSON reports and logs, `verification_summary.json`, this README, and the preserved failure record. The full temporary output, including the two initial attempts, remains below 100MB. No repository file, predecessor artifact or GPU state was modified. Native cooking, actuator/contact calibration and coupled-pose collision qualification remain pending.
