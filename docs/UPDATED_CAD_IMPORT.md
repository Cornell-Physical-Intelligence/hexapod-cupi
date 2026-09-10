# Updated direct-drive CAD import

The user confirmed on 10 September 2026 that the updated robot has **three direct-drive joints per leg, with no external four-bar**. Its versioned asset is [hexapod_mkii_updated_v1](../robot/hexapod_mkii_updated_v1/README.md): 1,753 original part instances, 59 unchanged meshes, 19 rigid bodies and 18 revolute joints. The user has visually approved this model and selected its motor-weight-corrected URDF as ground truth for all future training. The [canonical selection](../robot/active_model.json) pins exact URDF/model/USD identities. The C-study policy and historical four-bar task retain their own assets, controllers and evidence for reproduction.

The supplied `HexapodLegUpdatedV2.zip` contains a single rigid link and **zero exported joints**. The preserved part cache includes individual masses, centers of mass, inertia tensors and assembly transforms. We recovered the axes from the actual motor/bearing geometry and inferred grouping from the six differently posed leg copies. The user confirmed the serial topology; individual fastener, bearing and rotor ownership still needs mechanical review. This distinction matters more to accuracy than the missing material names, which the user explicitly waived.

## Inspect the result

From the repository root, run:

```sh
python3 -m http.server 8347 --bind 127.0.0.1 --directory robot
```

Open [the local inspection viewer](http://127.0.0.1:8347/hexapod_mkii_updated_v1/preview/). Press **Play all 18 joints** for a complete review: each joint goes from zero to its lower endpoint, pauses, travels to the upper endpoint, pauses, and returns to zero. It takes about 3 minutes 19 seconds at 1×; 2× and 4× are available. Pause/resume or reset at any time. The moving group and its descendants are highlighted. Alternatively, choose a joint and use its slider or **Sweep**. Click a part to see its source ID, mass and assigned link. Link colors, X-ray, link isolation, individual-part separation and the **CAD overlap parts** filter expose the grouping. Preview bound edits can be saved as JSON; they do not modify the versioned URDF. Motion is kinematic and lifts the body 200 mm for clearance during the wide travel sweeps; it is not a simulated gait or support test.

The femur and tibia zeros remain exactly as in the original viewer: femur elevation +20° and relative knee angle −110°, placing the tibia downward. Following the user’s review, each coxa zero is centered in its own standoff-plate clearance interval instead of pointing radially away from the chassis center. Changing that starting pose does not move a bearing axis. **Export pose** restores all six original, different joint coordinates. Joint rotations follow local +Z; positive pitch raises the distal segment. The CAD body convention is +X left, −Y forward, +Z up, and remains explicit for any later runtime adapter.

| Coordinate | Joint travel about the displayed zero | Provenance |
|---|---:|---|
| Coxa yaw, each leg | Per-leg plate-clearance interval | Exact source geometry with 0.5 mm standoff-plate buffer |
| Femur pitch, every leg | **−120°…+80°** | User entered these values in the original viewer |
| Tibia pitch, every leg | **−5°…+180°** | User entered these values in the original viewer |

The user explicitly distinguishes joint travel from configuration-dependent collision avoidance. These independent bounds do not prevent the tibia, femur or neighboring legs from striking the body in a combined pose. The pitch zeros are unchanged; their absolute geometric angles are femur elevation −100°…+100° and knee relative to femur −115°…+70°. The viewer and saved settings identify the current model and its zero convention.

The yaw opening is asymmetric in the original radial coordinates. The new zero is its midpoint, which changes the corner starting directions by about 17.5–18.1° while preserving the bearing axes and every original CAD placement. Absolute azimuth is counterclockwise about +Z from body +X (left); forward is −90°. Intervals stay unwrapped across ±180°.

| Leg | Absolute lower / ° | New absolute zero / ° | Absolute upper / ° | Viewer yaw range / ° |
|---|---:|---:|---:|---:|
| LF | −125.01 | −50.715 | +23.58 | ±74.295 |
| LM | −47.23 | −0.515 | +46.20 | ±46.715 |
| LR | −24.61 | +50.140 | +124.89 | ±74.750 |
| RF | −204.61 | −129.860 | −55.11 | ±74.750 |
| RM | +132.77 | +179.485 | +226.20 | ±46.715 |
| RR | +54.99 | +129.285 | +203.58 | ±74.295 |

The [yaw evidence](../artifacts/mkii_updated_2026-09-10/yaw_envelope_002/README.md) includes a full-revolution search, exact triangle-distance refinement, limiting source part IDs and a continuous clearance proof. All 141 coxa parts per leg were tested against all six standoffs. Endpoints are rounded inward to 0.01°. Independently rerunning all 216 certificate midpoint queries with the integrated, rebased model confirms a weakest continuous lower bound of **0.500128 mm**. An existing coxa fastener-to-enclosure gap is about **0.150 mm**, so the 0.5 mm claim applies specifically to standoff plates.

The [joint revision](../artifacts/mkii_updated_2026-09-10/joint_review_002/README.md) preserves pitch zero, CAD transforms and both variants' masses, COMs and full tensors. The [successor USD audit](../artifacts/mkii_updated_2026-09-10/usd_002/README.md) passes 45 kinematic samples per variant: neutral, all 36 individual endpoints, six tibias at +179.999°, and two combined endpoint poses. These check coordinate consistency, not collision-free motion. Positive +180° is preserved as an unwrapped scalar limit.

The initial conservative envelope (yaw ±25°, femur ±20°, knee −3°…+20°) had passed a 133-pose geometric sample after a wider negative knee extension caused structural interference. That result remains [historical evidence](../artifacts/mkii_updated_2026-09-10/import_001/joint_validation/README.md). It is **not a collision-free certificate for the new user-supplied travel ranges**. The source meshes, CAD-pose reconstruction and inertial checks remain applicable; the new limit/zero contracts receive a separate audit.


## Weight and inertia

The raw CAD sum is **5.147603654203 kg**. Each of the 18 vendor motor sets contributes only **62.195301 g**, substantially below the established RS05 nominal **191 g**. A separate dynamics candidate adds the missing 128.804699 g per motor, yielding **7.466088235226 kg**. It uses a provisional solid cylinder at each actual housing center, Ø46 mm ×35.1 mm, while keeping all original part records unchanged. This is a sourced nominal-mass estimate, not a measured complete-robot weight or identified rotor inertia. See the [RS05 specification review](RS05_SPEC_REVIEW.md).

| Rigid group | Count | CAD mass each / kg | Nominal motor corrected each / kg |
|---|---:|---:|---:|
| Chassis | 1 | 1.908769948 | 1.908769948 |
| Coxa | 6 | 0.243634014 | 0.501243411 |
| Femur | 6 | 0.173673948 | 0.302478647 |
| Tibia | 6 | 0.122497656 | 0.122497656 |
| **Complete modeled assembly** | | **5.147603654** | **7.466088235** |

All 19 tensors in both variants are positive definite and satisfy the principal-moment triangle inequality. Each tensor is expressed about its own COM, in its declared link frame; full off-diagonal terms and parallel-axis contributions are retained. The [asset mass/inertia breakdown](../robot/hexapod_mkii_updated_v1/MASS_INERTIA.md) lists every link, and the [immutable mass ledger](../artifacts/mkii_updated_2026-09-10/import_001/mass_audit/evidence/MASS_MATERIAL_AUDIT.md) accounts for every source part. Missing batteries, electronics, cables, sensors or payload cannot be inferred from this CAD inventory; their actual inclusion and measured COM remain a hardware inventory task. Source appearance colors are preserved without inventing physical material identities.

## Geometry findings that need review

All 10,888,953 source mesh vertices across the 1,753 instances return to the supplied CAD pose within **0.738 µm**, consistent with the source XML's rounding. Independently fitted motor/bearing axes agree within **0.874 µm** line separation and **0.001744°** unsigned direction. These quantify agreement within the exported CAD, not manufacturing accuracy. The new joint-center lengths are about 49.000 mm yaw-to-shoulder and 73.502 mm shoulder-to-knee; the tibia's distal surface is 130.000 mm along its own +X from the knee. Its rounded-square tip is **not a sphere**.

There are 72 existing tibia/M3-screw surface overlap pairs, 12 per leg. Four representative right-middle screw penetrations are approximately **0.286–0.299 mm in the original supplied CAD pose**. They were not introduced by the new neutral stance. Those screws follow femur transforms across the six copies; the tibia moves independently. Review both attachment side and clearance. The viewer filter shows the six tibias and 72 screws, with exact source IDs. Another 54 baseline cross-body pairs concern motor/bearing internals. Preserving all these observations prevents collision simplification from hiding an unresolved source issue.

## Isaac preparation and remaining admission work

The asset provides three URDFs: exact raw CAD with actuation disabled for inspection; raw CAD with sourced RS05 peak fields; and the separate nominal-motor-mass candidate. Use the latter for subsequent dynamics preparation, retaining its provisional mass-distribution label. All use the same source geometry, reviewed zero conventions and user/geometry limit provenance. Peak 5.5 N·m and no-load 50.265482 rad/s describe different envelope endpoints; a later actuator runtime must enforce actual torque-speed, duty and thermal behavior.

The [portable successor USD preparation](../artifacts/mkii_updated_2026-09-10/usd_002/README.md) preserves all visual geometry and inertias while selecting exposed structure for concave SDF collisions. Its audit reads the saved stage back against the URDF, checking units, body/joint frames, COM, principal-axis inertia reconstruction and mesh hashes. Plain whole-part convex hulls would fill the hollow tibia and bracket cavities, so they are not substituted. Internal hardware and fasteners remain in rendering and mass but are explicitly omitted from environment collision, with a complete inventory; exposed fastener snagging is consequently not modeled.

**This is prepared for native Isaac processing, not physically admitted.** SDF cooking and its GPU memory/contact behavior still require Isaac Sim 6.0.1. Imported-asset inspection must precede suspended joint sweeps, gravity/contact settling, solver/effort diagnostics and a separate model-bound admission report. No drive gains, armature, friction, damping or thermal envelope are guessed. Review the flagged CAD ownership/clearance issue, identify actual mass distribution and encoder signs/stops, and add the measured sensor/payload inventory before sim-to-real claims. No Spark allocation or PPO launch is part of this offline intake. The user-supplied wider travel does not resolve the existing CAD overlap or identify controller collision avoidance.

## Onshape export conventions

For a future authoritative articulated export, keep each rigid group as an assembly instance, name the active revolute mates `dof_<joint_name>`, and assign `link_<name>` frames deliberately. Verify each mate's signed Z axis at its bearing. Keep dynamics retrieval enabled and limits enabled; `ignore_limits: false` and `no_dynamics: false` express that intent. A version-pinned CAD URL will make that export reproducible. The provided ZIP remains immutable and the recovered geometry stays traceable until an actual mate graph replaces the inference. These conventions follow the official [design guide](https://onshape-to-robot.readthedocs.io/en/latest/design.html) and [configuration guide](https://onshape-to-robot.readthedocs.io/en/latest/config.html).

The full [intake evidence bundle](../artifacts/mkii_updated_2026-09-10/import_001/README.md) preserves source hashes, numerical audits, the rejected first knee interval and the actual Claude Code Fable 5.1 maximum-reasoning review with root's disposition.

## Main model transition

The user requested this detailed direct-drive model as the main URDF after visual review of every joint. The [prepared selection](../artifacts/mkii_updated_2026-09-10/main_selection_001/README.md) pins the nominal RS05 mass-corrected URDF and model and replaces the generic mock-gait viewer entry point with this detailed inspector. The user approved the full-range animation and grouping, then explicitly selected the motor-weight-corrected URDF as ground truth for all training. The [activation receipt](../artifacts/mkii_updated_2026-09-10/main_selection_001/activation_receipt.json) records that authorization and exact identities. The canonical selector and generic viewer are active. Do not start further simplified-study or four-bar training; retain their identities solely for historical reproduction. A new physical runtime must handle per-leg yaw limits, the actual +X tibia geometry, the joint signs/zeros and the calibrated actuator contract before separate native admission. An existing 18-action checkpoint is not thereby compatible.
