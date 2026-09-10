# Reviewed joint travel and coxa midpoint revision

This immutable successor records revision `20260910_user_travel_coxa_midpoint_v2`. The user confirmed femur travel **−120° to +80°** and tibia travel **−5° to +180°** relative to the previous viewer's pitch zero. Both pitch zeros are unchanged. These are joint travel limits; full-robot collision-free combinations have not been established. The earlier 133-pose result in `import_001` applies only to its historical narrow inspection envelope.

Each coxa zero moves to the midpoint of its independently checked standoff opening. Published absolute endpoints were rounded inward to 0.01°. The attached continuous certificate establishes at least **0.5 mm** separation over each full published interval between that coxa group's triangles and all six standoff plates. Its scope excludes femur/tibia and other body surfaces; an existing coxa-screw to enclosure gap is only about 0.150 mm. It does not establish measured hardware stops.

| Leg | Absolute coxa bounds (°) | New absolute zero (°) | Relative bounds (°) |
|---|---:|---:|---:|
| LF | −125.01 to +23.58 | −50.715 | ±74.295 |
| LM | −47.23 to +46.20 | −0.515 | ±46.715 |
| LR | −24.61 to +124.89 | +50.140 | ±74.750 |
| RF | −204.61 to −55.11 | −129.860 | ±74.750 |
| RM | +132.77 to +226.20 | +179.485 | ±46.715 |
| RR | +54.99 to +203.58 | +129.285 | ±74.295 |

Absolute azimuth is counterclockwise about body +Z, with +X (left) = 0°, +Y = 90°, and forward (−Y) = −90°. Bounds are continuous, unwrapped intervals. `J_new = J_old Rz(shift)` and `q_CAD,new = q_CAD,old − shift` preserve the source CAD pose exactly. Hardware encoder zeros remain uncalibrated.

## Preserved geometry and dynamics

The original Onshape export is a single rigid link with zero joints. Its articulated interpretation is the user-confirmed direct-drive mechanism: **19 bodies, 18 revolute joints, 1,753 individual mesh instances, and 59 unique meshes**. Body membership remains geometrically inferred. This successor preserves every source part placement, mesh byte, link mass, link COM, and full inertia tensor; only yaw coordinates and user-confirmed joint bounds change. The arbitrary original CAD pose remains available in `cad_pose.json`.

The raw total remains **5.147603654203134 kg**. The separate nominal RS05 mass variant remains **7.466088235225788 kg**, retaining the same 18 missing-mass additions and provisional cylinder allocation from the initial audit. No material density or mass correction was introduced by this revision. `rs05_mass_correction.json` identifies the current raw assets and labels the original calculation hashes separately.

All 1,753 CAD placements and both variants' aggregate mass, COM, and full inertia pass invariance checks. The largest raw-model transform change is 4.44e−16; all three serialized URDFs reconstruct every source placement within 1e−12 maximum matrix-entry error. The recomputed default root height is 0.08161109101311204 m, placing the lowest mesh vertex 5 mm above the ground. Source screw penetrations and wide-travel self-intersections remain distinct physics preparation issues. This evidence grants no Isaac admission or hardware qualification.

## Contents and identities

- `baseline/`: exact preceding model, URDF, CAD-pose, mapping, and dynamics-report inputs, needed because the active robot package now contains the successor. No meshes are duplicated.
- `compose_joint_review.py`, `apply_joint_review.py`, and `test_apply_joint_review.py`: reproducible input composition, migration, and numerical regression tests.
- `joint_review.json`, `BUILD_REPORT.md`, `build_report.json`, and `part_*`: current review provenance, checks, and per-part errors/default clearances.
- `current_output_hashes.json`: exact identities of both current models, three URDFs, and their companions in `robot/hexapod_mkii_updated_v1/`. All nine package files matched at integration.
- `yaw_published_joint_bounds.json`, `yaw_geometry_limits.json`, `yaw_geometry_certified.json`, and `continuous_clearance_certificate.json`: stable endpoint input, exact historical geometry input, its strengthened descendant, and the continuous certificate. Independent yaw algorithm evidence is maintained by the geometry worker as a sibling bundle.
- `source_copy_receipt.json`: original paths, sizes, and hashes of every copied source file. Absolute scratch paths inside frozen reports describe the actual run and are not portable dependencies.
- `revision_tests.log` and `copied_revision_tests.log`: four passing tests, including unchanged pitch zero, nonzero yaw rebase/CAD invariance, wide user limits, and unwrapped intervals beyond 180°.
- `failed_attempts/001_enospc/`: the first final build's disk-full failure and partial output. Recovery replaced only this worker's verified identical prototype mesh copies with a symlink to the unchanged original scratch meshes. Failed evidence was preserved.
- `SHA256SUMS`: finalized hashes for this bundle, excluding the manifest itself. `import_001` remains unchanged.

The composer consumed the certified yaw report before a later `component_containment_check` field was added. The exact precursor bytes were recovered and match recorded SHA-256 `1d840edcbc9a74ee9768638900f21b2b71a2d7be8230fbda05fb0e4abad617c8`; they are preserved as `yaw_geometry_limits.json`. The strengthened descendant is `yaw_geometry_certified.json`. Published endpoints are identical. `yaw_provenance_receipt.json` records both identities and the recovery method; no final model or URDF bytes were changed to repair this provenance.

## Reproduction

From the repository root, use a Python environment with the pinned dependencies recorded in `../import_001/reproduction_requirements.txt`. The recorded run used Python 3.12.12, numpy 2.5.2, scipy 1.18.1, and trimesh 5.1.0. Set `HEXAPOD_REVIEW_PYTHON` to that interpreter. All writes below go to a fresh scratch directory; the existing mesh directory is shared read-only in practice and its bytes are checked against `mesh_mapping.json` before use.

```sh
HEXAPOD_REVIEW_REPO="$PWD"
HEXAPOD_REVIEW_EVIDENCE="$HEXAPOD_REVIEW_REPO/artifacts/mkii_updated_2026-09-10/joint_review_002"
HEXAPOD_REVIEW_SCRATCH="$(mktemp -d /tmp/hexapod-joint-review-reproduction.XXXXXX)"
cp -R "$HEXAPOD_REVIEW_EVIDENCE/baseline" "$HEXAPOD_REVIEW_SCRATCH/baseline"
ln -s "$HEXAPOD_REVIEW_REPO/robot/hexapod_mkii_updated_v1/meshes" "$HEXAPOD_REVIEW_SCRATCH/baseline/meshes"
mkdir "$HEXAPOD_REVIEW_SCRATCH/output"
ln -s "$HEXAPOD_REVIEW_REPO/robot/hexapod_mkii_updated_v1/meshes" "$HEXAPOD_REVIEW_SCRATCH/output/meshes"
"$HEXAPOD_REVIEW_PYTHON" -B "$HEXAPOD_REVIEW_EVIDENCE/compose_joint_review.py" \
  --yaw-review "$HEXAPOD_REVIEW_EVIDENCE/yaw_published_joint_bounds.json" \
  --yaw-geometry "$HEXAPOD_REVIEW_EVIDENCE/yaw_geometry_limits.json" \
  --source-model "$HEXAPOD_REVIEW_SCRATCH/baseline/model.json" \
  --revision 20260910_user_travel_coxa_midpoint_v2 \
  --out "$HEXAPOD_REVIEW_SCRATCH/joint_review.json"
"$HEXAPOD_REVIEW_PYTHON" -B "$HEXAPOD_REVIEW_EVIDENCE/apply_joint_review.py" \
  --source-model-dir "$HEXAPOD_REVIEW_SCRATCH/baseline" \
  --joint-review "$HEXAPOD_REVIEW_SCRATCH/joint_review.json" \
  --out "$HEXAPOD_REVIEW_SCRATCH/output"
"$HEXAPOD_REVIEW_PYTHON" -B -m unittest discover \
  -s "$HEXAPOD_REVIEW_EVIDENCE" -p test_apply_joint_review.py -v
```

The input JSON, model, CAD pose, and serialized URDF bytes should match their recorded identities with the recorded dependency versions. A newly produced report records the new `source_model_dir`, so its complete bytes will differ while its numerical checks remain comparable. The root integrator owns associated Markdown, the required `docs/PROJECT_SITE.md` update/check/build, and publication; this bounded evidence integration performs no Git operation.
