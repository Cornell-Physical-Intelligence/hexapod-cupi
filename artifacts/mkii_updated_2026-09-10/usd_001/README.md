# Updated direct-drive robot: offline USD preparation

**Status: offline prepared; native collision cooking and physics validation pending.** The user confirmed that this design has no four-bar. Both saved USD assets are floating-base, 19-body / 18-revolute-joint candidates. Neither is an admitted Isaac asset or a qualified simulation-to-real model.

## Selected deliverables

- `rs05_mass_corrected/robot.usda`: prepared nominal RS05 mass-corrected candidate; total mass **7.466088235 kg**. Its 18 provisional cylinder mass/inertia additions bring each modeled actuator to the repository RS05 nominal mass of 191g. These additions are engineering estimates and are **not exported Onshape inertia**.
- `raw_cad/robot.usda`: separate uncorrected CAD-property candidate; total mass **5.147603654 kg**. Raw properties are preserved exactly up to USD float storage precision and are not relabeled as corrected.
- Each directory contains its own `geometry.usdc`, `geometry_manifest.json`, `collision_recipe.json`, `offline_audit.json`, `preparation_manifest.json`, and byte-preserved source model, URDF and mesh-map snapshots. USD references are relative and resolve entirely within that directory. The recorded URDF snapshot is provenance, not a second standalone URDF export: its original STL dependencies live in the separately supplied source asset.
- `prepare_updated_usd.py`: portable preparation and independent saved-USD audit CLI, using NumPy and standard OpenUSD only. No Isaac importer, native PhysX engine or additional dependency is needed for these offline steps.
- `test_prepared_usd.py` and `regression_tests.log`: five passing checks, including deliberate inverse-principal-axis, displaced-joint-pivot, missing-collider and wrong-visual-mesh defects.
- `overlap_review_parts.json`: unchanged copy of the independent source overlap review. Its 72 femur-screw/tibia pairs remain unresolved mechanical/ownership evidence, even though fasteners and adjacent-link contact are absent from this environment-collision recipe.

Integrate the two named final directories plus the CLI, regression test/log, overlap review and this README. `candidate_001/` and `candidate_002/` are earlier development outputs and are not the selected artifacts.

## Geometry, frames and inertia

All **1,753 source visual instances** reference **59 shared meshes**, with exact STL float vertex positions, triangle connectivity and source face normals. Nothing is decimated or replaced by primitive visuals. Source STL SHA-256 and canonical geometry array hashes are recorded; the independent audit checks the saved library and every composed visual/collider against them.

The articulation root is `/Robot/body`. Bodies are flat siblings below `/Robot`, avoiding nested rigid-body transforms. Eighteen `PhysicsRevoluteJoint` prims connect the named bodies; every axis uses joint-local +Z. USD local joint frames are constructed from the candidate's joint transform, with the child frame at its link origin. The audit independently recomputes forward kinematics from the saved URDF's fixed-axis RPY values, validates both joint frames in world space and checks limits converted from radians to USD degrees. Joint bounds remain provisional engineering limits inherited from the builder; this conversion does not qualify physical hard stops.

Each body carries explicit mass, COM, principal moments and principal axes. NumPy eigenvectors are columns; the authoring code transposes their rotation for Gf's row-vector matrix convention. The independent check reconstructs the full tensor using `Gf.Rotation.TransformDir`. Across both candidates the maximum full-tensor component error is **1.844e-9 kg·m²**; the largest joint-frame matrix error is **6.848e-7**, consistent with float joint quaternion storage. The inverse-quaternion defect is specifically rejected by regression testing. No collision volume is used to regenerate mass properties.

The root transform places the body origin **81.611091 mm** above Z=0. All six tibia surfaces have minimum Z of approximately **5.000–5.001 mm** in the supplied zero-coordinate inspection stance. The complete visual bounds are X[-0.261584367,0.261533445], Y[-0.343314906,0.343508771], Z[0.005,0.158482367] metres. This is a clear geometric starting placement, not a demonstrated stable standing controller. No fixed world joint, ground, scene, controller or lighting is embedded in the robot asset.

## Explicit collision recipe

**153 collision instances** are selected by an exact 17-mesh allowlist in the CLI and listed individually by mesh name, source name and occurrence count in `collision_recipe.json`. These cover the chassis/enclosures, structural plates/spacers, exposed motor housings/end pieces, bearing holders, output flanges and complete tibias. The other **1,600 source instances** retain their visuals and assigned mass/inertia but are omitted from environment collision: fasteners, bearings, rotor/gear pieces and other small internal actuator details. This deliberately omits exposed fastener snag/contact behavior; it is not a claim that the omitted parts are physically absent.

Each selected mesh has `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI` with approximation `sdf`, and the official `PhysxSDFMeshCollisionAPI` recipe. The complete tibia includes the actual rounded, nonspherical foot cap. There is no duplicate sphere or overlapping cap collider. Native SDF preserves openings in principle; actual resolution, contact shape and GPU cooking must be checked in Isaac.

Recipes request sparse subgrid resolution 6, 32-bit subgrid pixels, margin 0.01 and narrow-band thickness 0.01 relative to mesh diagonal, with remeshing disabled and triangle reduction factor 1.0. Source surfaces stay unchanged. Resolution varies by part; the tibia requests 1024, giving about **0.150 mm grid spacing**. Measured 1.6/2.032mm structural plates request enough samples for at least four cells through their measured plate thickness. Other small feature widths are not automatically qualified by that test. The large enclosures request 900, and the 1.6mm chassis plate requests 900 (about 0.393mm spacing). These are requested sampling scales, **not certified surface error bounds or cooked GPU memory measurements**. [Official SDF schema](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html), [NVIDIA thin-wall guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/guides/collision_guide.html).

Rest offset is 0 and contact offset is 1mm, an explicit simulation preparation choice. No physical contact material is assigned: friction, restitution, coating/compliance and contact stability still need identification. Self-collision is enabled; adjacent body contacts are filtered by articulation/joint behavior, while nonadjacent collisions remain enabled. Collision omission or adjacent filtering must not be used to dismiss the recorded CAD screw interference. [NVIDIA articulation collision behavior](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/articulation_stability_guide.html).

The standalone Python environment does not contain `PhysxSchema`. Official schema names and property types are therefore authored explicitly as USD schema tokens/attributes, and their presence is verified offline. Only the pinned Isaac runtime can validate recognition, cook SDFs and exercise their behavior. There is **no unsupported plain dynamic triangle-mesh fallback**. [NVIDIA collider compatibility](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html).

## Actuator boundary

The repository RS05 baseline appears as 5.5N·m effort metadata and 50.265482rad/s nominal joint speed. `physxJoint:maxJointVelocity` is authored as 2880degrees/s using the documented angular units. **The effort metadata is not an applied torque cap.** No `DriveAPI`, stiffness, damping, gains, friction or armature is invented; the asset is passive until a calibrated actuator runtime is supplied. That runtime must enforce torque-speed, continuous duty, current/thermal/voltage limits and protections. Peak torque and no-load speed are not simultaneously available operating points. [PhysX joint velocity units](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_joint_a_p_i.html).

## Reproduction and standalone verification

Use a new output directory for each preparation; existing results are never overwritten.

```sh
python prepare_updated_usd.py prepare \
  --source /path/to/updated_urdf_bundle \
  --model /path/to/updated_urdf_bundle/model_rs05_mass_corrected.json \
  --urdf /path/to/updated_urdf_bundle/urdf/hexapod_updated_rs05_mass_corrected.urdf \
  --output /path/to/new_corrected_usd_bundle

python prepare_updated_usd.py prepare \
  --source /path/to/updated_urdf_bundle \
  --urdf /path/to/updated_urdf_bundle/urdf/hexapod_updated_rs05_raw_cad.urdf \
  --output /path/to/new_raw_usd_bundle

python prepare_updated_usd.py audit /path/to/new_corrected_usd_bundle
python test_prepared_usd.py /path/to/new_corrected_usd_bundle
```

Before native admission: resolve the mechanical/ownership overlap, obtain accepted actuator and contact parameters, cook in the pinned Isaac runtime under the guarded allocation, inspect actual cooked geometry/schema warnings, validate reset/joint sweeps/contact forces and confirm no invalid states. Only then consider bounded physics and training admission. No GPU or Spark action occurred in this preparation.
