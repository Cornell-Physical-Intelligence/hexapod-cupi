# Updated Onshape export: contact and collider audit

**Disposition:** The final USD candidate uses the complete source tibia SDF, not these optional cap hulls. This bundle preserves the earlier geometric comparison and its selected four-cap alternative.

This is an offline geometric audit of the export from `HexapodLegUpdatedV2.zip` (source ZIP SHA-256 `31f33044285c3de2dbd85180e7c4886c31d80da4270dfd6b94bbd50d150a4cf7`). It does not qualify a policy, prove hardware joint limits, or claim successful Isaac cooking. Visual meshes and mass/inertia values remain separate source data. No source mesh was edited.

## Deliverables and reproduction

- `audit_collisions.py`: deterministic generation and numerical audit using the existing numpy/scipy/trimesh/rtree environment; no added dependency.
- `collision_audit.json`: source mesh hashes, geometry measurements, rejected sphere fits, cap mesh hashes and errors, plate recommendations and official PhysX authoring sources.
- `meshes/tibia_foot_cap_*.stl`: four convex distal cap components, in the original `tibia.stl` part frame and metres. Each has 124 vertices and 244 triangles. These are **contact-only components, not a complete tibia collision model**.
- `orthographic.png`: actual source-mesh orthographic views; a geometry illustration, not an experimental robot image.

```sh
/Users/andreboufama/Documents/CUPI/HEXAPOD/.venv/bin/python audit_collisions.py SOURCE_EXPORT OUTPUT_DIRECTORY
```

The script clips the actual surface triangles at local X=115mm and Y/Z=0, includes the solid partition-axis endpoints, forms four quadrant hulls and greedily selects source hull vertices. It then compares support distances to the entire original tibia in 6,000 directions within 60° of the footward +X direction. A second check measures signed distance of candidate surface triangle centroids/edge midpoints against the complete source mesh. These are sampled geometric checks, not a continuous error proof.

## Foot geometry: no single sphere is faithful

The tibia mesh origin is at the independently inferred knee axis, local Y up to sign. Local +X extends toward the foot. Its extreme X is 129.999995mm. The final contact face is a **2.5mm square**, with Y/Z bounds ±1.25mm; the rounded cap starts at X=115mm. The shaft's Y/Z outer bounds are ±16.25mm at X=60–110mm. This is a rounded CAD cap and shaft, not an exact sphere. A convenient endpoint frame at `[0.130, 0, 0]` is a contact-reference location, not a sphere center or the ground clearance of an arbitrary pose.

| Candidate | Maximum support overfill | Maximum support undercoverage | Comment |
|---|---:|---:|---|
| Radius 16.25mm sphere at X=113.75mm | 0mm | 2.316mm | Matches distal extreme, misses other contact directions |
| Least-squares distal sphere | 1.786mm | 0.730mm | Up to 2.961mm radial fit residual |
| Four cap hulls, 124 vertices each | <0.000001mm | 0.08191mm | 0.02745mm support RMS; suitable offline candidate |

The cap hulls' 3,268 exterior face samples also show localized deviations of **0.159mm outside** and **0.334mm inside** the original triangulated surface. Contact-height support accuracy alone must not be presented as full-surface accuracy. No compliance, coating thickness, measured friction or soft-foot model can be inferred from these meshes.

An exploratory 16-part axial subdivision did not reduce the localized surface discrepancy enough to justify four times as many hulls; its scratch-only trial was not selected for this evidence bundle. That exploratory output is not the selected candidate and used a 220-vertex budget that does not itself guarantee the PhysX face limit. Only the four selected comparison meshes and main audit are packaged here; they are not the final USD recipe.

## Structural concavities must stay open

| Part | Whole-hull volume / CAD solid volume | Important exposed geometry |
|---|---:|---|
| Tibia | 4.35× | 56.5mm gap between fork cheeks; hollow shaft |
| Femur first stage | 10.61× | U bracket with 56.5mm inner gap, 2.032mm side plates |
| Coxa bottom plate | 16.86× | U bracket with 49mm inner gap, 1.6mm plates and bearing holes |
| Tibia attachment plate | 14.28× | U bracket with 49mm inner gap, 2.032mm plates and bearing holes |

A single convex hull of one of these parts fills large empty spaces. Use separate base/side-wall components and preserve exposed bearing holes with ring sectors or a qualified decomposition/SDF collider. The JSON records exact measured split ranges. Coxa side bearing holes have center X=-49mm/Z=26.1mm and radius 16mm. The tibia attachment holes have center X=0/Z=-28.502mm and radius 16mm. Coxa base bearing hole is centered X=Y=0 with the same radius. These numbers come from source-mesh cross sections; fitted circle RMS is approximately 7.6µm due to tessellation. Twenty-four ring sectors would incur a theoretical maximum circle-chord intrusion of 0.137mm before other approximation effects. Bolt holes need an explicit clearance decision if suppressed.

NVIDIA documents dynamic nonconvex mesh support through SDF or convex decomposition. A plain dynamic triangle-mesh collider is insufficient. SDF retains concavities, but thin walls need enough resolution: the documented spacing is longest extent/resolution, and shells thinner than roughly two grid spacings may disappear. The 1.6mm coxa wall therefore calls for spacing smaller than 0.8mm, with an additional accuracy margin established by cooking/measurement. [NVIDIA colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html), [SDF collision guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/guides/collision_guide.html).

## Isaac authoring and remaining admission

PhysX limits both convex vertices and polygon faces to 255. The selected 124-vertex/244-triangle cap pieces remain below both counts. Explicitly author `PhysxConvexHullCollisionAPI` with `hullVertexLimit=124`; its documented default 64 would change the imported shape. These counts are necessary, not sufficient for a cooking pass. Cook with the pinned Isaac version, inspect cooked geometry and warnings, and repeat contact/support tests on that cooked representation. [PhysX geometry](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html), [Convex-hull schema](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_physx_schema_physx_convex_hull_collision_a_p_i.html).

Do not use hull volumes to recompute CAD masses/inertias. Add the remaining fork, shaft, chassis and structural colliders; validate joint sweeps and reset clearance; establish collision filters from real rigid-body ownership. No Isaac or GPU job was launched in this audit.
