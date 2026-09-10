# Updated MKII direct-drive asset v1

Detailed inspection and offline physics-preparation candidate from the user’s `HexapodLegUpdatedV2.zip`, received 10 September 2026. The user confirms **no external four-bar**: six legs, three direct-drive joints per leg. All 1,753 original parts and 59 mesh byte streams are preserved in 19 rigid bodies and 18 revolute joints.

Start with the [complete import report](../../docs/UPDATED_CAD_IMPORT.md), [mass/COM/inertia breakdown](MASS_INERTIA.md) and [independent geometric validation](../../artifacts/mkii_updated_2026-09-10/import_001/joint_validation/README.md). This candidate has not been admitted for native Isaac dynamics or hardware transfer.

## Interactive inspection

From the repository root:

```sh
python3 -m http.server 8347 --bind 127.0.0.1 --directory robot
```

Open http://127.0.0.1:8347/hexapod_mkii_updated_v1/preview/ . Select any joint and press Sweep, select parts to inspect grouping, use X-ray or separate parts, and filter the 78 parts involved in the exported screw/tibia overlaps. Preview edits are local and can be saved as JSON. The 59 meshes load once and are shared across part instances; Three.js is the repository’s existing vendored copy in the older asset preview.

## Model variants

| File | Mass | Purpose |
|---|---:|---|
|[Inspection URDF](urdf/hexapod_updated_inspection.urdf)|5.147603654kg|Exact source CAD inertias; effort disabled; kinematic review|
|[RS05 raw CAD URDF](urdf/hexapod_updated_rs05_raw_cad.urdf)|5.147603654kg|Exact source inertias with sourced actuator peak fields; motors underweight|
|[RS05 nominal mass candidate](urdf/hexapod_updated_rs05_mass_corrected.urdf)|7.466088235kg|Missing motor mass added explicitly; distribution provisional|

[RS05_VARIANTS.md](RS05_VARIANTS.md) and [rs05_mass_correction.json](rs05_mass_correction.json) record every addition. Original data remains in [model.json](model.json); [model_rs05_mass_corrected.json](model_rs05_mass_corrected.json) changes only aggregate mass/inertia and related metadata. The meshes use portable ASCII names and retain source SHA-256s in [mesh_mapping.json](mesh_mapping.json).

## Geometry and conventions

The export contained one fixed body and zero joints. This articulated graph is recovered from actual cylindrical axes and six-copy relative poses; the original mate graph was absent. Units are metres, kilograms and radians. Keep CAD axes +X left, −Y forward, +Z up. Anatomical prefixes are lf/lm/lr/rf/rm/rr. Every joint uses local +Z. The canonical inspection zero is radial yaw, femur +20° above horizontal, relative knee −110°; [cad_pose.json](cad_pose.json) restores the exported pose without moving hinge axes.

Yaw bounds are ±25°, femur ±20°, and knee −3°…+20° about that inspection zero. These are provisional inspection envelopes. All 133 tested poses avoid new cross-body contacts and primary-structure interference. Source CAD already has 72 screw/tibia overlap pairs; representative penetration is 0.286–0.299mm. Use [overlap_review_parts.json](overlap_review_parts.json) for exact IDs. Ownership of bearing races/rotors and the 79 uniquely placed chassis components remains inferred and flagged.

The nonspherical tibia tip and hollow brackets are retained. The root starts at 0.081611091m to leave 5 mm minimum mesh-to-ground clearance at the inspection stance. Kinematic joint sweeps lift it 40 mm farther; this is neither a planted-foot nor standing simulation.

## Physics processing

The source-per-part URDF collision elements retain provenance and are **not the final Isaac collision representation**. Use the sibling [USD preparation bundle](../../artifacts/mkii_updated_2026-09-10/usd_001/README.md) for the explicit uncooked SDF collision recipe and preserved full tensors. It includes a raw and a nominal motor corrected stage, floating articulation, all 1,753 visuals and an exposed-structure collision inventory. Native GPU cooking, contacts and memory use remain unverified; no drive gains or measured contact material have been invented.

This asset is a new physical-design lineage. It does not replace or qualify the selected C-study checkpoints, acceptance gates, old four-bar assets, old motor adapter or historical evidence. Do not route it through the four-bar runner. Native admission and a new runtime adapter must bind this exact source, joint ordering, signs, actuator configuration and asset hashes.

## Reproduction and evidence

The [immutable intake bundle](../../artifacts/mkii_updated_2026-09-10/import_001/README.md) contains the ZIP, safe extraction recipe, full part ledger, geometric recovery, builders, independent validation, rejected initial knee range and actual Claude Code review. [BUILD_REPORT.md](BUILD_REPORT.md) is the builder’s numerical report. Follow the bundle’s commands to rebuild into a new directory and compare source-bound outputs.
