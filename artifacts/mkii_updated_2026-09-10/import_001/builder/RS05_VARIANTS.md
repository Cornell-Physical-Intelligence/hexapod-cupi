# RS05 dynamics variants for the updated direct-drive robot

**Raw CAD stays unchanged.** Three URDFs are provided:

- `hexapod_updated_inspection.urdf`: raw CAD mass ledger and finite inspection bounds; physical actuation disabled in this visualization-only version.
- `hexapod_updated_rs05_raw_cad.urdf`: identical raw mass/geometry, with sourced RS05 peak torque and speed fields. The motors remain underweight.
- `hexapod_updated_rs05_mass_corrected.urdf`: RS05 peak torque/speed plus missing nominal motor mass restored on the corresponding housing links. This is the dynamics candidate; missing-mass spatial allocation remains provisional.

All variants retain the same 1,753 meshes, 19 rigid bodies, 18 joints and tightened finite knee inspection interval [−3, +20] degrees. Raw mass **5.147603654203 kg**; corrected mass **7.466088235226 kg**.

## Motor correction

The 18 unique housing instances and repeated vendor component counts establish 18 CAD motor sets, each contributing 62.195301054g. The established RS05 nominal is 191 g. Each housing link receives **128.804698946g**, totaling **2.318484581023 kg**.

The added mass is centered at the actual housing mesh bounding-box center [0.0, 0.0, 0.005949999671429396] m in its mesh frame. Its approximate solid cylinder inertia uses **46.000000 mm diameter ×35.099999 mm height**, measured from this export. This reproduces the existing repository mass top-up method, without reusing old robot total masses or changing unrelated material density.

**This is not measured rotor/stator mass distribution.** Motor COM, internal rotor inertia, output-side inertia, housing ownership and real manufactured mass remain calibration tasks. All original exported part masses, COMs, full tensors and meshes are unchanged; 18 separate additions are recorded in the model and correction JSON.

## Actuator fields

Sourced from `docs/RS05_SPEC_REVIEW.md`: **5.5 N·m peak effort** and **50.265482457437 rad/s (480rpm) maximum speed**. They describe separate envelope limits, not a simultaneous operating point. Runtime must enforce torque-speed behavior, continuous duty, thermal/current/voltage protections and identified actuator dynamics. Viewer sweep speed remains 0.5 rad/s and has no hardware-limit meaning.

## Per-link mass before and after

| Link | Raw CAD (kg) | Addition (kg) | Corrected (kg) | Motors |
|---|---:|---:|---:|---:|
|body|1.908769948|0.000000000|1.908769948|0|
|lf_coxa|0.243634014|0.257609398|0.501243411|2|
|lf_femur|0.173673948|0.128804699|0.302478647|1|
|lf_tibia|0.122497656|0.000000000|0.122497656|0|
|lm_coxa|0.243634014|0.257609398|0.501243411|2|
|lm_femur|0.173673948|0.128804699|0.302478647|1|
|lm_tibia|0.122497656|0.000000000|0.122497656|0|
|lr_coxa|0.243634014|0.257609398|0.501243411|2|
|lr_femur|0.173673948|0.128804699|0.302478647|1|
|lr_tibia|0.122497656|0.000000000|0.122497656|0|
|rf_coxa|0.243634014|0.257609398|0.501243411|2|
|rf_femur|0.173673948|0.128804699|0.302478647|1|
|rf_tibia|0.122497656|0.000000000|0.122497656|0|
|rm_coxa|0.243634014|0.257609398|0.501243411|2|
|rm_femur|0.173673948|0.128804699|0.302478647|1|
|rm_tibia|0.122497656|0.000000000|0.122497656|0|
|rr_coxa|0.243634014|0.257609398|0.501243411|2|
|rr_femur|0.173673948|0.128804699|0.302478647|1|
|rr_tibia|0.122497656|0.000000000|0.122497656|0|

Full before/after COM and inertia matrices, addition centers/tensors, source hashes and all checks are in `rs05_mass_correction.json`. Every corrected link tensor remains positive definite and physically consistent. Raw files were verified byte-identical after variant creation.

**Isaac qualification remains pending:** collision meshes need preparation; source-export topology was absent and per-part ownership remains geometrically inferred; simultaneous clearance, hardware stops, reset/contact stability and bounded physics probes must pass. Sourced peak fields plus corrected total mass do not by themselves establish sim-to-real readiness.
