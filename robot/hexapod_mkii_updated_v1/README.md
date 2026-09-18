# Approved MKII direct-drive model

You select this robot through [active_model.json](../active_model.json).
The approved model has 19 rigid bodies, 18 direct-drive joints and a total mass
of 7.466088235 kg after the RS05 motor correction. The source contains 1,753
parts and 59 mesh byte streams. Keep the original meshes and corrected inertias.

## Simulation inputs

[inputs.json](inputs.json) pins the approved URDF, model, USD, geometry and
neutral stance. Run `uv run python -m locomotion.inputs check` from the repository
root. Use `inputs pack` to copy those inputs for a fresh native allocation.
The [kernel guide](../../locomotion/README.md) describes admission and evaluation.
Historical standing results apply to their named source packs; the consolidated
source needs fresh native admission. Hardware calibration remains pending.

Use [prepare_updated_usd.py](../../tools/assets/prepare_updated_usd.py) and
[prepare_geometry.py](../../tools/assets/prepare_geometry.py) for asset preparation.
Write changed assets to a fresh directory and preserve this approved input set.
The existing USD uses the 153-collider SDF recipe. Geometry extrema accelerate
measurement against those meshes; they do not replace the collision surfaces.

## Inspection

Run `npm ci` and `npm run dev` from `viewer/`, or serve `robot/` with:

```sh
python3 -m http.server 8347 --bind 127.0.0.1 --directory robot
```

Open the [joint inspector](http://127.0.0.1:8347/hexapod_mkii_updated_v1/preview/index.html).
You can sweep a joint, play the full sequence, reset the pose and inspect part
groups. The preview uses its local `preview/vendor/` dependencies. A CAD sweep
shows joint travel; walking acceptance requires a native policy recording.

## Source and conventions

Use CAD axes +X left, −Y forward and +Z up. Anatomical prefixes are
lf/lm/lr/rf/rm/rr. The approved joint order lives in
[env_config.py](../../locomotion/env_config.py). Keep metres, kilograms and radians.

[UPDATED_CAD_IMPORT](../../docs/UPDATED_CAD_IMPORT.md) records the recovered
articulation and joint limits. [MASS_INERTIA](MASS_INERTIA.md) and
[RS05_VARIANTS](RS05_VARIANTS.md) record the inertias and motor correction.
The inspection and raw-CAD URDFs preserve the source comparison; use the
mass-corrected URDF for training. Read [BUILD_REPORT](BUILD_REPORT.md) for the
import audit and [overlap_review_parts.json](overlap_review_parts.json) for the
known source screw/tibia overlaps.

The [pinned intake bundle](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/mkii_updated_2026-09-10/import_001)
preserves the source ZIP and import builders. Restore it through
[the archive tool](../../docs/PIPELINE_LINEAGES.md) for a named CAD rebuild.
