# Updated detailed CAD import — immutable evidence, 10 September 2026

This bundle preserves the source, reconstruction decisions, successful checks and failed attempts for the user's updated **direct-drive, no-four-bar** robot. It does not replace the active simplified C-study or the historical physical four-bar campaign.

The supplied `HexapodLegUpdatedV2.zip` contains **one rigid link, zero joints, 1,753 part instances and 59 unique STL meshes**. Its pickle retains exact per-part mass, COM and full inertia tensors; it does **not** retain a joint, closure or named-frame graph. The user confirmed the new direct-drive topology. The resulting **19-body/18-joint graph and individual part ownership are a documented geometric reconstruction**, not an exported mate graph or hardware calibration.

## Verified results and limits

- All 1,753 source parts are represented exactly once. Full source-pose transforms, original mesh bytes and tensor sums are preserved.
- Raw CAD mass is **5.147603654203134 kg**. There are 18 distinct motor housings and repeated vendor part sets; the vendor CAD contributes only **62.1953010543 g per motor**, including 108 zero-mass internal instances.
- A separate nominal RS05 variant adds **128.8046989457 g per motor** and totals **7.466088235225788 kg**. The added housing-centered cylinder inertia is an explicit approximation, not measured motor/rotor mass distribution. Original CAD part masses and unrelated material densities are unchanged.
- Source mass/COM/full-inertia sums match every fused URDF field within its written decimal precision. All positive-mass part tensors and all 19 link tensors pass physical consistency checks.
- Independent XML/STL verification covers 10,888,953 instanced mesh vertices: worst source agreement is **0.738 μm**, consistent with the rounded source XML. Near-gimbal serialization is independently checked.
- The initial ±20° knee envelope failed structural collision checks and remains preserved. The retained knee offset is **−3° to +20°** around a −110° neutral relative knee angle. The finite exact-triangle follow-up found **1.027 mm minimum primary-structure separation** at the lower bound. These are inspection bounds, not measured stops or continuous-sweep certification.
- The final 133-pose check found no new cross-body contact pairs or primary-structure intersections. It retained **126 existing contact pairs**, including **72 tibia/M3-screw pairs**. Representative screws penetrate the original CAD tibia by approximately **0.286–0.299 mm**. Geometry and attachment-side review remain necessary; the original overlap was not erased to pass a check.

Assigned material names are absent, but the user explicitly accepted the exported mass properties without them. Estimated mass/mesh-volume density and appearance RGBA remain separate from engineering material identity.

**This evidence establishes a detailed, inspectable geometry and mass candidate, not Isaac physics admission or sim-to-real qualification.** Collision/USD preparation and viewer QA are recorded in sibling bundles owned by root integration. Existing gates, task IDs, checkpoint bytes and historical failures are unchanged.

## Contents

| Path | Preserved evidence |
| --- | --- |
| `source/` | Exact 12,141,111-byte ZIP, original per-file extraction/name-decoding manifest, bounded safe extractor and byte-verification receipt. No duplicate expanded source tree. |
| `mass_audit/` | Restricted pickle reader/auditor, five tests, original execution log, all 1,753 instance and 59 type ledgers, source hashes and [mass/material report](mass_audit/evidence/MASS_MATERIAL_AUDIT.md). |
| `geometry_recovery/` | Current-mesh axis/ownership recovery, reports, exploratory scripts and the preserved legacy-importer failure. |
| `joint_validation/` | Independent serialized-URDF/source-vertex audit, exact-triangle finite sweeps, knee-bound scan, source/neutral penetration evidence and the initial failed envelope. Local installed `deps/` and caches are excluded. |
| `builder/` | Articulated/RS05 builders, three numerical regression tests, exact build/mass reports, mesh/CAD-pose mapping and a hash inventory of original builder outputs. Expanded meshes and viewer models are delivered in [`robot/hexapod_mkii_updated_v1`](../../../robot/hexapod_mkii_updated_v1); they are not duplicated here. |
| `claude_partner/` | Exact prompt, result, launch/exit/runner/stderr, verified successful Fable 5.1 maximum-reasoning consultation and [decision disposition](claude_partner/DECISION_DISPOSITION.md). The review preceded the user's no-four-bar confirmation. |
| `tests/` | Fresh execution of the five mass and three builder tests from the copied evidence files, with a receipt. |
| `copy_receipt.json` | Original scratch/download paths, copied destinations, sizes and hashes; copied source bytes were not rewritten. |
| `SHA256SUMS` | Final file integrity manifest for this evidence bundle; excludes only itself. |

The integration follows [`docs/PROJECT_SITE.md`](../../../docs/PROJECT_SITE.md). Root owns the associated poster update, current execution/design documents, publication and remote/deployment checks. This worker modified only this evidence directory and did not commit.

## Reproduce into a fresh scratch directory

Run the following from the repository root. The temporary environment is separate from the repository's `uv` workspace. Python 3.12.12, NumPy 2.5.2, SciPy 1.18.1, trimesh 5.1.0 and python-fcl 0.7.0.11 were used; exact scratch pins are in `reproduction_requirements.txt`. FCL wheels may depend on platform availability.

```sh
export HEXAPOD_AUDIT_EVIDENCE="$PWD/artifacts/mkii_updated_2026-09-10/import_001"
export HEXAPOD_AUDIT_SCRATCH="$(mktemp -d /tmp/hexapod-urdf-reproduce.XXXXXX)"
uv venv --python 3.12 "$HEXAPOD_AUDIT_SCRATCH/.venv"
uv pip install --python "$HEXAPOD_AUDIT_SCRATCH/.venv/bin/python" \
  -r "$HEXAPOD_AUDIT_EVIDENCE/reproduction_requirements.txt"
export HEXAPOD_AUDIT_PY="$HEXAPOD_AUDIT_SCRATCH/.venv/bin/python"
```

Verify or extract the immutable source. The extractor pins the ZIP SHA-256, limits member count/size/expansion, rejects special/path-traversing/encrypted members, verifies every member hash and never loads the pickle. `--verify-only` creates no expanded copy. For reconstruction, select `--out` instead:

```sh
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/source/extract_source.py" \
  --archive "$HEXAPOD_AUDIT_EVIDENCE/source/HexapodLegUpdatedV2.zip" --verify-only
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/source/extract_source.py" \
  --archive "$HEXAPOD_AUDIT_EVIDENCE/source/HexapodLegUpdatedV2.zip" \
  --out "$HEXAPOD_AUDIT_SCRATCH/source"
```

Audit masses, recover the current geometry and construct all three variants:

```sh
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/mass_audit/audit_onshape_mass_properties.py" \
  --source "$HEXAPOD_AUDIT_SCRATCH/source" \
  --archive "$HEXAPOD_AUDIT_EVIDENCE/source/HexapodLegUpdatedV2.zip" \
  --out "$HEXAPOD_AUDIT_SCRATCH/mass"
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/geometry_recovery/recover_geometry.py" \
  --source "$HEXAPOD_AUDIT_SCRATCH/source" --out "$HEXAPOD_AUDIT_SCRATCH/geometry"
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/builder/build_updated_urdf.py" \
  --source "$HEXAPOD_AUDIT_SCRATCH/source" \
  --mass-ledger "$HEXAPOD_AUDIT_SCRATCH/mass/part_instances.json" \
  --geometry "$HEXAPOD_AUDIT_SCRATCH/geometry/joint_geometry_recovery.json" \
  --ownership "$HEXAPOD_AUDIT_SCRATCH/geometry/provisional_body_ownership.json" \
  --out "$HEXAPOD_AUDIT_SCRATCH/model"
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/builder/build_rs05_variants.py" \
  --raw-dir "$HEXAPOD_AUDIT_SCRATCH/model" \
  --mass-audit "$HEXAPOD_AUDIT_SCRATCH/mass/mass_audit.json" \
  --part-types "$HEXAPOD_AUDIT_SCRATCH/mass/part_types.json" \
  --spec-review "$PWD/docs/RS05_SPEC_REVIEW.md"
```

Run independent validation and the focused tests from the preserved files:

```sh
"$HEXAPOD_AUDIT_PY" "$HEXAPOD_AUDIT_EVIDENCE/joint_validation/validate_candidate.py" \
  --model-dir "$HEXAPOD_AUDIT_SCRATCH/model" \
  --source "$HEXAPOD_AUDIT_SCRATCH/source" --out "$HEXAPOD_AUDIT_SCRATCH/validation"
"$HEXAPOD_AUDIT_PY" -B -m unittest discover \
  -s "$HEXAPOD_AUDIT_EVIDENCE/mass_audit" -p 'test_*.py' -v
"$HEXAPOD_AUDIT_PY" -B -m unittest discover \
  -s "$HEXAPOD_AUDIT_EVIDENCE/builder" -p 'test_*.py' -v
```

The scratch FCL path in `validate_candidate.py` is optional: its normal Python import uses the installed scratch environment when an adjacent `deps/` is absent. Reproduction reports may differ in runtime paths, timestamps and dependency metadata; compare the recorded numeric, topology, source-byte and geometry/bounds contracts. Do not overwrite frozen evidence with a rerun.

## Frozen exploratory paths and failures

`geometry_recovery/inspect_geometry.py`, `probe_local.py`, `mass_audit/probe.py`, `joint_validation/refine_knee_bounds.py`, `inspect_neutral_contacts.py` and `inspect_source_cad_contacts.py` retain their exact original `/tmp/hexapod-urdf-…-20260910` and/or original-checkout paths. Those paths identify the inputs actually used; they are **not portable entry points**. The main argument-driven recovery/audit/builder/validator commands above are the supported reproduction route.

To repeat a specific exploratory follow-up, first copy its script to a new scratch directory, explicitly rebind its recorded source/model/output/dependency paths there, and preserve that adapted script alongside the new outputs. The two penetration scripts load the prefix of `refine_knee_bounds.py` through its recorded absolute path. Do not run these frozen scripts blindly or mutate their archived text. No live secrets, credentials or environment files were copied.

`geometry_recovery/legacy_import_diagnostic.log` preserves the old importer failing when its old structural families have no matches. `joint_validation/initial_unsafe_envelope/` preserves both the rejected knee envelope and the pre-fix XML serialization defect. Later successful validation does not rewrite either failure.

The Claude runner also writes alongside itself. Its original successful response is evidence, not an instruction to rerun a billable consultation. Any future authorized consultation should copy the runner and a new prompt to a fresh scratch directory; never overwrite this archived session.

## Integrity

From this directory, run `shasum -a 256 -c SHA256SUMS`. The manifest covers all finalized source, evidence, scripts, reports and receipts in this bundle except the manifest itself. Sibling collision, USD and viewer evidence have separate ownership and manifests.
