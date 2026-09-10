# Canonical tibia SDF query probe — prepared, no native query result

This is a small read-only addon for an already initialized native canonical robot. It compares the compiled SDF query with the exact nonspherical source tibia mesh. **It has not run against native SDF distances and admits neither contact, standing nor training.** Native inspector003's successful 153-shape presence check remains separate evidence.

The owner changes no tracked file, frozen inspector, physics setting, robot state, target, step count, host or guard. `run_probe(sim, articulation_view, fixture_path, asset_root, output)` in `native_probe.py` is the only integration seam. The root dispatcher must compose it into a new versioned entrypoint, bind this complete bundle and the canonical asset read-only, and retain existing initialization, deadline, cleanup and shutdown ownership. Call it at a declared observation boundary after the existing articulation kinematic refresh; it adds zero requested simulation steps. An actual invocation needs a fresh output directory outside both inputs. A returned `status: completed` means acquisition completed; the caller must inspect the separate diagnostic booleans.

## What is measured

The exact `mesh_056_tibia.stl` source has 11,523 vertices and 23,090 triangles. The CPU topology check finds a closed, consistently oriented mesh with positive volume. Its distal source coordinate is +X 130 mm; that is a CAD coordinate, not a measured hardware dimension. The fixture retains the original source arrays and their published hashes. It does not use a sphere approximation or an old distal +Y classifier.

Each of six exact tibia collider paths gets 140 source-local points and one reversed-order repeat, for 12 bounded native calls. The link also carries a separate flange collider, so a tibia-link wildcard is insufficient. The inputs comprise 24 points each around distal faces, distal edges, shaft surfaces and fork surfaces; eight geometry-verified void points; and 36 finite-difference stencil points. The voids are exterior space between opposed source surfaces, mainly fork cavities. They are not presented as a cavity in the distal contact pad.

The actual installed Tensor110 Python provider is pinned by SHA, without copying its implementation. Its documented distance-first/three-gradient ordering is a hypothesis. The scorer evaluates 144 channel/permutation/sign/unit hypotheses, records alternatives, and never silently adopts a favorable interpretation. Both signs and three-dimensional normal coverage are required. Sixteen anchor normals have condition number 1.63. It compares source-local, link-local and world-coordinate geometric interpretations, including the corresponding gradient rotation. Actual collider-to-link rotations make the link alternative distinguishable; an identity or symmetric ambiguous case fails the identification check. The world comparison can involve points outside the native field and is reported with that limitation rather than used as an unrestricted covariance proof.

Every live mesh's points, triangle indices, source tag, transform and stage units are checked. Exact native paths/counts/provider are recorded. Each returned persistent tensor is copied immediately. The first raw result is saved before a second native call can fail; nonfinite raw tensors are retained before rejection. A final identity check seals the output inventory. Before and after poses, velocities, joint angles/rates and explicit counter must be identical. This proves no observed state change during the addon; it does not measure an independent hidden native clock.

## Proposed diagnostic bounds and limits

`score_probe.py` contains new proposed measurement bounds, not inherited robot qualification gates. Distances are compared against the source triangles; gradients against nearest-surface displacement and a distance finite difference at source-verified smooth anchors. Nominal source AABB/resolution spacing is 0.149881 mm. The diagnostic uses 0.299762 mm RMS and 0.599524 mm maximum distance limits, gradient-component RMS 0.2, finite-difference discrepancy 0.25, and distance-fit slope 0.85–1.15. Slopes and intercepts are both reported. These limits distinguish gross metre/millimetre or sign/channel errors; they do not establish precision metrology or prove an optimal contact mesh.

The nominal spacing is **not** a readback of the cooked grid, sparse-band width or interpolation error. The preparation does not infer those from a model suggestion. A distance mismatch can reflect query semantics, frames, cooking or geometry. Raw errors and alternative scores must be examined before any runtime change. Exact repeat comparison may reject a nondeterministic backend; it does not waive the mismatch. Surface/edge/void results remain separately visible even if semantics pass.

Distances alone cannot establish contact point placement, contact/rest offsets, normal forces, friction, penetration, ground support, loaded deformation, torque safety or walking quality. Those require later native contact and load experiments with this same nonspherical asset. Proposed sequence: resolve query semantics and source-shape error; then separately sweep contact orientation/edge transitions and signed penetration against actual contact reports; only then evaluate supported loads and locomotion admission. No new physical threshold is adopted here.

## CPU evidence and reproduction

- `TESTS.txt`: 17 tests pass, including analytic box scale/sign/edge cases, rotated/translated gradients, a real void, actual mesh closest-point cross-check, channel/sign/unit failures, frame ambiguity, first-query preservation, nonfinite evidence and mutation-call audit.
- `CPU_NATIVE_CHECK.json`: the complete addon ran with a **fake** SDF backend and real source USD, using exact selected actual003 final poses. This exercises integration and output paths, not native distance accuracy. API-provider/identity bindings are deliberately replaced only inside that CPU test.
- `CPU_RECEIPT.json`: fixture reconstruction and source-oracle comparison. The two initial failed logs are preserved: Trimesh's absolute edge-region determinant tolerance at metre scale, and a syntactic mutation audit mistaking a local dictionary `update` for simulator advancement. The oracle was unchanged for the first; float64 millimetre normalization makes the independent Trimesh result agree at all 140 points within 4.2e-17 m. The second was resolved with plain dictionary assignments.
- `PARENT_NATIVE_SNAPSHOT.json`: exact selected final sample and body names from actual003, with both parent file hashes. This is a selection, not a claim to copy the complete parent run.

From the repository root with NumPy, trimesh and pxr available:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tmp/canonical_sdf_distance_probe_001 -p test_probe.py -v
.venv/bin/python -B tmp/canonical_sdf_distance_probe_001/verify.py
```

`prepare_fixture.py --asset-root <canonical-nine-file-directory> --output <fresh-directory>` regenerates the fixture without Isaac or a GPU. `cpu_native_check.py` needs that asset and a fresh `--output`; its backend remains fake. `replay_probe.py --run <actual-output> --state-sha256 <externally-verified-hash> --output <fresh-json>` verifies a future actual raw inventory and recomputes channel/geometry/frame diagnostics. It preserves failed prefixes without inventing missing results. That future native replay remains unexecuted.

## Partner review and publication

`partner/fable_final.json` is the final-only response and usage metadata from one requested `claude-fable-5-1 --effort max` consultation, session `b07dd2ff-c339-4454-b532-75d933aaf24c`, tools and MCP disabled. No internal thinking content or JSONL event stream is included. The CLI reports Fable primary usage and a small auxiliary Haiku metadata entry; no fallback model was requested or used to replace the review. `partner/DISPOSITION.md` records our independent acceptance, limits and rejection of unsupported recommendations. The partner is not an admission authority.

Only this new tmp directory is proposed for later publication. The live repository's `docs/PROJECT_SITE.md` applies: root must add a bounded central update, relevant current documentation and validated poster build. Suggested evidence destination is `artifacts/mkii_updated_2026-09-10/canonical_sdf_distance_probe_preparation_001/`. No runtime integration path or actor architecture has been adopted.
