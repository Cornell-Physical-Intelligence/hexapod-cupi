# Canonical model: native import inspection, Phase A successor003

This preparation opens the exact user-selected 7.466088235 kg direct-drive robot in Isaac Sim 6.0.1. It is a new standalone inspection program, with no historical robot task, actor, checkpoint, action pipeline, controller gain or actuator adapter. **No native execution has occurred for this bundle. Physical admission and training authorization remain false even if this inspection completes.**

The fixed inspection uses one floating articulation, zero gravity, no floor, no added suspension joint, and no targets or state setters. It preserves the authored self-collisions, all 153 concave SDF collider instances and their resolutions, all 19 masses/full inertia tensors, and 18 per-name joint coordinates/limits. Zero gravity is an isolation setup, not a promise that the robot remains stationary: self-contact and constraint correction can move it. CAD inspection height is not a standing stance.

## Exact inputs and interface

`ASSET_SHA256.json` binds all nine existing files under `artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/`; no large geometry is duplicated here. Its SHA-256 is `4cf88f1a658c23e20ddf3f8ed7a08a100ce2ddd337e59605c6f611bc7570d858`. `CANONICAL_SELECTION.json` copies the selector current at preparation. The exact model/URDF/USD digests match it. All source/asset files must be mounted read-only, with a separate fresh writable output.

```sh
python3 -B run_inspection.py --asset /asset --output /output/inspection --preflight-only --device cuda:0 --headless
# Only root's separately reviewed owned host may dispatch this native invocation:
/workspace/isaaclab/_isaac_sim/python.sh /inspection/run_inspection.py --asset /asset --output /output/inspection --device cuda:0 --headless
```

`inspection_contract.py` is standard-library-only. `verify_inputs(args)` checks exact own freeze and asset inventories before AppLauncher. `args.asset` and `args.output` are required; host can also pass `args.source`. `validate_result(directory, identity)` verifies terminal status, unchanged identities, required checks and sealed raw output hashes. `runtime_binding.runtime_tree_sha256` is this new inspector freeze, scoped canonical inspection, although the generic host's compatibility field/marker retains its inherited name. `REFERENCE_SCREEN_APP_READY` prints only after actual AppLauncher construction. Parent ownership supervision retains two locks, exact container ownership/cleanup, 90-second AppReady and 600-second total phase limits. No automatic retries or resolution reductions are authorized by this preparation.

## What it measures

Before native initialization, the script reads the composed USD units, named body/joint graph, local joint frames, mass/COM/principal-axis reconstruction, dynamic/floating status, no-drive state and collider inventory. Raw native articulation views then read the actually imported body/DOF order, masses, COM frames, full inertia tensors, joint limits, gain/damping/armature/friction/max-force/max-speed properties. Reads occur by observed names; no assumed index order or 18-action checkpoint compatibility is used. A missing/failed getter preserves earlier successful properties and identifies the failing getter.

The native SDF view must contain exactly the 153 authored collider paths and report valid objects after `sim.reset()` returns and both physics/articulation views are valid. The identity explicitly names `canonical_native_sdf_initialization_v2`. An optional legacy cooking task getter is recorded if available; its absence is represented by null, never invented zero. A present positive count still rejects completion. This establishes representation presence, **not SDF distance accuracy, collision fidelity, absence of every possible fallback, or contact qualification**. No convex replacement or whole-robot collision simplification is introduced. Native API/version/source paths are retained for first-run review.

The first readable native state follows SDK reset/play. The exact installed manager source contains two `update_simulation(dt,0)` calls during warmup; play also pumps the application. Therefore the explicit counter is **not** proof of exactly two warmup steps or ten total native steps. The saved pre-reset USD and first native state bracket this lifecycle; warmup motion itself is unobserved. Eight further explicit 2.5 ms steps produce nine stored samples, including the initial readable state. Each carries raw link poses in XYZW, COM linear/angular velocity, joint angle and raw SDK joint rate. Joint-to-link FK is checked in the actual world frame. Angle-difference rates are separate interval measurements and never replace SDK data or score a quiet gate.

Native `get_inertias` is interpreted from the actual extracted110.1.13 `api.py`, with exact SHA in `ACTUAL_NATIVE_API.json`: full COM-referenced inertia in the rigid-body-prim frame, column-major. The installed high-level data wrapper's older world-frame wording is not used. The first actual readback is required to validate this interface; no guessed alternate frame is silently accepted. A tolerance failure retains raw tensors and quaternions for diagnosis.

## Failure and lifecycle evidence

`state.json` records inspection scope and inputs, never a training/standing pass. `usd_readback.json`, `native_readback.json`, `sdf_readback.json`, `samples.json`, runtime API/config/stage exports and motion diagnostics are sealed before `SimulationApp.close`, which may terminate the process directly. Terminal input changes fail the receipt. Tracebacks, partial readbacks and nonfinite values (explicit JSON `nonfinite` markers) survive failures. Mutable `isaac_logs/**` and `native_errors.json` are explicitly excluded from the pre-close seal and must be inventoried by the host after exit. The standard-library result validator independently reads the post-exit native error file and rejects any late event or recorded runtime error. The explicit native step counter must advance exactly once per observed step, from whatever baseline the lifecycle established. There is no claim that shutdown logs were already immutable.

## CPU evidence and reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tmp/updated_native_inspection_003 -p 'test_*.py' -v
python3 -S tmp/updated_native_inspection_003/verify_bundle.py
```

Tests inspect the actual composed canonical USD offline, permute native joint/body fixtures, reject wrong frames/full inertia/limits/gains/missing SDF paths/nonfinite data, preserve a failed native getter, and verify failure sealing before process-exit-style close. Synthetic native fixtures do not prove the native APIs, cooked shapes, hardware throughput or physical behavior.

Remaining phases require a separately reviewed exact-model actuator/effort contract, appropriate suspended motion/coordinate tests, contact/support classification for the real +X tibia surface, gravity/stance admission and measurement consistency. Existing 1.6 Nm historical applied caps and raw-versus-applied/400 Hz diagnostics are lessons and candidate constraints, not an identified new motor model. URDF peak 5.5 Nm and no-load 50.265 rad/s are not an admissible simultaneous envelope. No new training can reuse the historical C/four-bar asset.

## Actual001 failure and narrow002 correction

Frozen001 reached AppReady on Spark then failed before creating SimulationContext because its provenance readback assumed private `omni.physics.tensors.impl.api`. The actual6.0.1 image uses public `omni.physics.tensors.api`, extension110.1.13. Root extracted its exact public factory/provider and package init using a CPU-only image container.002 resolves source through the public `create_simulation_view` function and verifies both extracted hashes; it does not import a guessed private module. Full proprietary/extracted SDK tensor source remains in the separate local readback and is not republished here. This bundle contains only hashes/signatures/our review.

The legacy cooking task method was not established in that installed API. This successor declares a changed inspection-only initialization contract: returned native reset plus actual valid physics/articulation/SDF representations, preserving all153 paths. The [NVIDIA cooking guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html#generate-mesh-colliders-cooking) says simulation startup blocks for required cooked data. That does not prove all cache tasks everywhere are idle or qualify collision accuracy. Optional legacy API absence remains explicit unknown. No geometry, inertia/frame math, drive/gain, gravity, timing, explicit-step count, contact or physical acceptance threshold changes. The original failed attempt and parent manifest remain immutable.

`test_native_api.py` uses the actual extracted factory definition without calling it or loading physics; it verifies exact provider hashes. That one regression needs the separately preserved local `tmp/canonical_native_sdk_readback_001/files`; the portable public-only fake-package and absence/positive-count/initialization tests remain self-contained. The complete current CPU run used the actual source and had no skipped tests.

## Actual002 native import and narrow003 binding correction

Root observed actual002 pass composed USD/native body-DOF identity, full inertia, limits, frames, scene and zero-drive-gain checks. It then failed when `create_sdf_shape_view` forwarded a Python list to the compiled110.1.13 backend, whose actual TypeError requires `pattern: str`. This contradicts the extracted Python wrapper documentation.003 changes only this factory argument to the string `/Robot/*/collisions/part_*`. It still compares all153 returned paths as an exact set with the immutable USD collider inventory, and rejects extra/missing shapes, invalid views or incomplete initialization. Neither this correction nor the successful prefix is contact/physical/training admission.

The added CPU regression enforces the actual string-only backend signature; the unchanged core test rejects extra/missing returned paths. No source asset, scene/solver property, drive, inertia/frame convention or timing changed. Prior failures remain immutable.
