# Canonical model: native import inspection, Phase A

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

The native SDF view must contain exactly the 153 authored collider paths and report valid objects with no pending cooking tasks. This establishes representation presence, **not SDF distance accuracy, collision fidelity, absence of every possible fallback, or contact qualification**. No convex replacement or whole-robot collision simplification is introduced. Native API/version/source paths are retained for first-run review.

The first readable native state follows SDK reset/play. The exact installed manager source contains two `update_simulation(dt,0)` calls during warmup; play also pumps the application. Therefore the explicit counter is **not** proof of exactly two warmup steps or ten total native steps. The saved pre-reset USD and first native state bracket this lifecycle; warmup motion itself is unobserved. Eight further explicit 2.5 ms steps produce nine stored samples, including the initial readable state. Each carries raw link poses in XYZW, COM linear/angular velocity, joint angle and raw SDK joint rate. Joint-to-link FK is checked in the actual world frame. Angle-difference rates are separate interval measurements and never replace SDK data or score a quiet gate.

Native `get_inertias` is interpreted according to the [107.3 tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html): full COM-referenced inertia in the rigid-body-prim frame, column-major. The installed high-level data wrapper's older world-frame wording is not used. The first actual readback is required to validate this interface; no guessed alternate frame is silently accepted. A tolerance failure retains raw tensors and quaternions for diagnosis.

## Failure and lifecycle evidence

`state.json` records inspection scope and inputs, never a training/standing pass. `usd_readback.json`, `native_readback.json`, `sdf_readback.json`, `samples.json`, runtime API/config/stage exports and motion diagnostics are sealed before `SimulationApp.close`, which may terminate the process directly. Terminal input changes fail the receipt. Tracebacks, partial readbacks and nonfinite values (explicit JSON `nonfinite` markers) survive failures. Mutable `isaac_logs/**` and `native_errors.json` are explicitly excluded from the pre-close seal and must be inventoried by the host after exit. The standard-library result validator independently reads the post-exit native error file and rejects any late event or recorded runtime error. The explicit native step counter must advance exactly once per observed step, from whatever baseline the lifecycle established. There is no claim that shutdown logs were already immutable.

## CPU evidence and reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tmp/updated_native_inspection_001 -p 'test_*.py' -v
python3 -S tmp/updated_native_inspection_001/verify_bundle.py
```

Tests inspect the actual composed canonical USD offline, permute native joint/body fixtures, reject wrong frames/full inertia/limits/gains/missing SDF paths/nonfinite data, preserve a failed native getter, and verify failure sealing before process-exit-style close. Synthetic native fixtures do not prove the native APIs, cooked shapes, hardware throughput or physical behavior.

Remaining phases require a separately reviewed exact-model actuator/effort contract, appropriate suspended motion/coordinate tests, contact/support classification for the real +X tibia surface, gravity/stance admission and measurement consistency. Existing 1.6 Nm historical applied caps and raw-versus-applied/400 Hz diagnostics are lessons and candidate constraints, not an identified new motor model. URDF peak 5.5 Nm and no-load 50.265 rad/s are not an admissible simultaneous envelope. No new training can reuse the historical C/four-bar asset.
