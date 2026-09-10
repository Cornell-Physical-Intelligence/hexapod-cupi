# Canonical detailed-model native inspection 001: API startup failure

The first isolated Isaac Sim 6.0.1 inspection reached application readiness, then failed on a private Python module import before creating the simulation context or loading the robot. It measured no native CAD dynamics or collision shapes. This is preserved infrastructure evidence, not a negative result about the robot and not physical or training admission.

The exact failure is `ModuleNotFoundError: No module named 'omni.physics.tensors.impl'` in the diagnostic source-provider lookup. Subsequent CPU-only inspection of the installed image found tensor package110.1.13 and its public `api.py` provider. The older107.3 documentation did not describe that installed private-module layout. A successor must bind the installed public API explicitly;001 remains unchanged.

## Allocation and evidence

The canonical selector pins the7.466088235kg,19-body/18-joint direct-drive robot. The allocated scope was zero gravity, no ground, no drive gains or target/state setters, SDK initialization followed by eight explicit2.5ms steps, exact native identity/inertia/limits and153SDF paths. It never reached those steps. Source/asset mounts were read-only. Both shared GPU locks, exact ownership, AppReady90s and total600s bounds remained in force.

Root independently passed15 inspector and31 host/guard CPU tests, the exact21-source/9-asset/926-supervisor input checks, and actual Spark Python3.12.3 standard-library host setup before dispatch. Those checks did not prove native API imports. The failed native job ran under invocation`87a75ef9d130496a9e040b86b14367e9`. Its generic inherited supervisor error mentions standing; this allocation contained no standing controller.

`terminal/audit.json` verifies all immutable input inventories, the authentic failed outcome, both recorded owned identifiers absent, exact weather restoration and two complete raw hash passes. All13 original raw payloads/27,786bytes are included. Weather timers were restored at1789071659.3738954. No unrelated workload was signalled. The terminal audit distinguishes `audit_verified=true` from `inspection_completed=false`.

## Reproduction and follow-on

Run `python3 -B -S verify_bundle.py` in this directory to verify every published payload. Nested preparation freezes, original failure/state/logs, actual root setup, terminal auditor and the bounded Fable5.1 maximum-effort review are retained. The reviews accurately identify their draft snapshots; they are not a claim of successful native execution.

Prepare a separately frozen retry with the installed tensor public-provider binding and an explicit supported cooking/initialization observation contract. Keep the exact CAD, masses/inertias, limits, SDF detail and no-admission boundary. Native import must still precede model-bound actuator tests, suspended coordinate motion, contact/support diagnostics and scratch PPO admission. Stage2 and Stage3 remain incomplete.
