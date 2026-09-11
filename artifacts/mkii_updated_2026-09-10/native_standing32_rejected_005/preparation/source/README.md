# Standing005: controlled TGS 32/0 comparison

This is an unexecuted successor of frozen standing004 (`03701335…`). It changes only the physical setting of articulation velocity iterations from 4 to 0; position iterations remain 32. The exact model, 153 SDF shapes per robot, inertias, material, 400 Hz external PD, 50 Hz held neutral target, provisional 48 V/1.6 N·m envelope, reset, contact classifier and physical/quiet thresholds remain unchanged. All nine asset bytes remain bound. The battery and motor model remain provisional.

The previous 32/4 single-robot experiment completed 8,000 steps with no support, nonfoot or motor rejection, but LM/RM tibia SDK-rate RMS was about 0.038802/0.039215 rad/s against the unchanged 0.03 bound, while joint position ranges were only about 42/46 µrad. It was rejected; no 32-robot follow-on ran. This candidate assumes neither that more iterations improve this case nor that small position excursions establish zero physical velocity.

The recipe first requires actual un-authored 32/1 asset fallbacks, then explicitly authors and verifies 32/0 before initialization, after reset and after all controlled steps. These are USD readbacks, not independent backend iteration introspection. Execute one robot first; only a successful, exact same-source one-robot result admits 32. Earlier accepted one-robot results cannot admit this source. No training admission is conferred.

The numerical hypothesis is consistent with NVIDIA's documented [TGS velocity-iteration limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/current_limitations.html). The tendon and D6-drive warnings on that page are not claimed to describe this no-drive, no-tendon robot. PhysX also documents that [constrained pose increments and reported velocities need not agree](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html). The unchanged SDK-rate gate and PD derivative therefore remain intact; finite differences are retained separately.

Three read-only diagnostics add evidence:

- `legacy_friction_readback.json`: actual deprecated `get_dof_friction_coefficients`, by native joint order. Nonzero values remain valid observations; no friction setter or zero-value requirement exists. New-model friction triples remain in `native_readback.json`. [Official issue 498](https://github.com/NVIDIA-Omniverse/PhysX/issues/498) motivates checking both, but its CPU reproduction does not establish the installed GPU/parser behavior.
- `link_com_velocity_world`: copied `(N,19,6)` global linear velocity at link COM, then angular velocity, in native body order. This derives from native articulation state and is not independent velocity ground truth.
- `floor_contact_force_matrix_world_n`: copied `(19N,1,3)` native aggregate for the sole `/Ground` filter, in `contact_view.json` sensor order. It may share the detailed patch backend. It neither replaces toe/shaft classification nor independently calibrates tangential friction.

The latter two channels are retained at every controlled 400 Hz step and at 50 Hz endpoints. Shape/finite, source, coverage and sealed-file checks concern evidence integrity only. A getter failure preserves the actual step counter and preceding partial fields. `DIAGNOSTIC_API_REVIEW.json` binds the actual installed tensor110 source without redistributing its implementation.

All 28 CPU tests pass; exact parent common-row/effort/counter and NumPy RNG parity, copied-buffer isolation, getter failures, nonzero legacy values, missing diagnostic rejection and unchanged numeric gates are covered. The earlier test-only module-restoration failure is preserved and explained in `SOLVER_CHANGE.json`. Run from the repository root:

```sh
.venv/bin/python3 -B -m unittest discover -s tmp/updated_native_standing_005 -p 'test_*.py'
```

Root owns final freeze, source staging and guarded dispatch. The separate tools-disabled Fable MAX review is pending and cannot manufacture native admission. No frozen predecessor was edited. The central project-site contract governs later publication.
