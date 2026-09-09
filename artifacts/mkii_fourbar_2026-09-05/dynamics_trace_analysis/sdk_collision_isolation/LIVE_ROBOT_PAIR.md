# Live robot-pair collision control fixture

This fixture has **not been executed on a GPU yet**. CPU tests establish report/control logic and USD relationship checks. They do not prove native collision isolation.

`live_robot_pair.py` creates two actual v5 robots with the current source's motor profile, timing, solver settings and geometry. It resets both to the world origin. The filtered case uses the production task's explicit collision groups. An artifact-only subclass creates the negative case by adding reciprocal environment-to-environment allowlinks after the production setup and before physics initialization. Ground/global allowlinks stay unchanged. No production bypass switch is introduced.

Two native `PhysxManager.get_physics_sim_view().create_rigid_contact_view(...)` views observe body0→body1 and body1→body0. Each source and target is a fully resolved rigid-body path, with exactly one filter. The fixture records `get_contact_force_matrix(dt=...)` and the pair counts returned by `get_contact_data(dt=...)`; counts and nonzero forces must both appear in the negative control. The filtered control requires zero pair contacts and force at most 1e-6 N, while both robots must independently register ground support through their existing ground-filtered foot sensors. Unsupported view layouts, missing contacts in the negative control, missing filtered ground support, full contact buffers or nonfinite states fail the fixture.

The manual loop uses the task's normal position scheduler, explicit motor write and physics/scene updates every 1.25 ms. It omits RL termination/autoreset so intentional overlap cannot erase the negative-control evidence. It is not a standing, workspace or training qualification. Both cases always report `pass=false`, `simulation_training_admission=false` and `hardware_admission=false`.

The composed USD reference audit checks all 12 passive mimic joints in each clone and requires each reference to resolve to that clone's corresponding lever joint. This checks the actual composed USD paths; it is not direct introspection of PhysX's internal constraint graph.

Run each case in a **fresh, root-owned, resource-guarded container/process**. The script does not launch containers. Mount the fixture separately from the frozen source and provide `--source-dir` explicitly. Use separate, new output directories:

```sh
/workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/overlap_fixture/live_robot_pair.py --source-dir /workspace/hexapod \
  --case filtered --physics-steps 256 --solver-multiplier 2 \
  --report /workspace/validation_artifacts/filtered/report.json --viz none --device cuda:0
```

Repeat in a separate process with `--case unfiltered_negative` and a new output path. The usual reviewed Kit telemetry exclusion argument may also be supplied through the launcher. Defaults are 256 physics steps (0.32 simulated seconds) and 128 position / 1 velocity solver iterations. The accepted step range is 32–512, divisible by 16.

After both runs, compare them without launching Isaac:

```sh
python live_robot_pair.py --compare filtered/report.json unfiltered_negative/report.json --report pair/report.json
```

Require `fixture_complete=true`, `control_expectation_met=true`, an empty error list and complete `trace.npz` evidence for each case, then `pair_control_success=true` from the comparison. Kit teardown can exit the process before normal Python return; a process exit code alone is insufficient. Reports are persisted before teardown and include production source identity, the independent fixture script hash, resolved runtime settings, actual initial positions, pair bindings, group changes and mimic references. The comparison rejects differences in physical runtime settings beyond the explicit collision-isolation override.

SDK integration follows the installed source at `source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`: native view creation at lines 321–329, force matrices at 381–383 and contact-data/count buffers at 427–443. `DirectRLEnv.reset` at lines 365–370 writes the reset state and forwards kinematics. The live runs still need to confirm those native interfaces are usable for the exact pair fixture.

Local checks:

```sh
uv run python -m unittest discover \
  -s artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/sdk_collision_isolation \
  -p 'test_live_robot_pair.py' -v
```
