# Installed SDK collision-isolation review — 2026-09-05

The inspected GPU task does **not** enable automatic environment-ID collision
filtering. Co-locating its worlds is not justified by `replicate_physics=True` or
the scene's `envIdInBoundsBitCount=4` attribute. Explicit USD collision groups
and a live isolation test are needed before using overlapping origins.

This is a read-only source finding, not a GPU experiment. No production source,
asset, acceptance bound, mass, gain, or shared coordination file was changed by
this review. `capture.py` imports only the Python standard library; it reads the
installed files and executes the actual cloner callback against inert stubs.

## Evidence and call chain

`sdk_source_evidence.json` records full-file SHA-256 values and numbered excerpts
from the installed `isaac-lab-base` container, captured at
**2026-09-05 20:37:35.911021 UTC**. `task_source_evidence.json` records the matching
frozen task files in
`/home/orionh/HEXAPOD_runs/mkii_placement_diagnostics_v1/source`, source commit
`d6d5863`. Later task fixes are outside this evidence snapshot.

1. `InteractiveScene` selects `isaaclab_physx.cloner.physx_replicate` for the
   PhysX backend. Its `clone_environments()` writes the bit-count attribute and
   calls that function. These are separate operations.
2. The installed backend function hardcodes **`useEnvIds=False`**, with a TODO
   explaining its heterogeneous-scene limitation and dependence on USD groups.
   Its docstring's claim that complete rows enable IDs disagrees with the
   executable line. Replaying the actual callback with one homogeneous source
   covering all **32** environments produced 31 clones and `useEnvIds=False`
   for both `device="cpu"` and `device="cuda:0"`. This checks Python call behavior,
   not the implementation or operation of the native replicator.
3. `InteractiveScene` automatically creates collision groups only inside the
   `has_scene_cfg_entities` branch. The frozen task uses an ordinary
   `InteractiveSceneCfg` and constructs its robot, sensors and terrain afterward
   in `DirectRLEnv._setup_scene()`. That automatic branch is not taken. Its
   explicit `scene.filter_collisions()` call is guarded by `device == "cpu"`.
   No later group-authoring call appears in the inspected setup chain.
4. The terrain's **`collision_group=-1` is an Isaac Lab configuration convention**:
   automatic scene setup translates it into the global collision-group list.
   The installed `TerrainImporter` itself has no `collision_group` reference.
   This value is not evidence that the native PhysX replicator assigned the
   ground a particular environment ID. With replication IDs disabled, the
   proposed correction uses explicit global-group membership instead.
5. Foot sensors resolve one named tibia body per environment, create a contact
   view against the shared ground expression, and reshape backend force data
   across environments. This is compatible with explicit group filtering, but
   sensing does not create collision isolation. Ground-only foot matrices also
   cannot rule out contacts against another robot; a live pair audit is needed.

The inspected relevant file hashes include:

| Installed file | SHA-256 |
| --- | --- |
| `isaaclab/scene/interactive_scene.py` | `caf1c0646707d6c5a3c28a81020cab741987d96e74f3e43290f0584e221afe40` |
| `isaaclab_physx/cloner/physx_replicate.py` | `a0e7eb7f47e45ce27351002a0ef6c137ddb3c1c4a20a5efcbbd389c3ff5a426c` |
| `isaaclab/cloner/cloner_utils.py` | `80cdeda2fe6dc10391ac799cde36dc4eb9167e83613758bc497b3f08f148607a` |
| `isaaclab_physx/sensors/contact_sensor/contact_sensor.py` | `09507db17049bcadd6a48f5992bc20f79b019356c13f1ba3ef1527b2e4188b2c` |

Full absolute paths and the remaining hashes are in the JSON. The installed
Isaac Sim `isaacsim.core.cloner.Cloner` has another implementation that forwards
an `enable_env_ids` argument, but this task's Isaac Lab chain does not call it.

## Explicit group topology to verify

The installed `scene.filter_collisions(global_prim_paths=["/World/ground"])`
uses `scene.physics_scene_path`, discovered from the actual stage. It authors:

- Physics scene attribute `physxScene:invertCollisionGroupFilter = True`.
- `/World/collisions/group{i}` for each complete environment root, with type
  `PhysicsCollisionGroup`, applied `CollectionAPI:colliders`, expansion rule
  `expandPrims`, and exactly `/World/envs/env_{i}` in its includes relation.
- Each environment's `physics:filteredGroups` contains itself and
  `/World/collisions/global_group`, with no other environment group. Because
  filtering is inverted, these relationships allow the listed interactions.
- The global group's includes relation contains the shared terrain root;
  its allowed groups are itself and every environment group, reciprocally.

Verify all group names, includes/excludes, expansion rules, relationships and
actual collision-shape coverage, including environments **16–31**. Unexpected
extra groups or cross-environment allowlinks must fail. The helper returns
early if `/World/collisions` already exists, so calling it is not sufficient
without verifying the result. The existing articulation's self-collision
setting still applies within each environment.

NVIDIA documents environment-ID filtering as an automatic native facility and
mentions co-located environments as a possible optimization. Its bounds bit
count controls GPU broadphase encoding; the documented range 1–16 describes
that encoding, not a limit of sixteen environments. Those capabilities do not
override this installed caller's explicit `False` argument. [NVIDIA PhysX
replicator API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.IPhysxReplicator.replicate)

## Minimum proposed live proof, before co-origin training

Run only after the separate origin-versus-translation experiment and an
isolated source release. Preserve source, model, gains, timing, solver and
original physical bounds. These short tests establish isolation; they do not
replace standing, driven-motion or solver-convergence qualification.

1. **Topology and binding inspection before stepping.** Record the resolved
   physics backend, both collision attributes, all group relationships, shape
   membership, articulation paths and sensor body/filter paths. Each sensor
   must bind the intended environment row and the same single shared ground.
   Record the origin coordinates, not merely requested spacing.
2. **Two-robot positive control with filtering.** Use identical XY origins.
   Start robot 0 at its normal reset and robot 1 high enough that its downward
   path would intersect robot 0. Keep robot 0's command and targets fixed;
   let robot 1 descend and apply one recorded small intervention to robot 1
   only. Robot 1 must contact the global ground normally while passing through
   robot 0 without a cross-robot contact pair. Record every substep's named
   pair contacts, body poses/velocities, applied torques and each robot's ground
   forces. Robot 0's trajectory should remain within the numerical variation
   measured by a matched, unperturbed reference run; do not assume bitwise
   equality between differently sized GPU scenes.
3. **Unfiltered negative control.** Repeat that exact initial geometry and
   intervention with cross-world filtering deliberately absent in the
   diagnostic only. Require observed cross-robot contact pairs and a measurable
   response. If no interaction occurs, the test geometry or contact tracing
   was ineffective; absence of contacts in the filtered run proves little.
   A zero net force alone is insufficient because opposing contacts can cancel.
4. **32-environment smoke test.** After the two-robot controls pass, inspect
   every group's resolved members and all contact-pair environment identities
   in a short co-origin run. Individually perturb environments 0, 16 and 31;
   check the intended sensor rows and independent responses. Require no
   cross-environment pairs, no missing ground interactions, and no contact
   buffer overflow. Continue enforcing the original geometric and motor bounds.

If the available GPU contact API cannot expose sufficient pair identity,
add diagnostic pair-specific contact views or another verified pair query;
the existing six ground-only force matrices are insufficient evidence. Keep
ray-casting and rendered navigation sensors as a separate future isolation
check: physical collision groups do not themselves prove that cameras or
LiDAR omit another co-located world's visuals.

## Reproduction

From the repository root, with the Spark free of a running supervised phase:

```sh
ssh -o ControlPath=/tmp/hexapod_fourbar_translation.sock orionh@100.82.166.9 \
  'docker run --rm -i --runtime=runc --network=none --entrypoint /bin/bash isaac-lab-base -c "/isaac-sim/kit/python/bin/python3 -"' \
  < artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/sdk_collision_isolation/capture.py \
  > /tmp/hexapod_sdk_collision_evidence_new.json
```

Compare source hashes before comparing findings. Run this reader between
supervised phases: a transient foreign Docker container disappearing between
the supervisor's inventory and inspect can abort older launchers. This review's
containers completed and were removed; no GPU workload was launched.
