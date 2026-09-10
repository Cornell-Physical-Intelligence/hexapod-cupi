The CPU adapters are correct for the scope their thirteen tests cover. I found no demonstrated correctness defect in the frozen course, observation, or support-query code. I did find one latent, reproducible defect in the phase3 delayed-stream reuse path, several contract labels that are declared but never enforced, and a clock representation that will become fragile once timestamps come from an accumulated simulator clock. The real blockers are unimplemented bridges: scene-origin synchronization, terrain-relative dones and rewards, a batched map product, and the actor schema that cannot exist before actor selection.

My recommendation up front: build the atlas/course runtime and the pure-function terrain reward/done module now, both CPU-only and actor-independent, then spend two short GPU slots on a one-course and a two-course symmetry gate. Everything else waits on the Stage 2 actor decision.

## Maturity and stale claims

| Capability | File and function | Evidence state | Missing |
|---|---|---|---|
| Exact triangle oracle | `inputs/isaaclab/hexapod_terrain/support_queries.py` `query_local` L116, `query_world` L176, `relative_base_height` L214 | CPU tests; CPU smoke of 131,072 points per README L220 | Device timing; the chunk loop at L137 and the `.any()` at L133 sync per call |
| Course placement | `.../observation_comparison_cpu_001/frozen/course_queries.py` `CourseQueries` L26, `reset_rows` L71, `course_to_world` L84 | 13 tests pass for root and an independent reviewer, STATUS L57 | No scene-origin store sync, no USD placement, per-row API |
| Map product and modes | `.../frozen/observation_modes.py` `packet` L98, `enqueue` L81, `reset_rows` L69 | Same 13 tests | Batched form; provenance and registration labels unenforced |
| Map fusion and timing | `inputs/tools/perception_replay.py` `LocalHeightMap.integrate` L244, `local_patch` L279, `transform_timed_points` L132, `PoseTimeline.at` L87 | Synthetic CPU fixture only | Per-cell Python loops; float-second clock; no batched port |
| Channels and lease | `inputs/tools/terrain_readiness.py` `terrain_channels` L34 | CPU | Integer-tick clock |
| Fixture geometry, 30 originals and 48 derivatives | `terrain_readiness.py` `terrain_mesh` L93; `prepare_curriculum.py` `derive_mild_fixture` L45 | Actual Isaac ray, PhysX and probe admission: `actual_review/report.json` L9 and L17 to L25; originals per `hexapod_terrain/README.md` L25 | Any robot on the derivatives; atlas placement |
| Reset catalog, 1,536 rows | `prepare_curriculum.py` `validate_reset_geometry` L97, `select_reset_records` L125 | CPU geometry only; zero joint noise; old plan hash | Rebinding to the selected actor reset distribution; runtime quaternion-convention readback. The curated copy holds two rows, so nothing here uses it as data |
| Full-C entry standing | `inputs/isaaclab/hexapod_terrain/fixture_adapter.py` `adapt_flat_cfg_for_fixture_smoke` L75 | Terrain005 pass reported at STATUS L59; that artifact is not among my inputs, so I treat it as reported, not verified | Standing on any derivative; any walking |
| Flat task seams | `inputs/tools/omni_flat_env.py` `_get_observations` L75, `_get_rewards` L95, `_reset_idx` L162 | Stage 2 pilots running, nothing admitted | Terrain-relative height at L116 and vertical velocity at L113; inherited world-Z done |
| Sensor transport emulation | `inputs/isaaclab/hexapod_phase3/sensor_model.py` `_DelayedStream.read` L263, `capture_lidar` L427 | Pure torch; smoke only on the archived mock | C-asset mesh targets, 400 Hz clock, drain API; `sensor_cfg.py` is mock-only at L32 and L51 |
| Footprint lease checker | `.../footprint_reacquisition_001/owner/footprint_checker.py` `query` L128 | 16 CPU tests; replay on flat actual004 | Executable commit/abort; contact-patch model |
| Actor and critic schema | none | Widths proposed at `comparison_plan.json` L75 to L78 | Blocked on actor selection |
| Hardware sensing | none | Owned Mid-360 and D455, no bags | Everything |

**Stale claims resolved by lineage.** PLAN.md L291, QUALIFICATION_PLAN.md L11, the immutable catalog status at `terrain_catalog.json` L3 and `qualification_draft.json` L805 all say the 48 derivatives lack Isaac admission. The later terminal review dated 2026-09-10 records the actual pass with zero contact-truncation warnings. The fixture admission is real; robot admission on derivatives is not, and the report says so at L65 and L68.

TRAINING.md L364 says corrected contact capacity still needs fresh simulator evidence, and the terrain README L334 describes attempt 004 failing its contact gate. STATUS L59 says terrain005 qualifies entry standing. The later statement wins for original-ramp entry standing only.

TRAINING.md L64 to L78 describes a 66-value, 200 Hz mock contract. STATUS L8 gives the C lineage: 315/318 values, 20 ms controls, eight 2.5 ms substeps. The phase3 scene config hardcodes 200 Hz and mock prim paths, so only the pure-torch transport model is reusable.

CPU_READINESS.json L19 says the independent review was deferred; STATUS L57 says both parties passed all tests. Later wins.

## Design: task, seams, sensor path

**Interfaces that can be coded now.** Everything below is actor-independent except the packer. Typed shapes use N environments, integer subticks at 2.5 ms, and ticks of eight subticks.

```python
# isaaclab/hexapod_terrain/course_runtime.py   owner: terrain runtime
@dataclass(frozen=True)
class AtlasManifest:                 # written once per campaign, hashed
    catalog_sha256: str
    placements: list[Placement]      # Placement(fixture_id, origin_w=(x,y,z), yaw_w)
    atlas_usda_sha256: str           # one combined static Mesh authored from placements
    env_to_placement: list[int]      # length N, fixed for the whole campaign

class CoursePlacementDevice:         # same math as CourseQueries, device tensors
    fixture_index: Long[N]; origin_w: F32[N,3]; yaw_w: F32[N]; cos: F32[N]; sin: F32[N]
    course_epoch: Long[N]            # stays 0 in campaign 1; placements never move
    def course_to_world(p_c: F32[N,P,3]) -> F32[N,P,3]
    def world_to_course(p_w: F32[N,P,3]) -> F32[N,P,2]
    def assert_env_origins(scene_env_origins: F32[N,3])  # exact equality, called at reset

# reset seam   owner: OmniTerrainEnv._reset_idx
class ResetBank:                     # from the full 1,536-row curriculum.json, split+level filtered
    root_pos_course: F32[R,3]; body_yaw_course: F32[R]; row_id: list[str]
    def draw(env_ids, generator) -> (pos_w: F32[k,3], quat_installed: F32[k,4], row_ids)
    # yaw_w = yaw_w[env] + body_yaw_course; one function quat_wxyz_to_installed()
    # readback after write: heading_w(root_quat_w) == R(yaw_w) @ [0,-1,0] within 1e-4

# privileged per-control geometry   owner: terrain_rewards.py, pure torch, no Isaac import
@dataclass
class GeometryProbe:
    pad_w: F32[N,6,3]        # body_pos_w[foot] + R_foot @ pad_offset_link (from distal collision vertices)
    shaft_w: F32[N,6,S,3]    # S samples along each tibia shaft
    body_w: F32[N,B,3]       # B body corner points
    stance_ref_w: F32[N,6,3] # nominal stance foot XY rotated by current root yaw, z = root z
@dataclass
class TerrainState:          # outputs of the oracle; mode-independent
    pad_clear: F32[N,6]; pad_support: Bool[N,6]; shaft_clear_min: F32[N,6]; body_clear_min: F32[N]
    rel_h: F32[N]; rel_valid: Bool[N]; ref_normal: F32[N,3]; span: F32[N]; contained: Bool[N]

# map product   owner: map_runtime.py
class MapProduct:
    channels: F32[N,100,100,4]       # rows forward, columns left, gravity aligned, height minus registered plate z
    assembled_subtick: Long[N]; map_epoch: Long[N]
    provenance: Int8[N]              # 0 blind, 1 ideal_raster, 2 synthetic_corrupted, 3 sim_lidar, 4 sim_depth, 5 fused
    registration: int                # 0 oracle, 1 estimated_<drift_model_sha256>
class PrivilegedGeometryView:        # critic and reward only; the actor packer signature cannot accept it
    ideal_channels: F32[N,100,100,4]; eligible: Bool[N,100,100]; hazard: Bool[N,100,100]

# corrupted-mode delivery   owner: sensor_bridge.py
class FrameBatch:
    env_ids: Long[k]; capture_subtick: Long; receive_subtick: Long; map_epoch: Long[k]
    points_w: F32[k,M,3]; var_z: F32[k,M]; valid: Bool[k,M]; robot_hit: Bool[k,M]; provenance: int
```

Clock and epoch semantics, stated once so the coding agent does not reinterpret them:

```text
subtick: physics substep index since simulator start (2.5 ms); tick = subtick // 8
episode_subtick0[e]: set at reset of env e; map_epoch[e] += 1; history cleared for those rows only
frame integrable for env e iff receive <= now and capture >= episode_subtick0[e] and frame.map_epoch == map_epoch[e]
cell usable iff observed and now - capture <= 100 subticks and sqrt(var) <= 0.015 m and finite
order inside one control: 8 substeps -> release frames -> integrate -> patch at registration pose(now) -> channels -> actor
```

The lease of 100 subticks is exactly 0.25 s and never depends on float rounding.

**Ownership of each seam.**

| Seam | Owner | Writes | Reads |
|---|---|---|---|
| Atlas and placements | `course_runtime.py` | USD once, device tensors once | catalog, manifest |
| Reset | `OmniTerrainEnv._reset_idx` | root pose, history, map epoch, sensor stream reset | ResetBank, placements |
| Geometry probe and dones/rewards | `terrain_rewards.py` and the env | nothing persistent | body poses, oracle |
| Map | `map_runtime.py` | per-env map buffers | frames, registration pose |
| Actor packer | `terrain_actor_critic.py` | nothing | proprio history plus `MapProduct.channels` only |
| Critic packer | same module | nothing | proprio, true linear velocity, `PrivilegedGeometryView` |

Two practical constraints follow from the flat env. The inherited constructor calls reset before subclass attributes exist, which is why `_sample_commands` at L39 guards with `hasattr`. Any terrain state used in reset must be created before the parent constructor runs or guarded the same way. Second, the reward bookkeeping at L95 to L160 is one function returning a scalar, so the terrain env should reimplement it rather than edit the frozen flat file. The new terms live in a pure module with unit tests, and the flat file stays byte-identical.

Terrain-relative replacements for the two world-Z terms:

```python
height_term   = where(rel_valid, (rel_h - reset_height_m)**2, 0)          # reset_height_m = 0.13653 from curriculum.json L38
vz_expected   = (v_xy_w * slope_xy).sum(-1)                               # slope_xy = -n_xy / n_z from ref_normal
vertical_term = (v_z_w - vz_expected)**2
swing_clear   = clamp(h_target - pad_clear, min=0) * ~distal_contact       # h_target proposed, not fixed
done_boundary = ~contained | ~rel_valid                                   # classified boundary, never fall
done_low      = rel_valid & (rel_h < 0.055)                               # replaces inherited world-Z check
```

The inherited low-height termination must be disabled by configuration or by a full override of `_get_dones`; I could not see `HexapodEnv`, so the coding agent must inspect it first. Keep base contact, gravity tilt, and the 5.5 N·m demand termination unchanged. The gravity tilt penalty at L115 stays as a deck objective on mild slopes; whether to add a slope allowance is decision D5 below.

**Adapter critique, demonstrated versus hypothesized.** The 100 by 100 sample centres, the forward-rows and left-columns axes, the upward normal flip at `support_queries.py` L163, the world normal rotation at L197, older-capture skipping at `perception_replay.py` L259, the exact lease boundary, and selective reset isolation all check out by reading and by the tests at `test_adapters.py` L100 to L188. Eligibility and usability are correctly separate, and eligibility never reaches the channels, which is by design. I have no demonstrated correctness defect to report in the three frozen files.

Latent defect, reproducible without a simulator, in the reuse path the plan names:

```python
# sensor_model.py L263-289: an out-of-order release regresses the "latest" frame and its capture time
cfg = TransportModelCfg(sample_period_s=.05, latency_s=.10, latency_jitter_s=.05, stale_after_s=.30)
s = _DelayedStream(cfg, seed=0)
s._jitter_rng.uniform = lambda a, b: b      # first frame: latency .15, ready 1.15
s.enqueue({"h": torch.tensor([[1.]])}, 1.00, torch.tensor([True]))
s._jitter_rng.uniform = lambda a, b: a      # second frame: latency .05, ready 1.10
s.enqueue({"h": torch.tensor([[2.]])}, 1.05, torch.tensor([True]))
f = s.read(1.20)   # values["h"] == 1., capture_time_s == 1.00: the older frame wins
# patch: inside the pop loop, skip per env where queued.capture_time_s < self._capture_time_s
```

The default configs cannot reach this because jitter is below half the period, but nothing enforces that. A second bridge defect in the same function: multiple frames released in one call collapse into the last one, which is right for a latest-sample observation and wrong for map fusion. Add a `drain(now_s)` returning every released frame.

Contract gaps that are accepted silently today:

- **Provenance mixing.** `enqueue` at `observation_modes.py` L83 accepts any member of the set per packet, so one map can hold synthetic and simulated-lidar points under the single mode label at L112. Freeze one allowed provenance per instance at construction.
- **Registration is a string.** The `registration` label at L37 is never checked against the pose passed to `packet`. Make `packet` take a pose sample carrying a source identifier and a capture subtick, and reject mismatches.
- **Map extent never checked against the course.** `reset_rows` at L69 accepts arbitrary map origins; a map that misses the course yields an all-unusable actor input with no error. Assert that the transformed course bounds lie inside the map.
- **Float clock at the lease boundary.** `terrain_channels` L56 compares `age <= 0.25` on floats. Hypothesis: an accumulated simulator clock can produce an age of 0.2500000000000003 at the nominal boundary. The subtick clock above removes the question.
- **Orthonormality tolerance.** `validate_transform` L36 rejects at 1e-7. Hypothesis: a float32 root quaternion converted without float64 renormalization can fail it. Renormalize in float64 before building the pose.
- **Blind mode raises on degenerate heading.** `packet` calls `local_patch` unconditionally, and L284 raises when body forward is vertical. That is unreachable before the gravity termination fires, but the device version should derive yaw from the quaternion and mark the patch invalid instead.
- **Origin aliasing.** `LocalHeightMap.__init__` L236 keeps a view of the adapter's origin row. Harmless today because the old map is discarded at reset; copy the array anyway.

Non-batchable by construction, so the device port is a re-implementation checked against these oracles: the per-row API and the four-tensor `snapshot()` clone on every `_row` call, the per-unique-key scan in `integrate` at L257, and the per-unique-timestamp loop in `transform_timed_points` at L154. To make exact parity possible, add a `reduction="mean"` option to `integrate`, since a batched port can use scatter-mean and span but not median. The default stays median.

**Causal sensor and latency path.** Each stage below names its owner and what it may see.

1. **Capture.** The ray caster or depth annotator samples at the declared cadence on a tick boundary. `capture_lidar` stores noisy ranges and validity with the capture subtick. Provenance is `sim_lidar` or `sim_depth`; the synthetic tier instead samples the ideal raster and applies declared dropout, noise, a foot-shadow mask, and a frame delay, labelled `synthetic_corrupted`.
2. **Delivery.** The delayed stream releases frames at the receive subtick. Whole frames are dropped for a reset env, and per-point straddling of a reset is trimmed rather than rejecting the frame.
3. **Registration.** Ranges times fixed ray directions give sensor-frame points. The pose timeline supplies world-from-body at the capture subtick, which is already known by the receive subtick, so this is causal. Under oracle registration the timeline holds truth; under estimated registration it holds truth plus a declared drift model with its hash in the label.
4. **Robot-return masking.** A hit on a robot leaf mesh sets `robot_hit` and is excluded. The ground behind it is simply unobserved. The mesh expressions in `sensor_cfg.py` L51 are for the mock asset and must be re-derived for the C prim tree.
5. **Fusion.** Latest capture wins per cell, no interpolation, no fill, variance from ray geometry plus pose sigma plus in-cell span, exactly as `integrate` does.
6. **Channels.** Height relative to the registered plate, usable mask, age, sigma, with the subtick lease.
7. **Cache.** The map persists between frames; the patch is re-extracted every tick at the registration pose for that tick, so age keeps counting between frames.
8. **Footprint and support queries.** Reward and termination use the exact oracle at the probe points in every arm. The footprint checker semantics apply to the supervisor path, which is not part of campaign 1.

The three tiers stay separate in labels and in scoring: ideal-map corruption, ray-sensor emulation, and hardware. Only the first two have code, and the hardware tier has nothing today. The D455 minimum depth at `sensor_cfg.py` L80 exceeds the distance from a body mount to this robot's feet, so near-foot sensing with owned hardware is an open question, not a modelling detail.

To test localization noise without leaking privileged state or changing the plant: hold physics, actuator limits, reset rows, command process, reward, critic inputs, and termination identical across arms. Change only the pose timeline used for point registration and patch extraction. The same noisy ranges are re-registered under truth and under the drift model, so the ablation isolates registration. A static leak audit of the actor packer should fail the build if any of root position, root velocity, contact force, friction, fixture index, course origin, `geometry_truth`, or `sample_world_xy` is reachable from its inputs.

Reconciling the long swing with the lease, without extending it or promising observations:

- **The conflict is lineage-specific.** The 2 s swing and 5.54 s stop belong to the reference branch per STATUS L17 and the footprint README L22. The direct actor's airtime term at `omni_flat_env.py` L127 rewards swings between roughly 0.2 and 0.45 s, so its swings sit near the lease already.
- **Decompose the stop.** Only feet in flight need map validity, and only for their residual swing. Planted feet are contact-supported. Settling to quiet is contact-only, so stop-to-quiet latency is not a map problem.
- **Verify at touchdown instead of promising.** Score each touchdown against the usable mask at the touchdown subtick. That is causal in every arm and becomes the per-arm metric "fraction of touchdowns on cells usable at touchdown".
- **Supported abort.** If the target cell goes stale mid-swing, return the foot to its own liftoff contact point. Support there was measured; the only added assumption is static terrain with bounded drift over one swing. This is a versioned assumption for review, decision D3, and it extends nothing.
- **Visibility is the real requirement.** Reacquisition during swing needs the upcoming foothold in view. The six-camera replay reports 82.1% for the active region, so the learnable behaviour is foothold selection inside the observed sector, and the remainder is mount geometry.

## Execution: order, gates, curriculum, backlog, decisions

**Implementation order.** Each task names its dependency, its minimal test, and its gate.

| Task | Depends on | Minimal test | Gate |
|---|---|---|---|
| T0 atlas builder and manifest in `course_runtime.py`, combined static mesh from placements | catalog, `mesh_usda`, `CourseQueries` | atlas triangles equal transformed fixture triangles; `query_world` on placements equals `vertical_surface_heights` on the combined mesh at random points | CPU |
| T1 `terrain_rewards.py` pure functions and `GeometryProbe` | oracle | ramp at nominal height gives zero height term; pit sample invalidates; negative clearance penalized; boundary classified | CPU |
| T2 `ResetBank` and quaternion conversion | full curriculum.json | WXYZ to installed order round trip; heading check math | CPU |
| T3 `map_runtime.py` batched ideal raster, corrupted integration, patch | T0, CPU adapter with mean reduction | parity against `TerrainObservationModes` on three envs with identical packets; report max error | CPU |
| G-A one course, one env, zero action | T0 to T3, fresh 32 by 1,000 flat standing | ray caster and PhysX heights at 64 probes equal oracle; foot contact points give clearance near zero; contact log complete | GPU, minutes |
| G-B two courses, translated and rotated, zero action | G-A | root pose in course frame equals reset row; contact statistics identical across the two envs; `assert_env_origins` exact | GPU, minutes |
| G-C 32 envs, four dev fixtures, blind actor at zero then constant command, no learning | G-B | no spurious termination; complete contact data; per-control query time logged | GPU, short |
| T4 actor migration and schema | actor selection | migrated actor in blind mode reproduces the original actions on the flat evaluator | CPU then GPU |
| P0 pilot | T4, G-C | see the matrix below | GPU |

Gate G-B is the cheapest decisive evidence in the whole programme. Any mismatch between the USD transform, the device placement, and the scene origin store shows up as asymmetric contact between two envs standing on identical geometry.

Dead-end gates to reject, and gates that must stay:

- **Waiting on the joint-rate bias** blocks nothing in terrain integration. Only energy and cost-of-transport claims depend on it, and those stay provisional as PLAN L385 already says.
- **Standing on all 48 derivatives** is 48 GPU runs for one fact. The start pad is flat and identical within a family, so standing on one continuous pad and one box pad covers it. Contact on the sloped part only comes from walking.
- **Both-direction start admission before any terrain PPO** blocks ramps forever, because no fixture has a high flat pad. Steps and ridges give both senses within one episode by crossing then reversing. Ramps need decision D5.
- **The 14,800-episode matrix** is post-training qualification and must not run before a pilot.
- **Necessary gates:** fresh flat standing, atlas parity, two-course symmetry, contact completeness, flat parity of the migrated actor, and the packer leak audit.

**Curriculum and held-out matrix.** Campaign 1 is a development discriminator, not qualification.

| Element | Value | Status |
|---|---|---|
| Fixtures | mild0 ramp 1103, mild0 smooth_rough 1103, mild2 step 1103, mild2 ridge 1103 | fixed by comparison plan |
| Regression | same families, seeds 7103 and 8209 | fixed, never final |
| Placements | 8 per fixture, yaws 0, 90, 180, 270 degrees, two translations each; fixed per env | proposed |
| Starts | reset rows from the full catalog in heading bins 0, plus or minus 90, 180 degrees, width 10 degrees | proposed |
| Motions | forward, reverse, left, right, turn left, turn right, arc plus, arc minus, walk to stop, turn to stop | fixed list plus two stops |
| Speeds | low 0.05 m/s; upper 0.10 m/s at level 0 | proposed; bind to measured flat capability |
| Episode | 30 s, scored on progress into the terrain region and stop quality | proposed |
| Arms | blind, ideal, synthetic corrupted, all with oracle registration | fixed |
| Budget | 1,024 envs, 24 controls, 50 updates per arm, one seed; then 300 updates, three seeds, only for arms with signal | proposed |
| Flat replay | 25% | from curriculum.json |

Split rules: every derivative of a source seed stays in its parent's split, the final layouts come from the reserved lists in `qualification_draft.json` and are not yet geometry, and any held-out layout used for tuning retires to development. Command sequences cannot be paired across arms during training because `sample_commands` draws from the global generator, so pair the terrain assignment, reset rows, and seeds, and pair the evaluation scripts exactly.

Fixed thresholds are the existing flat static, transition, quiet, and physical gates at `qualification_draft.json` L110 to L154, the 1.6 N·m applied cap, the 5.5 N·m demand termination, nonfoot contact, and contact completeness. Proposed thresholds for the pilot, to be frozen by root before launch: the relative height band, the swing clearance target, the boundary margin, and the progress success rule.

Promotion evidence from the pilot: a per-arm and per-case table with failures preserved; the paired ideal-minus-blind delta on terrain progress; corrupted between the two; the unchanged flat evaluator and quiet review on the same checkpoint at the common limiter; and the touchdown-observed metric per arm. If ideal does not beat blind on the same cases, the plant or reward is the limiter and perception work should not absorb GPU time.

**Ranked backlog.**

| Rank | Change | Seam | Benefit | Acceptance artifact | Unresolved |
|---|---|---|---|---|---|
| 1 | Atlas builder, manifest, placement device tensors | `course_runtime.py` | unblocks every GPU gate | `atlas_manifest.json`, CPU parity log | D1 |
| 2 | `OmniTerrainEnv` with terrain-relative dones and rewards, fixed placements, reset readback | env subclass plus `terrain_rewards.py` | removes world-Z terms; G-A and G-B | G-B JSON with symmetric contact stats | D5 |
| 3 | Batched map runtime, three modes, CPU mean-reduction parity option | `map_runtime.py`, `integrate` | real observation product at scale | parity report with max error | none |
| 4 | Actor and critic module with encoder, normalizer split, storage sizing, migration tool | `terrain_actor_critic.py` | training possible | flat-parity log of the migrated actor | D2, actor selection |
| 5 | Sensor bridge: drain API, out-of-order guard, subtick clock, C-asset mesh expressions | `sensor_model.py` and a new bridge | ray-emulation tier | unit tests plus one-env emulation smoke | none |
| 6 | Evaluation script reusing flat evaluators, touchdown-observed metric | `evaluate_terrain_comparison.py` | informative pilot | per-mode and per-case JSON | D4 |
| 7 | Provenance and registration enforcement in a new adapter version | `observation_modes.py` successor | closes label gaps | new tests, frozen copy untouched | none |
| 8 | Descending-ramp geometry or slope-start admission | fixture pipeline | both crossing senses | fixture admission report | D5 |

**Decisions for partner review.**

- **D1 atlas form.** One combined static mesh per campaign with fixed placements per env, versus per-course referenced meshes. I recommend the combined mesh: one collider path, unchanged contact filters, one buffer, and a cheap re-run of the existing ray and PhysX parity on the atlas.
- **D2 actor map input.** The full 100 by 100 patch in the observation vector costs roughly 4 GB per rollout buffer copy at 1,024 envs by inference, and the map block should bypass the empirical normalizer. A 60 by 60 centre crop is the alternative. This waits for actor selection.
- **D3 swing and lease contract.** Adopt touchdown-time verification plus the liftoff-point supported abort under a versioned static-terrain assumption, or keep the supervisor path blocked.
- **D4 reward parity across arms.** Keep the reward mode-independent and report "unobserved touchdown" as a metric only, or allow a caution penalty in the corrupted arm and accept that the arms no longer share a reward.
- **D5 slopes.** Choose new fixture geometry with a high pad versus slope-start admission for descending ramps, and decide whether the gravity tilt penalty gets a slope allowance.

Implement backlog items 1 through 3 first, all CPU and actor-independent, while the Stage 2 pilots hold the GPU. Then take G-A and G-B in the first free slot. They need minutes, and a failure there is far cheaper to find now than inside a training run.
