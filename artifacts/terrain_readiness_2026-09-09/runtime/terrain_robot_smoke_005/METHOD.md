# Method and evidence scope

Run005 used frozen source `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_source_005`
and output `terrain_robot_smoke_005`, with unit
`hexapod-terrain-robot-smoke-005-20260910.service` and forecasting pause020.
The source descends from published `9d6107794c8d329dded8997b6c88f7ea0d8695dd`
with the separately recorded Mesh, deep-copy, XYZW and contact-data fixes.
The current 662-file manifest is authoritative; `source_origin.json` retains
the historical published-files map as baseline provenance.

The unchanged original plan SHA-256 is
`6a234f2b1ffd4f30ba4470b5ebb806cb2964fa0cdc85f6e4433976749c98af2a`.
The unchanged C URDF SHA-256 is
`e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c`.
The pinned 16-file study runtime tree SHA-256 is
`abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280`.
The runtime reports 19 bodies, 18 joints and six feet, with total mass
8.26081134 kg and maximum absolute inertia-tensor error 1.62105e−9 kg·m².
The 1.6 N·m study cap and all original acceptance thresholds remained unchanged.

The [frozen host](frozen_source/tools/launch_terrain_robot_smoke_spark.py)
first made an independent writable 550-file asset copy and required fresh
32 × 1000 flat admission. The admitted bytes were then mounted read-only for
1 × 1000 terrain controls. Both loops used zero commands/actions with no policy,
and excluded the first 200 controls from settled torque/contact measurements.
The complete sequence took approximately 157 seconds including startup.

Terrain used only the originally admitted `train_ramp_1103` mesh, SHA-256
`c06b3aa6caa5106d1e26a80029e64a922215e28389d6a60880aeac6299d33d44`,
with fixture admission SHA-256
`90079bc52521cb8c38e08ffc80e012fe5145acc5704f44ca678862815a559b9c`.
Its typed collision Mesh remains at `/World/ground/terrain` without a ground
plane or convexification. Each full start footprint was checked before spawn.
The robot held the existing flat start at course X = −1 m with upright XYZW yaw;
it did not enter or traverse the ramp.

The [adapter](frozen_source/isaaclab/hexapod_terrain/fixture_adapter.py) raises
tracked point/friction data capacity to at least 128, preserving larger settings.
The [log audit](frozen_source/tools/terrain_contact_evidence.py) rejects any
reported incomplete contact/friction data, and the host waits for closed logs
before applying it to both phases. Zero such warnings, all six distal supports
and no shaft/coxa/femur/base contacts establish passing classification for this
specific smoke; capacity 128 is not a universal future-scene guarantee.

The minimal frozen source does not include the later shared quiet-telemetry
quaternion conversion. Neither standing path enables quiet or diagnostic
scoring, so no quiet-heading result or new velocity-policy admission is inferred.
The robot's spawn quaternion is already corrected to the installed SDK's XYZW
convention.

All downloaded raw results/restoration records match independent remote hashes.
The initial post-run checker incorrectly treated lowercase Docker `no such`
responses as a failed absence Boolean; the raw four inspect responses still
proved absence. That observation is preserved in
[postrun_verification_initial.json](postrun_verification_initial.json).
The corrected case-insensitive checker repeated after terminal and produced
[postrun_verification.json](postrun_verification.json). No runtime result or
frozen source bytes changed during this evidence review.

The [publication map](SHA256SUMS.json) covers every shipped file except itself;
[full_result_SHA256SUMS.json](full_result_SHA256SUMS.json) separately covers the
19 exact original run/restoration files. The source and 550 assets remain on
Spark with their exact manifests; selected executed source is included here.
