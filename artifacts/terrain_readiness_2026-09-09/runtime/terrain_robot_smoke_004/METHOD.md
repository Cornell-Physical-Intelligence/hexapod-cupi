# Terrain standing attempt004

This source is frozen attempt003 plus a one-line rotation correction and source
provenance. The installed Isaac Lab `AssetBaseCfg.InitialStateCfg.rot` uses
XYZW. The adapter now supplies `(0,0,sin(yaw/2),cos(yaw/2))`, which is yaw about
world+Z. The former tuple used the WXYZ convention and produced a roll when
interpreted by this runtime. An independent SciPy test of the exact assigned
expression at five headings confirms gravity remains vertical and body−Y
rotates into the expected horizontal heading. The parent-added portable
adapter regression passes as well.

Frozen source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_source_004`.
The661-file manifest SHA-256 is
`9fc04076a6397cc5377f6a02c3d9c1ffd5b18967a6d3d586dd178ae0d96c7e95`.
The exact original full-review plan, pinned16-file runtime, C asset, original30
fixtures and fixture attempt003 admission remain unchanged. Mesh schema and
deep-copy corrections from source003 are retained. No reward, termination,
torque, joint limit, geometry, inertia, solver or acceptance threshold changes.

Expected output: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_smoke_004`.
Expected unit: `hexapod-terrain-robot-smoke-004-20260909.service`.
Expected forecast pause017 includes exact-unit exit restoration and a separate
30-minute fallback. Each job owns both locks and has a10-minute deadline;
the service has a22-minute deadline. Immutable owned container IDs bound all
cleanup. Source stays read-only; flat admission writes only a separate per-run
asset copy, and terrain mounts that admitted copy read-only.

Dispatch follows the unchanged-orientation five-control pre-reset diagnostic
and verified cleanup. A fresh32-environment×1000-control flat admission remains
mandatory before one C robot×1000 terrain-standing controls. The fresh flat gate passed. Terrain completed1000 controls but the original
standing gate rejected: zero terminations/truncations and requested saturation,
settled maximum computed torque0.4984625N*m, minimum/mean root height
0.1291335/0.1295423m, zero base contacts, but800 nonfoot contact environment
steps and minimum5 distal support contacts. These contact classifications are
not complete evidence: PhysX reported contact data truncated at capacity8.
The flat log had no such warnings. `results/contact_data_completeness.json`
records exact counts, log hashes, first/last warnings and affected sensor setup;
full logs are preserved. No standing/traversal/derived-fixture/sensing admission.

`results/post_run_verification.json` checks661 source files and550 admitted
asset files unchanged, no source extras, both exact containers absent, CUDA
empty, both shared locks free, and both forecast timers restored. GPU ownership
was explicitly handed to root after this verification. A capacity correction
and an explicit overflow-rejection gate may be prepared on CPU, but no further
GPU job is launched by this preparation. This remains an original flat-entry
standing integration test; no production hardware or policy is qualified.
