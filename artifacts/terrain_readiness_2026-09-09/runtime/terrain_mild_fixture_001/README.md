All 48 mild-curriculum fixtures passed the bounded Isaac geometry/contact screen on 2026-09-10. This is a terrain-fixture preparation milestone; it does not qualify the full C robot, terrain walking, sensors or Stage 3.

The result covers 3,888 RayCaster rays, 192 PhysX queries, 144 sphere probes and 48 required outside-mesh misses. Maximum ray height error was 0.1722 µm and maximum absolute sphere-bottom gap 18.72 µm. All probes had contact for the final 40 steps; the full runtime log reported no contact/friction data truncation. Probe integration was 500 × 5 ms after one initial 5 ms query step. Nominal friction was 1 and restitution 0.

The family/split coverage is 18 smooth-rough, 18 ramp, six step and six ridge fixtures: 32 training and 16 held-out. These 48 do not include pits. Original 30-fixture and full-C entry-hold results remain separate; frozen catalog bytes were not updated.

`raw/` preserves every one of the 11 remotely hashed terminal payloads. `remote_audit.json` records unchanged 105-file source and 97-file catalog/geometry identities, exact owned ID/name absence, successful inactive unit and pause045 restoration. This is historical cleanup evidence, not a claim that a later GPU slot is free. `actual_review/` preserves the exact source-bound gate replay.

`preparation/` supplies compact source reconstruction from published Git geometry and exact overlays. `guard/` preserves the reviewed outer launcher and restoration code. Every nested frozen payload remains byte-identical.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B verify_payload.py` from this directory for portable hash, cleanup and recorded-result verification. No GPU is used.
