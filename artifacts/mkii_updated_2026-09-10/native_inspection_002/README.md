# Canonical native inspection 002: imported identity passes, SDF query fails

The detailed direct-drive robot loaded in Isaac Sim6.0.1. Native checks passed for19 body identities,18 named joints, masses/full inertia tensors, COM frames, joint limits, initial link/coordinate FK, zero-gravity400Hz scene settings and zero drive gains. Measured mass was7.466088220kg, matching the selected7.466088235kg model within float precision.

The inspection then failed before its eight explicit steps because the installed compiled SDF factory accepts a string pattern, while its Python documentation advertises lists too. Passing the153-path list raised `TypeError: create_sdf_shape_view(): incompatible function arguments`. No SDF view/count/distance/contact result was produced. This is useful partial native evidence within an authentic failed inspection, not completed physics or training admission.

## Exact scope and evidence

The source fixed001's private-module lookup by binding the installed public tensor110.1.13 factory and package hashes. The canonical nine-file asset, inertia/limits/collision detail, zero-gravity/no-floor/no-drive setup and eight-step budget remained unchanged. The separately declared SDF initialization contract uses returned initialization plus valid native views, with an unavailable legacy cooking queue counter recorded as unknown. Actual SDF validation was not reached.

Root passed19 focused inspector tests,21 updated guard tests and actual Spark Python3.12.3 host setup before dispatch. The source producer also records12 host tests and both exact-input preflights. Native run invocation`a16c19ec37164f5194f726fa7c755ef9` is terminal with both owned identifiers absent. Pause002 restored exactly at1789072420.7756908. All21 raw payloads/168,110bytes and two complete hash passes are verified by `terminal/audit.json`.

The first readable sample after SDK warmup contains zero joint angles and reported rates. It does not observe warmup motion, prove stationary contacts or certify the later driven model. Native maximum drive effort reads zero with no drive gains; this is not an identified actuator envelope. TGS defaults report external-force integration disabled and a noisy-velocity warning; later actuator/measurement admission must choose and verify its own explicit physics settings.

## Reproduction and successor

Run `python3 -B -S verify_bundle.py` in this directory. All frozen preparation files, actual host setup, failed native readback/logs and terminal audit are included. No old task, checkpoint or motor adapter was loaded.

The successor changes only the SDF factory argument to one supported string pattern. It must still reject any missing or extra path against the exact153 authored instances. It keeps physical tolerances and all model bytes. Completed import remains a prerequisite to actuator/coordinate tests, supported contact/standing, learning smoke and unchanged Stage2 quality gates.
