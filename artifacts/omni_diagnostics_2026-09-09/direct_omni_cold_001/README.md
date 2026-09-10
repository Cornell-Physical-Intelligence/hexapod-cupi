# Original direct-PPO cold baseline

The preserved 315/318-input policy completed a fresh standing admission and all 12 constant-command diagnostic cases, with four replicas per case and no terminations. It provides useful mean directional motion, but still has substantial joint motion, standing drift and requested torque saturation. **This is a completed baseline measurement, not Stage 2 qualification or training admission. No PPO updates were run.**

After each episode’s two-second settling interval, forward and reverse mean speeds were +0.08792 and −0.08747 m/s for ±0.10 requests. Left and right were +0.09460 and −0.08683 m/s. Pure turns averaged +0.19090 and −0.19762 rad/s for ±0.20 requests. Forward arcs under-tracked yaw at +0.13347 and −0.15257 rad/s.

Standing still had 0.01478 m/s mean left drift and 0.03789 m/s mean planar speed error. Across the cases, 8.91–11.41% of post-settle joint-control requests exceeded 1.6 N m; requested peaks were 5.776–7.054 N m. Applied torque remained clipped to approximately 1.6 N m. No moving-to-stop transition was tested.

The [complete scenario comparison](evidence/analysis_002/README.md) keeps the historical **0.03 rad/20 ms** result separate from this fresh **0.04 rad/20 ms** result. The checkpoint, other declared overrides, effective reward weights and diagnostic options match. Standing saturation was 5.75% historically and 10.70% in the fresh run; the target-rate setting is a software constraint, not a measured motor speed rating. This is not a paired statistical proof of causation.

## Evidence and provenance

- [Actual complete diagnostic](evidence/run/baseline/diagnostics.json) retains all 48 replicas’ scenario summaries, every joint, all/reset/post-settle windows and overlapping termination reasons.
- [Original saved trace](evidence/run/baseline/diagnostic_trace.npz) contains 600 controls for one of four replicas per scenario. It is byte-preserved; no resampling or truncation was used.
- [Root terminal audit](evidence/root_terminal_audit.json) verifies the complete 589-file cold source, 926-file supervisor, host, guard, contract and legacy runtime, the unchanged standing outputs, both exact container names and IDs absent, and pause 053 restoration at Unix time 1789052038.71282. The live unit invocation had been collected to an empty string; the audit preserves the matching historical journal identity rather than claiming it remained populated.
- [Frozen analysis](evidence/analysis_002/report.json), [read-only audit utility](audit_utility/README.md), and their original freeze manifests remain unchanged. The earlier analysis’s statement that the full root audit was separate is preserved; that later proof is now included here.
- [Historical 0.03 diagnostic](historical/diagnostics_003.json) is an exact small copy of the archived comparator already held by the baseline contract.
- [Preparation and frozen inputs](../direct_omni_cold_preparation_001/README.md) are a separate sibling bundle. Source trees, checkpoint weights and large duplicate traces are not repeated here.

The legacy `quaternion_world_wxyz` trace field contains raw SDK XYZW on this build. The analysis preserves that historical payload and does not derive transforms from the misleading label. Endpoint-reported joint rates and angle differences over a control interval remain separate. A 50 Hz trace cannot qualify physics substep torque or motion above its Nyquist frequency.

The recorded environment retains solver iterations 16/4, external forces every iteration disabled, position/derivative gains 30/0.6 and actuator effort cap 1.6 N m. Its separate native simulation effort-limit field remains 5.5 N m. These distinct configuration fields are preserved in the environment YAML; the measured applied torque above is the actuator-clipped output.

## Portable verification

```sh
python3 verify_payloads.py
python3 verify_payloads.py --preparation ../direct_omni_cold_preparation_001
```

This standard-library check reads only local files. It verifies the complete publication inventory, all 23 raw payloads and their byte counts, nested frozen reviews, identity and cleanup bindings, unchanged standing evidence, restoration and historical profile separation. APFS clones were used during assembly; file contents are ordinary standalone files, not symlinks or external references.

With NumPy installed, reproduce the analysis into a fresh directory outside this immutable bundle:

```sh
python3 -B audit_utility/analyze.py --run evidence/run --historical historical/diagnostics_003.json --output /path/to/new-analysis
```

The command makes only derived JSON/Markdown outputs. The seven original utility tests are preserved with their actual result log; their fixture paths refer to the original workspace and are not advertised as standalone publication tests.
