# Noisy-location and explicit-filtering controls

The transient at **(−2, 0) m reproduces with a single robot**. Its complete
3,200-sample physical/control trace matches the earlier eight-robot batch's
row 7 bit-for-bit: all 30 joint states, 31 body poses/velocities, actual and
processed targets, raw/applied torques, PD terms, motor headroom and six foot
forces. Its peak remains **0.96812451 N·m**, minimum support 5 and maximum
passive position error **33.7362 µrad**. Only derived hinge calculations have
tiny differences (at most 5.69e−14 m/chord and 7.46e−9 m/s).

The separate explicit-collision-filtering release produces an even stronger
control: its entire eight-robot NPZ is **byte-identical** to the preceding
unfiltered, spaced-world trace. Both hashes are
`71042eff15c71a63c2e345795f57c6de7c860c5a80c86936ff6a18a9d04d9bef`.
Explicit filtering fixes the missing isolation configuration without changing
these spaced-world dynamics. This does not prove overlapping-world isolation;
the deliberate positive/negative contact controls remain separate.

Combined with the earlier quiet origin and (2, −2) controls, the evidence
establishes **translation-dependent simulated behavior in this setup**.
Additional robots are not required for the observed transient. The responsible
floating-point operation or other native numerical mechanism is not identified.
Changing the sensor row interpretation cannot explain the matching native
joint/body states. The velocity-level mimic residual motivating a targeted
128/4 solver test is preserved independently in
`../velocity_constraint_residuals/`.

## Exact input identity

The noisy-location single and old batch both use **`d6d5863`**, identical full
source manifest and functional/runtime contracts. They have identical initial
actual root position, all 30 initial joint coordinates, motor/backend readbacks
and every delivered target. Their settled comparison is 0.8–4.0 s; complete
state equality covers all 0–4.0 s. The clone's USD environment root and the
single's translated reset were constructed differently, but their recorded
world-space physical trajectories match.

The filtered batch uses **`d863663`**. Its physical runtime identity is
unchanged except for the verified explicit-isolation descriptor. The functional
file differences are the new collision-isolation helper, the task's scene
setup and the pipeline lineage checker; exact hashes appear in `result.json`.
Both compared releases still use 128/1, Kp30/Kd0.30 and 1.25 ms ×16. They
precede the subsequent final-velocity-four candidate.

| New evidence | Report SHA-256 | Trace SHA-256 |
| --- | --- | --- |
| Single (−2, 0) | `4e323adda428d0be7c972da2d6ab490bd1d668445174b92964eca93911f5be14` | `fd7c0442fe292a5adb343445ad3b8cae3106a183c23fe2c1716fd3d0ccc38e19` |
| Explicitly filtered batch | `3fe5a9e257d9419b9bda7c6b4413795a5f534ac5ab9be6b08cb0e7e2b2d40183` | `71042eff15c71a63c2e345795f57c6de7c860c5a80c86936ff6a18a9d04d9bef` |

Original files are retained under sibling `translated_xminus2/` and
`filtered_batch8/`. Remote run roots:

```text
/home/orionh/HEXAPOD_runs/mkii_placement_diagnostics_v1/translated_xminus2_20260905T210539Z/hexapod-fourbar-diagnose-20260905T210539Z-62e98edd
/home/orionh/HEXAPOD_runs/mkii_collision_isolation_v1/batch8_20260905T210111Z/hexapod-fourbar-diagnose-20260905T210111Z-96687d5d
```

The genuine supervisors confirm completion, unchanged source and exact owned
container cleanup. Their source manifests are verified against supervisor
hashes. Archives remain on Spark; no `source.tar.gz` was downloaded. No old
evidence file or manifest was altered for this follow-up.

## Reproduction

From the repository root with NumPy installed:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/batch_followup_analysis/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/batch8/hexapod-fourbar-diagnose-20260905T205225Z-8faf9917/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/translated_xminus2/hexapod-fourbar-diagnose-20260905T210539Z-62e98edd/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/filtered_batch8/hexapod-fourbar-diagnose-20260905T210111Z-96687d5d/report.json \
  --out /tmp/hexapod_batch_followup_new.json
```

The script reuses the frozen batch reader, verifies trace hashes and ranges,
compares complete named fields, and rejects changed inputs or unapproved
physical-runtime differences. It launches no simulation. The accompanying
manifest uses paths relative to the parent `translation_diagnostics/` folder.
