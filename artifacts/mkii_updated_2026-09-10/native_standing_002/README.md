# Detailed robot: one-robot supported standing passes

The canonical 7.466088235 kg detailed robot completed 1,000 controls / 8,000 physics steps with one initial reset. Its 16-second scored quiet interval and all physical standing checks pass. Independent post-exit validation verifies unchanged inputs, the complete raw inventory, both owned container identifiers absent, and exact weather restoration. This is a provisional simulation standing result. **No new-model PPO, Stage 2 completion or hardware qualification is claimed.**

All 38 raw files / 179,449,714 original bytes are preserved. The 169,991,706-byte contact stream is stored losslessly as `terminal/run/standing/contacts.jsonl.gz`; `RAW_ENCODING.json` records the original size/hash and storage mapping. Every other raw file is unchanged. The portable verifier reconstructs the exact contact bytes and runs the original standing contract. Compression changes storage only, not the records or scoring.

The same detailed geometry, 153 SDF colliders, masses, limits, 400 Hz provisional servo and 1.6 N·m cap remain. The sole runtime correction from the failed first source recognizes exact finite zero-force / zero-normal / zero-separation tuples as inactive while retaining every record. Every nonzero-force invalid normal and all existing force, support, clearance and quiet thresholds still reject. The original failure remains in `../native_standing_001/`.

Peak requested and applied torque across all controlled substeps is 1.120546 N·m. No nonfoot contact, missing post-settle six-toe support, or requested saturation is recorded. Minimum plate height is 73.044 mm and minimum non-toe mesh clearance is 14.834 mm. Initial settling rates remain in the raw data; quiet metrics are scored only in their declared window. `RESULT.json` contains the exact quiet measurements and original limits remain in the native report.

The independent scalar geometry check finds all evaluated outputs finite and agreement with Python scalar summation within 2.776e-17 m. It does not establish the cause of intermittent local NumPy warning flags, and no warnings are suppressed or frozen math changed.

The source still uses a declared provisional 48 V motor curve and estimated motor inertia. Battery voltage, hardware calibration, payload and thermal behavior remain open. The next bounded step is same-source standing with 32 robots, followed by a fresh two-update PPO integration and the separately qualified all-direction curriculum.

Run `python3 -B -S verify_bundle.py`. Verification temporarily reconstructs about 180 MB of raw evidence. Frozen source/host/guard/auditor and root receipts are included; the exact canonical asset, prior actuation input and unchanged ownership supervisor remain separately published dependencies rather than duplicated trees.
