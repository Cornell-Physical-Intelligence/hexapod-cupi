# Step-1 evidence · 4 September 2026

The corrected **serial** conversion passes offline checks. Physical four-bar
accuracy and live simulator acceptance remain open. See
[MKII_STEP1.md](../../docs/MKII_STEP1.md) for implementation, reproduction,
inspection controls and exact limits of the claim.

| Evidence | Result |
|---|---|
| [Full CPU suite](cpu_tests.txt) | 636 tests passed; no Isaac Sim application required. |
| [Verification summary](verification_summary.json) | All 112 archived pipeline entries unchanged; all 12 prepared-bundle hash entries match. |
| [Local USD check](usd_cpu_check.json) | 19 bodies, 18 joints, 171 colliders; all masses, COMs and tensors pass. |
| [Spark installed-SDK check](spark_sdk_usd_cpu_check.json) | OpenUSD 26.8 pass; dependency hashes identical to local; no GPU devices or physics stepping. |
| [Stance geometry](stance_geometry.json) | Nominal minimum foot clearance 5.000 mm; 750 sampled reset-jitter leg poses minimum 1.265 mm. This sampling does not prove a continuous geometric bound. |
| [Linkage closure](linkage_closure.json) | Mimic topology passes; documented cut-point coincidence fails the stated 0.1 mm diagnostic. Maximum point gap 0.503 mm, mainly axial; maximum transverse gap 0.0183 mm. Serial frozen-rod stance gap 16.305 mm. |
| [Future wrapper dry run](train_wrapper_dry_run.txt) | Task/source wiring only; no training started. The host is a Mac without the container's training script. |

Serial URDF SHA-256:
`6109956e9e3a7a648bc4afa7310904c2fd3f7cb790e157d83212339a73b7fbc0`.
Prepared root SHA-256:
`033d31d58f18cc2ec83afcdf7962aef014d03afdcc45e37f7fa23332b14e886b`.
The asset's own `SHA256SUMS` covers its layers and provenance report; this
directory's `SHA256SUMS` covers these dated evidence files.

Browser checks loaded all 77 local STL files, played/paused the knee sweep,
moved all six knees, preserved a manually selected knee pose through model
comparison, and returned to the nominal stance. The default linkage view has
31 links, 18 independent joints and 12 mimic joints. The motion is kinematic;
it establishes neither contact stability nor a trained gait.

The read-only Spark snapshot at 23:23:47 UTC reported 0% GPU utilization and
the pre-existing viewer container. This was a point-in-time observation, not
a reservation or ongoing monitor. The new asset was staged separately under
`/home/orionh/HEXAPOD/robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/`.
No original USD, source mirror, historical checkpoint or unrelated job was
replaced. No new training, queue, service or automatic restart was started.

Independent implementation review found a nominal-vs-jitter clearance
overstatement and an abbreviated-argument bypass in the future training
wrapper. Both were addressed with regression checks. A green suite does not
close the explicit physical/model gates above.
