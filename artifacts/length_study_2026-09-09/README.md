# 49-variant Isaac length comparison — 9 September 2026

`isaac_length_study.png` is an unedited Isaac render captured at physics step
500 (1.25 simulated seconds) on the Spark at 17:12:16 UTC. All 49 floating
hexapods advance together in a 7 × 7 scene. Each uses a different femur/tibia
combination from 50–110% of the original mock lengths; coxa is fixed. Increasing
world X (columns) increases tibia length; increasing world Y (rows) increases
femur length. The camera looks from positive X / negative Y.

The batch completed 1,000 steps at 400 Hz. This is a standing comparison with
the current RS05 actuator configuration; no policy was loaded or trained.
It is not a training acceptance gate. All 49 × 19 imported inertia tensors
passed numerical round-trip verification, maximum absolute error
1.62105e-9 kg m². Applied torque reached the 1.6 N m continuous limit (float32
maximum 1.600000024); raw computed demand reached 3.632945 N m. These maxima
include startup. Saturation duty, ground-contact classification, falls and
dynamic gait performance have not been admitted by this comparison.

- `batch_state.json`: final run, each variant's imported-tensor check,
  observed joint/body order, raw/applied torque maxima and final state.
- `batch_launcher.json`: owned container identity, launch preflight and exit.
- `summary.json`: compact metrics, UTC capture time and image hash.
- `source_hashes.json`: frozen source/asset file hashes.
- `smoke_state.json`: successful one-robot baseline smoke test (`smoke_003`).
- `batch_001*`: first successful 49-robot comparison; its original camera
  clipped outer robots. Batch 002 changes only camera focal length.

The study package, assumptions and regeneration instructions are in
`robot/hexapod_mkii_length_study/README.md`. All local model tests passed:
288 existing Isaac-task tests, 8 new morphology tests, the current CAD URDF's
structural/mesh/inertia checks, and a separate CPU-only OpenUSD test covering
all 931 inertials. The actual imported tensors were also checked in both live
batches. Source assets and current training defaults were preserved.

Spark: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/`.
The final image/report are from `batch_002`; generated USD payloads are under
`source/robot/hexapod_mkii_length_study/usd/`. Each GPU invocation held only
job-scoped locks and ended after the finite comparison. No queue, automation,
PPO job or persistent GPU reservation was installed.

Initial startup attempts `smoke_001` and `smoke_002` exposed, respectively,
an Isaac package-path collision and the removed legacy `_urdf` API. The final
runner uses an explicit Isaac Lab Python path and its installed `UrdfConverter`.
The first launcher version reported the early Kit exception as a zero process
exit; the corrected launcher requires a completed state record and real capture
file before declaring success. The failed attempts are preserved on the Spark
and are not counted as simulation evidence.
