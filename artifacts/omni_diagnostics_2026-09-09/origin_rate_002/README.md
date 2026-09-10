# Origin002: completed matched-position diagnostic

The fresh32-replica standing admission and all five single-robot cases passed the unchanged physical and quiet gates. Moving the same cold robot state to different global XY positions **did not resolve the reported-joint-rate versus actual-angle disagreement**. The largest16s discrepancy remains0.39765–0.41040rad, on the same named joint `revolute_2_1`. This is completed diagnostic acquisition, not velocity-fidelity, PPO, Stage2 or terrain qualification.

Each case ran1,000 controls: the original4s startup/settling followed by16s scored holding. All five retained six distal supports, zero forbidden-contact steps, zero resets, zero postsettle saturation and zero executable-reference lag. The target trajectories were exactly equal across cases. Full startup torque traces remain included; the postsettle results do not qualify hardware startup.

| Case | Global XY (m) | Physical / quiet | 400Hz postsettle requested peak (Nm) | Rate integral minus actual angle change (rad) | Root-link integral/displacement gap (mm) |
|---|---|---|---:|---:|---:|
| origin_a | (0,0) | pass / pass | 1.478385 | +0.4084874 | 0.834318 |
| near | (3,−5) | pass / pass | 1.483903 | +0.4103968 | 0.870177 |
| far | (30,−50) | pass / pass | 1.453999 | +0.3976517 | 1.536228 |
| near_opposite | (−3,5) | pass / pass | 1.482163 | +0.4101662 | 0.826420 |
| origin_repeat | (0,0) | pass / pass | 1.478385 | +0.4084874 | 0.834318 |

These are independent400Hz diagnostics alongside the retained original50Hz gates. At the origin, the worst joint changed−0.0004253rad while its reported velocity integrated to+0.4080621rad. The far translation perturbed that discrepancy but did not remove it. One cold trial per translated location cannot establish a statistical model or native solver cause. No finite-difference metric replaces the SDK rate or existing gate.

The repeated origin's `trace.npz`, `physics_substeps.npz` and `physics_control_integrals.npz` are byte-identical to the first origin's files; the independent review also checks every recorded control/substep array. Generalized q/qdot, target, root Z/quaternion and COM-velocity inputs match the frozen selected reference009 replica6 state exactly; only declared root XY differs. Both separate terrain and public scene origin stores were synchronized and verified through stable storage/views. One enabled infinite collision plane at Z=0 preserves support geometry under translation. Derived19-body positions retain float32 differences after translation and are not claimed bit-identical.

Read [the independent actual review](independent_actual_review/README.md) for initial-state/frame checks, every-case quadrature, exact endpoint/counter coverage and limitations. [Root's independent comparison](result/root_review/report.json), [the original campaign comparison](result/run/matched_origin_comparison.json) and all raw traces remain separate unchanged evidence.

The deployed source map is `c028664ae782b053f0e126ace1e7a24e6eb8a89cf8dbaa9fd8b5628d28560160` (933files). This wrapper preserves the [compact source002 reconstruction](preparation/RECONSTRUCTION.json), its owner and independent19-test receipts, the [bound launch guard](launch_guard/launch_guarded_remote.py), all96 remotely hashed payloads, both actual reviews and the forecast pause038 restoration receipt. Origin001's failed object-identity check remains a separate immutable prior attempt.

[Root's remote audit](result/remote_audit.json) records933 source files and550 admitted assets unchanged, all six owned container IDs and names absent, unit `hexapod-origin-rate-002-20260910.service` successful/inactive, and both StormScope timers restored. The [packaging verification](PACKAGING_VERIFICATION.json) independently matches all96 fetched hashes, every nested frozen input map, all933 local source hashes, and the550 asset hashes against the exact source package. It does not claim a new live lock/workload check after subsequent jobs started.

`BUNDLE_SHA256.json` is the primary wrapper map. `verify_bundle.py` verifies this portable wrapper and its nested/raw maps without importing Isaac, contacting Spark or rewriting evidence. The original review/assembly scripts retain their original source paths and input-layout assumptions; their frozen outputs are included, and reproducing them requires the corresponding original/reconstructed `tmp` layout. Keep all frozen payloads unchanged; publish later conclusions as new append-only evidence.
