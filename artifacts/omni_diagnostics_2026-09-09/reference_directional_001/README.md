# Directional reference 001: reverse completed but was not admitted

The campaign failed. Reverse finished all 2,400 controls (48 seconds), including 11 qualified measured landings across all six legs and a passing quiet finish. It nevertheless failed the original position-versus-reported-velocity consistency gate: **5.876 mm against a 5 mm limit**. Left strafe, left turn and the forward-right arc never ran.

A separate finalization defect prevented writing the final verdict: `yaw_required` was a NumPy Boolean, which strict JSON serialization rejected. The exception handler retained the same unencodable gate, so its save also failed. The preserved raw reverse state therefore says `running` at control 2400, while the host campaign and systemd unit failed. The [root replay](root_actual_review/report.json) and [independent replay](independent_replay.json) reproduce both the serialization defect and the original physical rejection; neither rewrites the raw state or creates an admission.

| Evidence | Actual result |
|---|---:|
| Fresh standing | 32/32 physical and quiet pass |
| Reverse command | −0.005 m/s for 24 seconds |
| Measured displacement along requested direction | 117.423 mm |
| Mean reported body speed along requested direction | 0.004666 m/s |
| Qualified measured landings | 11, across all six legs |
| Original 50 Hz position/rate gap | **5.876 mm: fail** |
| Independent 400 Hz link trapezoid gap | 4.840 mm, diagnostic only |
| Post-settle 400 Hz requested/applied torque peak | 1.458 N·m |
| Post-settle saturation / nonfoot contact / resets | 0 / 0 / 0 |
| Minimum distal support count | 5 |
| Reference quiet after stop request | 5.54 seconds |
| Scored quiet after another 2 seconds of settling | 12.46 seconds, pass |

The 400 Hz integral does not replace the original 50 Hz acceptance gate. Joint rates also retain a substantial discrepancy: the worst moving-interval joint, `revolute_2`, changed by −0.02798 rad while its reported-rate integral was −0.27286 rad. Neither this run nor the denser diagnostic qualifies native velocity fidelity.

The wrapper preserves all 33 fetched raw payloads, the immutable 24-file [preparation](preparation/README.md), its original owner/root review, the later [frame review](late_frame_review/README.md), the exact dispatch guard, and root's raw replay/audit. Root's remote audit reports all 930 source files and 550 admitted assets unchanged, both owned containers absent by exact ID and name, and pause041 restored at Unix `1789030025.5451584`. This wrapper independently matches the 33 downloaded payload hashes; it did not perform another live lock or workload audit.

The independent replay verifies the complete 930-file local source, 33 raw hashes, all 32 standing results, full reverse gate equality, 19,201 substep samples, exact control/substep counters, cadence and control-endpoint parity. All eight mock guard-restoration tests pass. Those checks do not dispatch services or a GPU.

Run the portable, standard-library verifier from any location:

```sh
python3 /path/to/reference_directional_001/verify_payload.py
```

For numeric replay, use NumPy and the exact reconstructed directional source:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /path/to/reference_directional_001/replay_gates.py --source /path/to/exact_directional_source_001
```

[Reconstruction metadata](RECONSTRUCTION.json) references the published [reference009 parent](../reference_physics_009/README.md). The compact overlay contains all five added/replaced paths and the complete 930-file target map. Parent runtime and asset payloads are referenced rather than duplicated. Original preparation statements about pending results remain frozen historical text; this terminal wrapper supplies the actual result.

A separately versioned serializer/failure-export correction can assess the three unmeasured cases without changing wave005 or any physical gate. This failed run is not evidence of reverse admission, omnidirectional walking, PPO learning or Stage 2 completion.
