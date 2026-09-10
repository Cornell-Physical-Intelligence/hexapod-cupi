# Actual recovery admission001: independent numerical review

The complete actual 712-control consumer003 recovery trial passes this numerical
audit. This is learning infrastructure evidence, not physical controller
admission. No PPO updates occurred. Recommend only the already bounded
32-replica ten-update pilot, followed by unchanged cold initial/final forward-stop
and 32-replica quiet screens. Root must issue the separate bound review receipt.

Of 16,384 post-startup row-controls, 15,443 were active (94.26%) and 941 were excluded
recovery. Six rows failed at five event times, all with reference support failure.
Rows 6 and 27 completed exactly 200 recovery controls; rows 1, 2, 10, 19 were still
recovering when the bounded trial ended. These incomplete episodes remain failed
and are not promoted to acceptance. Eighteen moving rows completed all 512 active
controls, with independently measured mean forward speeds 0.004334–0.004653 m/s
against 0.005 m/s requested. Eight rows received quiet commands.

All 5,697 physical samples, named joint/pose/torque endpoints, 14 actual contact
clock streams and five selected reset-clock events replay exactly. Thirteen
compact ledger records preserve the actual final critic packet, zero excluded
reset packet, old/new episode identities, selected fresh history after recovery
and unchanged unselected history at reset. Recovered actor history bytes and
validity flags agree with the actual encoded packets. True terminal bootstrap is
zero. No timeout occurred on GPU; timeout correctness remains separately tested
CPU evidence, not a claim about this trial. No normalization/storage/optimizer
updates were made by the unoptimized probe. All residual action/goal/position/
velocity values were exactly zero.

Active requested motor torque peaks at 1.5907357 N·m, with no active saturation.
Requested transients during excluded recovery reach 3.4626656 N·m (165 joint
substep samples above 1.6); initial startup peaks 3.9618893 N·m. These are explicitly
retained and never called torque-safe physical admission. Applied torque remains
within the exact float32 representation of 1.6 N·m for every sample. Raw SDK rates
and measured displacement remain separate; no physical rate-fidelity claim.

The 512-control loop took 52.971 s: mean 0.103457 s/control, p95 0.109256 s,
maximum 0.256747 s. The measured loop rate is 309.30 environment-controls/s.
This supports the bounded 32-replica experiment; it does not measure PPO optimizer
cost or justify 128 replicas. Memory evidence is limited to one exact-owned
startup cgroup sample (~3.48 GB peak); the later container had already been removed
and the terminal user unit reports no MemoryPeak. No runtime peak is invented.

`report.json` contains all raw numerical results and input hashes.
`decision_evidence.json` binds the three actual phase states, actual source
identity, learning receipt and independent report. It is a recommendation and
explicitly not an allocation approval. Root's actual-source/ownership audit is
referenced by its hash; the complete 74 raw files are preserved separately under
`tmp/reference_learning_ppo_results_001`, avoiding duplicate large artifacts.

Run the read-only portable audit:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_learning_actual001_review/audit_learning.py \
  --phase tmp/reference_learning_ppo_results_001/run/learning_recovery_32 \
  --out NEW_REPORT.json
```

Two focused tests pass, including rehashed corruption of unselected/stale
history, critic packet, terminal bootstrap, recovery timing and normalization.
The existing independent raw clock/substep/episode auditor is included byte-exact
as `audit_base001.py`; its old consumer filter/main is unused. The new wrapper
explicitly binds consumer003 and adds actual ledger evidence. `report_attempt001`
is a preserved successful first replay before adding the explicit zero-residual
and mask-separated torque breakdown. Frozen runtime and main remain unchanged.
