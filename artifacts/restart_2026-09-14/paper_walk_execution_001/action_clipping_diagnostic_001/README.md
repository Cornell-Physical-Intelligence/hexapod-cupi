# Recorded action-clipping diagnostic

The retained PPO checkpoint200 is saturated in deterministic evaluation, beyond its reported47.24% stochastic training sample clipping. The precise per-joint/per-command distribution of training means is unavailable because those original rollout arrays were not saved. This diagnostic uses the actual recorded means, observations and targets from eval012/013 only.

| Recorded segment | BC012 mean actions outside ±1 | PPO013 mean actions outside ±1 | PPO013 held targets at action-range endpoints |
| --- | ---: | ---: | ---: |
| Forward, controls100–999 | 0% | 55.556% | 55.556% |
| Quiet from reset, controls100–999 | 0% | 22.296% | 22.296% |
| After forward→zero, controls500–1049 | 0% | 61.111% | 61.111% |

Forward holds all six femur means above+1 and LM/LR/RF/RM tibia means below−1 on every one of the900 analyzed controls. All10 corresponding actual targets stay constant at approximately+0.05rad: femur neutral−0.30 plus0.35, tibia neutral+0.40 minus0.35. These are front-end action-range endpoints. Every nominal ±0.35rad interval is strictly inside the measured URDF joint limits; the joint-position clamp changes zero targets. There is also no sustained slew-limit activity in these10 coordinates after control100. The zero-command stop segment adds RR tibia, yielding11 constant endpoint targets.

Quiet from reset has a different saturated posture: LF/RF/RM/RR coxa means clip98.11%,99.67%,100%,99.56% of the900 controls respectively, plus intermittent LR coxa clipping4%. Their targets are near−0.35,+0.35,−0.35,+0.35rad. This differs from quiet after moving and shows why command context/history matters.

Using each saved per-joint standard deviation and the actual forward means, the analytical probability that a fresh Gaussian coordinate falls inside ±1 averages0.106% for LF femur,0.679% for LM femur,0.0443% for RM tibia, and0.000224% for RF tibia. Other saturated coordinates remain much closer to the boundary: LM tibia26.6% and RR femur36.2% interior probability. These calculations describe distributions at recorded states, not newly executed trajectories. They do not establish zero PPO gradients or an irreversible policy: raw Gaussian likelihood gradients can still move the means.

All6100 emitted targets reproduce bit-exactly from recorded policy actions, previous held targets, native neutral/limits and the existing front-end arithmetic. Commands exactly match observation columns210:213. Recomputing the previous executed-target observation feature on CPU differs from recorded GPU division by at most5.96e−8; the analysis reports this under explicit1e−7 tolerance. The failed initial demand for feature byte equality and its original script are preserved as ATTEMPT_001.json/analyze_attempt_001.py. This numerical difference does not affect the bit-exact emitted-target reconstruction.

Eval012 actually used Config32 BCfit003. This diagnostic independently verifies its complete model and AMP tensor bytes equal Config128 BCfit004. All six control-trace hashes match their batch reports; checkpoint hashes match the recorded lineage. Source, observations and physics remain unchanged.

The first repaired20-update window is useful to test accepted KL and learned-state preservation, but a larger continuation should also show decreasing deterministic mean/target endpoint occupancy and useful measured motion. Stable KL alone cannot establish that this low posture and restricted exploration have improved. No escape-update estimate, alternative checkpoint choice, physical-gate change, fitting or native run is implied.
