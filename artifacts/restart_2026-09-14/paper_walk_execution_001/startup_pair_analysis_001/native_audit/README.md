# Independent native startup audit

`RESULT_004.json` is the completed CPU audit. Both diagnostic policy screens remain failed; no Stage 2 acceptance is assigned.

Cold retains all 1,000 BC controls / 8,000 physics steps. Settled retains all 200 scripted neutral controls / 1,600 physics steps followed without reset by 1,000 BC controls / 8,000 physics steps. Their exact initial native state and native configuration readbacks match. All 2,200 issued targets reproduce the actual frontend bit for bit. Every BC issued action equals its recorded actor mean; the 200 scripted actions and commands are explicitly zero and their actual held targets remain neutral.

The audit recomputes the full 400 Hz servo demand, speed-dependent effort ceiling and clipped input exactly, including pre/post state continuity, explicit counters and 50 Hz endpoint/hold aggregation. It independently transforms and reclassifies 394,457 saved contact patches (179,261 cold; 215,196 settled), reproducing distal/nonfoot forces and support flags exactly. No termination, truncation, reset, nonfoot event, joint/speed bound violation or applied-torque cap violation occurs. Complete full/prefix/policy physical windows match the source019 receipts, without allowing the prefix to dilute a policy failure.

| Window | Native steps | Requested saturation, all joints | Missing six-toe steps | Peak requested Nm | Peak applied Nm |
|---|---:|---:|---:|---:|---:|
| Cold BC, full 20 s | 8,000 | 0.234028% | 676 | 6.006391 | 1.600000024 |
| Scripted neutral prefix, 4 s | 1,600 | 0% | 12 | 1.095524 | 1.095524 |
| Settled BC tail, full 20 s | 8,000 | 0.478472% | 1,654 | 5.862412 | 1.600000024 |

The prefix's 12 missing-support samples have zero toes; its following 1,588 samples have all six toes. This is scripted settling, not learned quiet standing. After the unchanged two-second policy exclusion, cold has 100 missing-six-toe samples among 7,200 and settled has 1,063 among 7,200. These support diagnostics do not change the original forward profile's requirements.

All original checked metrics, bounds and failed-bound lists reproduce exactly on the separately copied 1,000-control policy views, with original timestamps. Both fail only planar tracking: cold error 0.050607534 m/s; settled error 0.059982400 m/s versus the unchanged 0.025 m/s limit. Mean forward COM velocities are −0.000047196 and +0.000358904 m/s, respectively. The raw SDK joint-speed maxima are 47.425510 and 43.736809 rad/s, below the existing native limits. Physical-window pass is not evidence of useful walking.

The first audit attempt preserves the existing generic loader's rejection of the new string action-source channel (`audit.py`, `RESULT.json`). The second and third attempts preserve overly strict descriptive-percentile equality failures (`audit_002.py`/`RESULT_002.json`, `audit_003.py`/`RESULT_003.json`). The final artifact-only loader accepts only `bc`/`scripted_neutral` in that named channel. Explicit float32 rank interpolation reproduces the saved vertical/tilt 95th percentiles; these are descriptive in this profile, and all actual gate values already matched. Original saved values remain unchanged and both local/native values are retained.

Limits: this is saved-evidence verification, not new physics. Hash-matching maintained servo/scorer functions and two pure prior-audit helpers are reused. Contact points are independently reclassified using saved native poses and frozen cap geometry, but all detailed mesh vertices are not retransformed for clearance. Independent actor-network/history reconstruction and video decoding/inspection belong to the parent pair analysis. Separate process identity relies on the root's receipts. Both cleanup records show their owned containers absent and the compute reservation retained. All 96 formal Stage 2 cases remain unrun.
