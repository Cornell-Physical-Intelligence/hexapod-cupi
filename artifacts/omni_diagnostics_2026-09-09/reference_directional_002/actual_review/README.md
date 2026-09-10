# Directional002: left-strafe support rejection

Fresh32 standing and unchanged quiet checks pass. The left-strafe case is correctly **rejected at control505 /10.10s**; the original scalar gate replays exactly and its complete terminal JSON strictly serializes. The finalization correction works. Left turn and forward-right arc did not run.

The intended swing leg is LM, following two confirmed LF/RR landings. At the final sample, RR's normal load is **0.945591N**, below the unchanged1N distal-support threshold, leaving only four admitted supports. RR still has a finite contact point and nonzero reaction. This is lost admitted load support; it is not evidence that RR fully detached or that the robot fell. No forbidden contact or reset occurred. The previous nine recorded RR loads are1.04–1.84N, so the final rejection follows a small remaining load margin. This review does not change the threshold or continue the failed run.

The50Hz requested peak is1.517451882Nm. All4,041 available400Hz samples are present with exact8-substep counters, timestamps, q/qdot/pose/torque endpoint parity; settled400Hz requested/applied peak is **1.517596006Nm**, at10.0975s on `revolute_2_6`, with zero samples above1.6Nm. Forces and contact points are recorded at50Hz only; this observer has no400Hz contact-force channel.

Only6.10s of requested motion after4s startup were completed. The prefix pose/rate comparison and joint-rate bias are explicitly diagnostic; there is no full24s motion or final quiet result. Neither left strafe nor the remaining directions are admitted.

All32 raw payloads match the remote audit, all930 source and550 admitted asset files are unchanged, exact owned IDs/names are absent, and forecast pause044 restoration is recorded. These are historical terminal observations; no later lock/GPU ownership is claimed. Preserve raw state/source and this review unchanged. The next declared screen may test only the two unmeasured directions after fresh standing, with the same wave005 and physical gates.
