The limited left-turn reference passed; the combined forward-right arc failed its five-support gate. The campaign therefore ended failed. Fresh standing and unchanged quiet passed for all 32 replicas before either cold directional case.

Left turn completed the full 48 s: 24 s commanded +0.015 rad/s yaw, 11 qualified generator landings across all six legs and 11.88 s measured final quiet. Mean measured yaw was 0.01395948 rad/s, actual heading change 0.33801961 rad, planar excursion 2.421 mm, settled 400 Hz peak torque 1.1403507 Nm. The original 50 Hz position/rate discrepancy passed at 0.960976 mm; the separate 400 Hz diagnostic was 1.002792 mm. No resets, nonfoot contact or post-settle saturation occurred.

Arc rejected at control508/time10.16 s: LM was swinging and RM carried only 0.883063 N, below the unchanged 1 N support threshold. RR remained loaded at 8.02679 N. Two generator touchdowns had completed; no final quiet window was reached. Settled 400 Hz torque remained below the limit at 1.4377017 Nm. These finite contact points and nonzero RM load indicate loss of admitted support, not proven detachment or falling.

`actual_review/` preserves source-bound scoring, full400 Hz endpoint/counter checks and the unresolved standing joint-rate discrepancy. It also preserves the tiny local-versus-Spark quiet-heading numerical difference; original raw verdicts are unchanged. All-recorded reset requested torque peaks are explicitly retained separately from settling-excluded qualification. This is neither hardware-startup nor full Stage2/PPO admission.

`raw/` contains all44 remotely hashed payloads. `remote_audit.json` proves unchanged930 source/550 assets, all3 owned IDs/names absent and pause046 restoration at that historical audit. `preparation/` is an unchanged compact reconstruction bundle; `guard/` retains the exact launcher/restorer. No prior reverse or strafe case was repeated and no frozen parent bytes were changed.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B verify_payload.py` for portable immutable payload and historical result verification. No GPU is used.
