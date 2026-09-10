Both diagonal static-load tests passed after fresh32-replica standing and unchanged quiet admission. LF/RR and LR/RF each completed2.02 s simultaneous measured unload, followed by six-foot return and10.02 s measured quiet.

LF/RR lifted5.295/5.781 mm with minimum projected support margin112.521 mm. LR/RF lifted3.821/5.697 mm with margin114.404 mm. All8801 diagnostic physics samples per case stayed under1.6 Nm requested torque; the full-interval peak was1.13638 Nm. The actual requested peaks during qualified unload were1.05841 and1.10666 Nm. No reset, nonfoot contact or missing retained support occurred.

![Actual static loads and all physics-step torques](actual_review/actual_diagonal_loads.png)

`actual_review/` independently replays both exact named scorers and every1100 post-step support/body/slip checks, with exact counter/timestamp and pose/joint/torque endpoint parity. It preserves initial startup torque transients separately and the unresolved reported-rate versus angle discrepancy. These results qualify the declared static diagnostics only; the moving paired sequence and faster gait still require actual measured proof.

`raw/` preserves all56 remotely hashed terminal payloads. `remote_audit.json` verifies unchanged934 source/550 admitted assets, all3 owned IDs/names absent and pause047 restoration. This is historical cleanup evidence. `preparation/` and `guard/` preserve exact frozen source reconstruction and guarded launcher/restorer. No original middle-pair or wave gate was weakened, and no prior physical case was repeated.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B verify_payload.py` for portable immutable payload and historical result checks. No GPU is used.
