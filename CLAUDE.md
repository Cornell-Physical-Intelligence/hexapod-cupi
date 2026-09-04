# Hexapod MKII (hexapod-cupi)

Read `AGENTS.md` first: it is the agent runbook for the robot model and for
Isaac Sim / Isaac Lab on the Spark. Essentials:

- The training asset is `robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf`
  (CAD assembly, 8.261 kg, RS05 hard-set to 191 g). The mock package and the
  `revolute_*` names are archived history.
- Conventions: body frame z up, forward -y, left +x; legs `lf lm lr rf rm rr`;
  joints `<leg>_coxa_yaw|femur_pitch|tibia_pitch`; root link `body`; zero =
  CAD pose (coxa_yaw zero = leg radial); limits and stance live in
  `robot/hexapod_mkii_assy/joint_limits.json` and `stance.json`, mirrored by
  `isaaclab/hexapod_rl/asset_cfg.py`.
- Isaac flow on the Spark: `tools/import_urdf_to_usd.py` ->
  `tools/enable_nested_contact_reports.py` -> `isaaclab/validate.py`
  (nothing trains before validation passes). Details and troubleshooting in
  `AGENTS.md`.
- Regenerate the model with `robot/tools/import_onshape_hexapod.py`; never
  hand-edit the URDFs. Keep `cd isaaclab && python3 -m unittest discover -s tests`
  green.
- Commit only what the user asks for; `tmp/` and the packed viewer HTML stay
  out of git.
