# Actual residual-PPO001 parser failure and isolated002 correction

The fresh standing phase completed 1000 controls. The smoke process then failed
before AppLauncher, calibration, checkpoint creation or PPO optimization.
The early parser accepted `--device cuda:0` as an abbreviation of its known
`--device-run` option and looked for a device proof under `cuda:0/campaign.json`.
This is an argument-parsing failure, not a rejected exploration experiment.

`parser.patch` records the sole executable successor change:
`ArgumentParser(add_help=False, allow_abbrev=False)`. Consumer002 is separately
frozen at `tmp/reference_residual_ppo_source_002`, manifest
`e9a1d6d687504fd1323d8addbcb962c3025745a258f71b9f076879778eedaa0b`.
It retains all physical, learner, reward, observation, noise, action and gate
settings. Of the original19 payloads,17 are unchanged; only the entrypoint and
README differ, and one actual-CLI regression is added. Consumer001 is unchanged.

The regression reproduces the old error and passes the corrected full command
with `--device cuda:0`, `--headless`, `--info` and the host's `--kit_args`, in both
flag orders. All three subprocesses stop before AppLauncher. The independent
three-file review is separately frozen at
`tmp/reference_residual_ppo_cli002_independent_review`, manifest
`5af44676f4f0042f6b26eddd61788654371da8b48227aef9b9e7925e049098c9`.

The raw folder contains all16 compact text run outputs, the exact restored
pause042 receipt, and SHA256s for all19 remote run files. The three large numeric
standing files are deliberately not duplicated here; root owns their complete
retrieval and terminal source/asset/ownership audit. Job records report exact
owned cleanup, and the restored receipt exists with its recorded hash. No GPU
signals, successor launch or main/Git mutation was performed by this reviewer.
A future actual002 campaign still needs fresh standing, calibration and all
bounded policy/retention gates. No physical policy pass or Stage2 completion is
claimed by this correction.
