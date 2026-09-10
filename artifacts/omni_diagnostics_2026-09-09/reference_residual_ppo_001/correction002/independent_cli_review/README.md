# Independent consumer002 CLI correction review

The exact 20-file consumer002 map `e9a1d6d687504fd1323d8addbcb962c3025745a258f71b9f076879778eedaa0b` verifies. The only runtime edit from frozen001 is `allow_abbrev=False` on the first argument parser. Seventeen existing payloads, including physical contracts, scores, session, learner, plan and checkpoint helpers, are byte-identical; README changes and one regression file accompany the fix.

The two focused tests pass independently with system Python. Their three subprocesses reproduce001's `--device cuda:0` overwrite of `--device-run`, then verify002 preserves the exact admitted device proof with the complete late AppLauncher flags before and after the required proof arguments. Every subprocess uses preflight-only and returns before AppLauncher; no output campaign is created and no GPU is allocated.

No new physical/RSL tests were necessary for this single parser edit. Previous physical/host and session/RSL receipts remain separate frozen evidence. This review does not imply calibration, physical exploration, PPO training or Stage 2 success.

Re-run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/reference_residual_ppo_source_002 -p test_late_app_launcher_flags.py -v
```
