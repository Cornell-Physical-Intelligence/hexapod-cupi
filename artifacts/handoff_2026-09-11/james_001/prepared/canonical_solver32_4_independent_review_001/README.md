# Independent source004 solver diagnostic review

Exact95-payload source004 freeze037013353a30c5f3751235634aafdb3f204b676837962b14df72c3051c0a2786 verifies. Eight focused solver and actual score/result contract tests pass independently. The source keeps every session, scorer, servo, geometry, asset and SDK byte unchanged. Removing only the explicit solver-readback insertion lines restores the entire parent native entry byte-for-byte.

The diagnostic requires installed un-authored32/1 schema values before any new authoring, then authors32/4 and checks each root after authoring, after reset and after all controlled steps. It rejects missing, noninteger, already-authored or unexpected parent values and changed/missing/unsealed terminal readback. The actual unexpected values now survive the exception; this resolves the one review finding. The result contract continues to reject quiet and physical failures even if the solver metadata is correct.

This is a controlled solver-configuration experiment, not an accepted repair. USD attribute readback does not independently measure all internal backend iterations. Position32, external-force-each-iteration, all physical gate thresholds and provisional PD are preserved. Same-source one and32 standing are required before a later source-bound PPO adaptation. Actual source00332 rejection is not waived. Fable MAX consultation is separate and pending at this freeze, not claimed complete.

Run `PYTHONPATH=tmp/updated_native_standing_004 .venv/bin/python3 -B -m unittest test_solver_recipe test_score_contract` from repository root. No GPU or native process was started. Root owns eventual publication with the mandatory STATUS/PLAN/NEXT_RUNS/site framework record and build. Source004 remains unchanged.
