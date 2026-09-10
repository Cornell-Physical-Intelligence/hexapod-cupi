# Full-C terrain standing attempt 003

**The simulator completed 1,000 terrain controls, but the standing gate rejected the run.** Fresh flat standing passed first. The previous Mesh-import and configuration-copy failures were resolved; this attempt does not admit terrain standing.

Every terrain control returned a termination, with two truncations, zero minimum distal support contacts and 54.5% reported requested-torque saturation. Applied torque remained capped at 1.6 N·m. The validator samples robot data after `env.step`, which can reset terminated environments: these are **reset-contaminated measurements**, not steady-support load or 1,000 physical falls. A first-step, pre-reset diagnosis is required.

The installed SDK declares `AssetBaseCfg.InitialStateCfg.rot` as XYZW. The adapter authored a WXYZ yaw tuple; at the selected π/2 angle that encodes a sideways rotation in this SDK. A separate source correction and independent quaternion test are appropriate, but the old run and its metrics remain unchanged. A runtime trace must verify the actual pose and termination cause before attributing the entire failure to that setting.

- [Terrain state and unchanged gate](run/terrain/state.json), [log](run/logs/terrain.log), [campaign](run/campaign.json).
- [Fresh flat admission](run/flat/admission.json), SHA-256 `b5e9b2a092bc1553a193749a6fcb9b60d7aa50442b0910a70e8a77511cdb7939`.
- [Source manifest](campaign_source_hashes.json), 661 files, SHA-256 `b6e2767475e514977d9ac26dbe50e408fe35f45a087c8908f91cb0770ce479bf`.
- [Installed configuration-copy reproduction](installed_copy_reproduction.json) and [copy correction from attempt 002](config_copy.patch).
- [Pause 015 restoration](forecast_pause_015/restored.json) and [17 exact result hashes](SHA256SUMS.json), independently matched between Spark and the local publication.

Both owned containers exited, CUDA processes and both locks cleared, and the forecasting timers were restored. The robot, plan, fixture catalog and pinned 16-file C runtime remained unchanged. The selected fixture's flat entry pad still does not establish slope standing, traversal, terrain training or physical four-bar qualification.
