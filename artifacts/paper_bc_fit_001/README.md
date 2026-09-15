# Offline BC fit diagnostic

Existing maintained learner, unchanged source SHA `a99798d7a9b0757292d2c509c02b7b220866102ce62a976f0ef8bd10a24d64ab`; original accepted 1860 steady native examples; CPU only, two threads, 1000 BC steps in 3.809 s. No native environment was constructed and no onset/stop examples were invented.

The update-zero checkpoint is `bc_checkpoint_update000000.pt`, SHA `f9d32d7fdfe132e7f9ba050c08364f641079d02b833548ddf949d2683e542660`. Config specifies 32 environments, initial std 0.1 and seed 20260914. Serialization and an independent full-dataset model/Adam/normalizer/RNG reload are exact.

| Recorded-input metric | Training cycle | Adjacent cycle 3 |
| --- | ---: | ---: |
| Actor action MSE | 0.0000133794 | 0.0000270143 |
| Copy-previous-action MSE | 0.000953138 | 0.000953138 |
| Zero-action MSE | 0.0642742 | 0.0642742 |
| Increment direction cosine | 0.98767 | 0.98090 |
| Equivalent unlimited target RMS error | 0.001280 rad | 0.001819 rad |

Every moving command improves over copying the previous action. Static-zero copying has exactly zero error and the fitted actor does not beat it. Increment MSE is algebraically the same as action MSE; it is not independent corroboration. No clipping occurs in predicted absolute actions on these recorded examples. Adjacent-cycle evidence repeats the same targets, shares temporal boundaries and is conditioned by original row selection; it is not independent generalization.

These results establish fitting on supplied observations. They do not show gait initiation, stable closed-loop motion, stopping or qualification.

The 20 updated parameter tensors belong to estimator/memory/policy and each has 1000 Adam steps. The shared PPO Adam retains those moments. Critic, discriminator, log_std and their relevant normalizers remain at fresh initialization. PPO counters are zero; bc_steps is 1000. Actor normalizer count is 1860.0001; critic/AMP counts are 0.0001. The complete Python, NumPy and Torch RNG states are in the checkpoint.

The estimator was optimized as an action latent, not supervised as velocity: its actual recorded-input velocity RMS error is about 0.774 m/s. Fresh discriminator/style and critic readouts are untrained outputs, unsuitable for comparisons with a trained PPO discriminator. See `ESTIMATOR_DIAGNOSTIC.json`.

Additional unchanged-recording queries find no normalizer clamps on reset-zero or settle rows; 27 of 1500 screened onset rows clamp, and onset MSE is 0.000991 versus copy 0.001387. This is weaker transient generalization than the steady fit, not a closed-loop result. `RECORDED_INPUT_GENERALIZATION.json` preserves the exact scope.

`REPORT.json` contains per-command metrics and complete provenance; `NAMED_JOINT_METRICS.json` binds the verified canonical joint order. `CPU_RUN.py`, invocation/status, input hashes, progress and initial checkpoint reproduce the actual fit. The tools-disabled Fable 5.1 max handoff review is preserved under `handoff_review/`. `HANDOFF_RECOMMENDATION.md` records independent caveats and the later subset-allocation instruction.

No native handoff or experiment is adopted by this artifact. Stage 2 remains incomplete.
