# Fresh BC fit004 — batch128 configuration

Candidate checkpoint: `candidate_checkpoint_update000000.pt`, SHA6b169d9a406b8a0d6208a3282f59111529b375461b2e86bc391dfb9fe0a2c8f9. This is a new, actually fitted checkpoint, not edited metadata on an existing checkpoint.

Root authorized a CPU-only rerun with every fit003 candidate setting unchanged except Config.num_envs32→128. The script constructs a fresh learner and verifies that exact config difference. It then proves initial model/AMP/RNG equality with fit003, performs1,000 BC steps with batch512, and proves final model/AMP/RNG equality with fit003. Any mismatch would have stopped the run. All comparisons passed.

Final learner source b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a; original AMP prior22f7b04b…; separate BC dataset f016af1a…; seed20260914; std0.1; velocity coefficient1 with BC-only detached estimator feature. Dedicated BC Adam3e-4 is discarded; PPO Adam1e-4 and discriminator Adam remain empty. The future PPO config includes target-KL0.02. Initial/final checkpoints preserve full source/config/AMP identity and RNG. Full recorded-observation inference and all optimizer/model states passed independent strict reload.

Actual fit4.145s; entire invocation6.327s with two CPU threads. Outputs remain about10.64MB under the40MiB cap. PPO updates, transitions, PPO optimizer steps, discriminator steps and episodes are0; BC steps are1,000. No maintained source, original dataset or existing checkpoint changed, and no native environment was constructed.

The fit003 recorded-input estimator/action metrics apply exactly because model parameters and normalizers are identical; REPORT.json binds that report's SHA. This reuses the same diagnostic evidence and is not another validation sample. In particular, the estimator's related-cycle3 RMSE0.00613m/s does not establish native gait, quiet stopping or robustness.

Native128-replica admission remains a separate prerequisite for root's larger PPO run. This CPU artifact provides no physical admission and does not promote Stage2 status. Any future native run starts from its actual reset; simulation resume is unsupported.
