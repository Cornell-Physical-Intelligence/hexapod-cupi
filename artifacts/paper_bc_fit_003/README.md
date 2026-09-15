# CPU BC velocity-supervision contrast 003

Completed with final learner b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a after root independently passed 148 prototype tests. This artifact contains no simulator execution, PPO transitions or walking qualification. Total invocation10.46s, two CPU threads, about21.65MB before these summaries (150MiB bound). No failed fit attempts occurred.

Candidate checkpoint: `candidate_checkpoint_update000000.pt`, SHA a48fddf1f73fcaf5b93db44241390184dd36c60e031daaad87d3a63bfd2bd81a. Legacy parity checkpoint: e35e5e47be2fac6c14cceb7cf06d41d06c32e4b14b653711edf14c7d6d4a71ce. Both strict full-dataset actor/optimizer/RNG reloads passed. Updates, transitions, PPO optimizer steps, discriminator steps and episodes remain0; each arm performed exactly1,000 BC Adam steps with batch512.

Both arms use config32, std0.1, seed20260914, original AMP prior22f7b04b… and distinct frozen BC dataset f016af1a… (3,760 genuine native rows). The first arm disables the velocity objective/detach and proves exact fitted model, AMP and RNG equality with fit002. Its dedicated BC optimizer is intentionally discarded, whereas fit002 retained shared Adam; the change does not alter the model/RNG trajectory. Small printed MSE differences reflect NumPy versus prior Torch reduction, not different weights.

The candidate changes the BC objective and gradient routing together: normalized action MSE plus coefficient1 raw-m/s velocity MSE; only the BC policy input receives a detached estimator output. Actual velocity targets are pre-hold body-origin navigation values [-AMP37, AMP36, AMP38]. They remain supervised labels, not privileged actor inputs. No warmup or extra phase input is used.

BC Adam3e-4 is discarded after fitting; PPO Adam1e-4 and discriminator Adam remain empty. The checkpoint config includes the adopted target-KL0.02 safeguard for future PPO, but neither that safeguard nor PPO LR changes affected these deterministic BC fits. Critic, discriminator, log-std and critic/AMP normalizers remain freshly initialized and unchanged; actor-normalizer count is3760.0001. Simulation resume is unsupported; future native tests use original reset.

| Recorded-input metric | Action-only parity | Velocity-supervised candidate |
| --- | ---: | ---: |
| Training action MSE | 7.055e-5 | 7.002e-5 |
| Related cycle3 action MSE | 4.744e-5 | 4.597e-5 |
| Training velocity RMSE, m/s | 0.83895 | 0.005843 |
| Related cycle3 velocity RMSE, m/s | 0.90326 | 0.006127 |
| All zero-command action MSE | 9.874e-6 | 1.061e-5 |
| Final settled second action MSE | 6.350e-6 | 7.052e-6 |

On related cycle3 the candidate's per-axis velocity RMSE is0.00639/0.00659/0.00533m/s, correlation0.973/0.965/0.950 and predicted-to-target standard-deviation ratio1.032/1.005/1.018. Skill against the training-mean predictor is94.2%/92.9%/89.8%; it also beats the fixed ridge baseline on each holdout axis. Thus this recorded-input result is not merely a near-zero prediction. The fixed ridge model used only training history, train statistics and predeclared regularization1e-3; it is a diagnostic baseline, not a controller or mathematical recoverability bound.

REPORT.json provides per-command/per-phase/named-joint action and velocity metrics, zero and mean/ridge baselines, correlations and variance ratios. RIDGE_BASELINE.npz stores the fitted training-only baseline. Both arms' bc_metrics.jsonl retain20 logged sampled-batch loss/gradient/clipping observations, every50steps. Action clipping and predicted target-step occupancy are not actual torque saturation. Undefined zero-variance correlations/baseline ratios are null. No shifted-target hard gate was introduced.

The original steady/onset/zero dataset is unchanged, has no genuine walking-to-stop examples and excludes heldout cycle3 from fitting. Cycle3 remains a selection-conditioned temporal neighbor from the same trajectories, not independent validation. Root's native fit002 tests still failed walking/quiet/stop; calibration is a bounded intervention, not a diagnosis or demonstrated repair. Zero-command action fit slightly worsens, so quieter native behavior must not be inferred. Original-reset forward20s, quiet20s and move8s→stop13s probes remain the next native comparison under unchanged gates; other required cases remain outstanding.
