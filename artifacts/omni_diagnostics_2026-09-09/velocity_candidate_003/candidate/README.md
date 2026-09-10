# Candidate 003: actual RSL early-checkpoint lifecycle fix

Candidate 002 passed the full standing gate and both 32-replica calibration trials, including std 0.005 for 20 seconds with zero requested-torque saturation, terminations or truncations. Its runner smoke failed before optimization with `AttributeError: 'Logger' object has no attribute 'writer'`. That failure and all source 002 bytes remain preserved.

The exact installed **RSL-RL 5.0.1** logger does not create `writer` in its constructor. `OnPolicyRunner.save()` writes a checkpoint, then calls `Logger.save_model()`, which reads `writer`. `learn()` initializes it later. Our intentional pre-learning checkpoint exposed this API lifecycle defect; it was not a motor or gait failure.

Only `candidate_runner.py` changes among the six candidate files. Before an early checkpoint, an absent `writer` is initialized to `None` **only for the configured local TensorBoard logger**. The normal `learn()` call still creates its real writer once. Existing writers are preserved; unsupported external logging configurations fail explicitly. Model weights, normalization, optimizer settings, source binding, immutable checkpoint checks, std 0.005, 20-second calibration, all physics and acceptance gates remain unchanged. `candidate003_minimal.patch` shows the complete change from source 002.

## Real installed RSL regression

The real installed runner was executed in the Spark image using `--runtime=runc`, CUDA hidden, one CPU and a 3 GiB memory limit. It did not request or use a GPU and ran alongside root's separately owned terrain job. The vector environment was an explicitly synthetic CPU fixture using the actual candidate action state; this test cannot qualify physical contacts or locomotion.

The regression:

1. Reproduced the original `Logger.writer` exception using the real unmodified `OnPolicyRunner.save()` before learning.
2. Initialized a separate 495/498 scratch actor/critic with zero actor mean and std 0.005.
3. Saved its immutable initial checkpoint through the corrected helper.
4. Completed **two actual PPO updates / 48 rollout steps** using the installed 256/256/128 networks, 24-step rollouts, five epochs, four minibatches, fresh Adam and the original adaptive learning-rate schedule.
5. Saved the final checkpoint, deliberately perturbed model and optimizer state, then loaded strictly. All actor/critic and Adam state matched, and deterministic actions were identical with maximum difference **0.0**.
6. Verified normalizer inference-buffer compatibility and rejected wrong source identity and corrupted checkpoint bytes.

The report records **17 optimizer state entries**, learned changes to every actor MLP tensor, `cuda_available: false`, RSL-RL 5.0.1 and Torch 2.10.0+cu130. See [real_rsl_cpu_report.json](real_rsl_cpu_report.json), [real_rsl_cpu_regression.log](real_rsl_cpu_regression.log), and [real_rsl_cpu_regression.py](real_rsl_cpu_regression.py). `installed_agent.yaml` is the actual candidate 002 runner configuration, changed only to CPU device inside this test. `installed_logger_source.py.txt` preserves the exact inspected module and its copyright/license header.

The synthetic checkpoints remain under `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_velocity_cpu_003/attempt_001`; they have an explicit CPU-fixture identity and must never be used as robot checkpoints.

Thirty-one regular CPU tests also pass:

```sh
.venv/bin/python -B -m unittest discover -s tmp/omni_velocity_candidate_003 -p 'test_*.py' -v
```

To repeat the real integration, copy only `candidate_runner.py`, `velocity_action.py`, `installed_agent.yaml` and `real_rsl_cpu_regression.py` to a fresh CPU-test input directory. Mount it read-only at `/test`, mount a separate writable output directory, and run the installed Isaac Python entrypoint with `real_rsl_cpu_regression.py --output <fresh path>` under `runc`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=-1`, one CPU and 3 GiB. No `AppLauncher`, simulator or GPU is initialized.

## Fresh physical admission still required

The six-file overlay, bounded validate→probe host launcher and source builder retain the reviewed candidate 002 std/physics/plan. The new code identity requires a fresh matching standing admission and calibration before its actual two-update physical runner smoke. Root owns dispatch and identified forecasting pause/restoration; agents have not launched a GPU job.

```sh
.venv/bin/python -B tmp/omni_velocity_candidate_003/build_source.py --candidate tmp/omni_velocity_candidate_003 --manifest tmp/omni_velocity_candidate_003/FREEZE_SHA256.json --output tmp/omni_velocity_candidate_003/source_003
```

There is no automatic walking extension or new noise bracket. A later 50-update scratch pilot and external progress recording require root's review of the actual completed probe. Source 002-bound walking/recording launchers must be explicitly rebound to the new source identity; they cannot silently accept source 003. Stage 2 remains incomplete.
