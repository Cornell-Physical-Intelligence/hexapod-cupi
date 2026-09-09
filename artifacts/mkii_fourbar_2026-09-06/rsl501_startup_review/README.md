# RSL-RL 5.0.1 startup and checkpoint API review

This review found and repaired two independent blockers before the physical
robot task reached PPO. The CPU runs here use a synthetic eight-environment
linear recurrence. They are **not robot training, physics validation, an
admission report, a locomotion policy, or a Spark throughput benchmark**.

## Repairs

1. Isaac Lab's `RslRlMLPModelCfg` still serializes deprecated `stochastic`,
   `init_noise_std`, `noise_std_type`, and `state_dependent_std` fields.
   RSL-RL 5.0.1's actual `MLPModel` constructor rejects them. The trainer now
   calls the installed Lab `handle_deprecated_rsl_rl_cfg` adapter before
   producing the runner dictionary. The policy remains a 256/256/128 ELU MLP
   with Gaussian initial standard deviation 0.15; actor and critic observation
   normalization stay enabled. No PPO or physics settings were relaxed.
2. During rollout, RSL 5.0.1 assigns `obs_normalizer._std` inside
   `torch.inference_mode()`. The trainer's immediate strict checkpoint reload
   then fails when `load_state_dict` copies into this inference tensor outside
   inference mode. `prepare_algorithm_buffers_for_load` clones only inference
   buffers into normal tensors before loading. Values, shapes, dtypes, and
   devices are preserved; parameters and optimizer state are not converted.
   Loading remains outside inference mode, so continued Adam updates receive
   ordinary tensors. The pre-existing adaptive-learning-rate restore remains.

The original CPU checkpoint failure is preserved in
`cpu_api_failure_001.json` and `cpu_api_failure_001.log.txt`. It occurred after
two successful synthetic PPO updates and before strict reload completed.

## Evidence and limits

- `sources.json` lists 14 read-only captures from the installed Spark stack,
  with byte hashes. Five RSL modules (runner, PPO, model, logger, utilities)
  match the official PyPI `rsl-rl-lib==5.0.1` package exactly.
- `cpu_api_probe.py` executes the captured Lab configclass, serialization,
  deprecated-config adapter, and vector wrapper sources. It avoids native
  package initializers; unused native array conversions are disabled with
  fail-on-use dictionaries, and the wrapper's environment type is synthetic.
  No Isaac, PhysX or CUDA session starts.
- `cpu_api_pass_002.json` records source hashes, dependencies, complete emitted
  configurations, finite losses, parameter digests, normalizer keys and
  successful immediate/fresh-runner checkpoint checks. The synthetic sequence
  is two updates, strict full-state roundtrip, a further update in the same
  runner, a new checkpoint, then strict fresh-runner resume and one further
  update. Logged iteration numbers are exactly 0, 1, 2, 3. Deterministic
  inference is finite and repeated calls return identical actions.
- `cpu_api_pass_001.json` is the preceding successful probe before input hashes
  were added to the report. Both successful runs and the failed attempt are
  retained. Temporary synthetic checkpoints/event logs are discarded by the
  probe and must never be used as robot checkpoints.
- The normalizer implementation is included separately as
  `upstream_rsl501_normalization.py.txt`. Its provenance is the official PyPI
  wheel, not an independently captured Spark copy. Local supporting dependency
  versions are recorded and are not claimed identical to the complete Spark
  dependency stack.

## Remaining API checks

| Interface | Observed behavior / verification |
| --- | --- |
| Runner construction | Actual unadapted configuration reproduces `unexpected keyword argument 'stochastic'`; adapted configuration constructs actual actor, critic, storage and PPO. |
| Vector wrapper | Captured wrapper returns TensorDict observations and four-item RSL step results; termination and timeout values are combined and timeout bootstrapping information is retained. |
| Learning | A single `learn(n)` call executes `n` additional iterations from `current_learning_iteration`; 24 samples per environment per update are observed. |
| Logger | Installed callback accepts `it`, `start_it`, `total_it`, collection/learning durations, losses, learning rate and action standard deviation. Actual `value`, `surrogate`, `entropy` losses are finite. |
| Checkpoint | Actual production `GuardedRunner` class is extracted from the trainer and exercised, including source sidecars, SHA-256, strict load and next-iteration metadata. Synthetic contract is explicitly nonphysical. |
| Resume | Actor, critic, normalization buffers, optimizer moments and adaptive learning-rate state hash identically across immediate and fresh-runner loads; both paths permit further optimization. |
| Inference | `get_inference_policy(device="cpu")` returns a deterministic callable accepting the captured wrapper's TensorDict, with finite 8-by-18 output. |
| Physics lifecycle | Not exercised by this artifact. Existing every-substep training-guard tests remain required, followed by real Isaac qualification and robot PPO. |

## Reproduce

The regular repository environment runs the five focused regression checks
without adding RSL or Isaac dependencies:

```sh
.venv/bin/python -m unittest discover -s isaaclab/tests -p 'test_mkii_fourbar_runner_config.py'
.venv/bin/python -m unittest discover -s isaaclab/tests -p 'test_mkii_fourbar_training_guards.py'
```

These passed: 5 new configuration/reload tests and 16 existing training-guard
tests. For the optional actual CPU API probe, install RSL 5.0.1 and CPU
dependencies in an isolated environment (versions are in the passing report),
then run:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python cpu_api_probe.py --report /tmp/new_cpu_api_report.json
```

The script refuses to overwrite its report. Do not modify the archived
reports or their manifests when rerunning it. A new runtime source identity
and fresh physical qualification are required before these trainer repairs
can enter the guarded campaign.
