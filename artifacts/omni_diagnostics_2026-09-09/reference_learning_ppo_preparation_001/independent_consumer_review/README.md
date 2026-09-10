# Independent moving003 recovery / critic review

**No concrete blocker found for the bounded, unoptimized recovery infrastructure screen.** This is CPU/source review clearance, not a physical gait pass, Stage 2 completion or permission to allocate PPO updates automatically.

The reviewed owner is `reference_moving_ppo_003`, final **51-payload** freeze **fd9bef87dda976f9b229e7541408d24e674245ca2c51231dfce9075b81d38334**. All final owner payload hashes were independently verified. No owner file, GPU process or tracked document was changed.

Seven focused tests independently passed:

- Five learning-integrity / allocation tests, including the real RSL 5.0.1 two-update callback and normalization regression. The CPU session test executes 200 startup + 512 recovery-probe controls with one finite terminal, selected physical reset and 200-control recovery. Synthetic physics is explicit.
- Two inference-reset tests reproduce the original inference-buffer exception, then verify the corrected actual reset path uses **int32 row IDs** inside the narrow `torch.inference_mode()` scope and returns to the original outside scope.

The reviewed reset preserves the final observation and episode before calling the inherited physical reset. The selected rows publish excluded zero placeholders in their next actor/critic packet. Unselected history stays unchanged during the selected reset. After exactly 200 actual controls, the recovered row clears old history, starts at step zero with one valid history frame and keeps the interval-angle rate invalid for that first frame.

The new ledger matches the final critic packet and next-episode placeholder to the unique same-control reset event. True terminal rows receive zero bootstrap. Time-limit transitions retain the final, pre-reset critic input from the same episode; the real RSL regression checks its selected-batch numerical value. Recovery rows are excluded from normalizer updates, returns and minibatches; finite terminal transitions remain learnable.

The `learning_recovery_32` entry point records the actual callback evidence while leaving PPO storage, optimizer state, normalization updates and learning iteration untouched. It requires at least one finite event and completed recovery. A new `train_10` allocation requires an external, exact-source root review receipt binding all three completed phase states and the exact integrity evidence, plus explicit event/progress/timing/raw-integrity review. The old cold evaluation gates remain unchanged.

The tests exercise actual installed RSL Python code and reproduce the SDK's documented reset-buffer allocation semantics with a synthetic environment. They do not prove the next Spark run succeeds, classify a recovered episode as admitted locomotion, establish instantaneous velocity fidelity, or change observation widths. GPU event/clock/physics evidence and the separate root allocation decision remain required.

Reproduction:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/rsl_deps /path/to/python -m unittest discover -s /path/to/reference_moving_ppo_003 -p 'test_learning*.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/rsl_deps /path/to/python -m unittest discover -s /path/to/reference_moving_ppo_003 -p 'test_inference_reset.py'
python verify.py /path/to/reference_moving_ppo_003
```

[review.json](review.json) binds the final owner and methods reviewed; the two logs retain the complete independent test outputs. The verifier is read-only and uses the standard library.
