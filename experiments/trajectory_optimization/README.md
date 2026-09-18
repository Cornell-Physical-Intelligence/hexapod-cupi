# Optional forward-reference experiment

You use this package to optimize and replay one forward cycle, or reproduce the
recorded action-imitation comparison. The maintained simulator, PPO learner,
evaluator and Spark guard live in [`locomotion/`](../../locomotion/README.md).

## Code and scope

| File | Responsibility |
| --- | --- |
| `model.py` | Read approved masses and inertias; calculate floating-base Newton–Euler inverse dynamics. |
| `optimize.py` | Solve a 1.2-second tripod cycle at 0.05 m/s with CasADi/IPOPT and audit the stored arrays. |
| `prepare.py`, `replay_native.py` | Bind the cycle to the kernel and replay motor targets through native physics. |
| `forward_experiment.py`, `prepare_forward.py` | Prepare the recorded scratch-versus-action-imitation comparison using stock PPO. |

The optimizer constrains friction, force balance, joint travel, speed, the
1.6 N·m torque cap and the 0.040 rad target-change bound. Point contacts omit
native mesh patches and impacts. Midpoint feasibility needs native validation.
This package implements reference generation inspired by
[Liu et al.](https://arxiv.org/html/2511.03167v1#S3.SS1); it supplies no paper reproduction.

## Recorded conclusions

The [first native replay](../../artifacts/trajectory_optimizer_20260917/review_001/RESULT.json)
completed 1,000 controls but failed speed tracking. The
[improved reference](../../artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
passed the forward screen at 0.05 m/s with 0.003029 m/s planar error. It remains
an executed target sequence, without learned feedback or omni qualification.

The [standard PPO comparison](../../artifacts/ppo_reference_comparison_20260917/COMPARISON_001.json)
failed forward, quiet and stop probes after 1,200 updates. The
[paired initialization comparison](../../artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
also failed: both arms failed their scheduled forward screens, including the
copied actor before PPO. The result establishes no benefit from that initialization.
See [TRAINING](../../docs/TRAINING.md) for the current conclusion and measurements.

## Prepare a new allocation

Run the CPU optimizer with a fresh output directory:

```sh
uv run python -m unittest discover -s experiments/trajectory_optimization/tests
uv run python -m experiments.trajectory_optimization.optimize --help
uv run python -m experiments.trajectory_optimization.prepare --help
uv run python -m experiments.trajectory_optimization.prepare_forward --help
```

Native packers require `--inputs` from matching kernel standing admission.
Pass the resulting input declaration with the trajectory or experiment protocol
and fresh local/remote output directories. The packers call `locomotion.prepare`;
start their frozen `locomotion.launch` module under the procedure in
[OPERATIONS](../../docs/OPERATIONS.md). Preparation starts no compute.

The evaluator records `force_metrics.json`, raw contacts, requested/applied
motor torque and selected policy video. Preserve failed captures. Historical
checkpoints require their original frozen sources and declaration schemas.
The [pinned experiment guide](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/experiments/trajectory_optimization/README.md)
retains the completed protocols, original reproduction commands and archive
restoration procedure. Git history holds the run narrative; this guide owns
current entry points.
