# Independent residual-PPO session and CPU lifecycle review

The reviewed session timing seam and actual RSL 5.0.1 two-update CPU lifecycle pass independent tests. This receipt covers the source hashes in [review.json](review.json), including the two small session corrections described below. It does not approve the still separately prepared physical entrypoint, exploration calibration, admission guards or a GPU training run.

## Command and next-observation timing

Five tests execute the actual `ReferenceResidualSession` class methods extracted from the candidate source, together with the frozen wave005, residual002 and 846/849 observation builder. Physical stepping, capture, recorder and startup dependencies are explicitly synthetic/mocked. They cannot establish physical support, motor behavior or startup trajectory admission.

The tests verify:

- The optional initial command is copied at construction, remains inactive for all 200 startup calls, and enters the first encoded policy packet before the first post-startup action. Mutating the caller's original command tensor cannot change it later.
- A queued command leaves the current cached packet unchanged. The next current control still uses its previously observed command and reference; only afterward does the queued command enter the next observation. The following action/reference step uses that command. A queued stop follows the same explicit 20 ms timing.
- Invalid bearings, speeds and nonfinite commands reject without rewriting the current packet. Returned observations are copies and cannot mutate cached state through a caller alias.
- Once a session fails, a retry cannot advance the planner, emit another physical command or return a valid packet. Its first-failure planner state remains intact.
- Stand-only optimization cannot execute a 49th control or accept a pending movement command.

Two small code-review findings were corrected in the owner before the final tests: `initial_commands` now clones the validated tensor, and `step()` rejects an already failed session before calling the wave generator. The command queue timing itself remains unchanged. The next packet legitimately contains the newly requested command alongside the preceding executed reference state; this preserves action/reward/reference timing without rewriting an already encoded same-step history entry.

The final bound run passes all five tests; see [bound_seam_tests.log](bound_seam_tests.log). An earlier test invocation omitted the isolated dependency path and failed to import `tensordict`; that dependency-only attempt is preserved in [seam_dependency_attempt.log](seam_dependency_attempt.log). Correct dependency-path runs pass. No source failure is inferred from that import attempt.

## Actual RSL runner, synthetic measurement fixture

The independent regression imports the actual local RSL 5.0.1 wheel. The five runner/algorithm/logger/model/utility file hashes match the recorded installed Spark sources, and the complete frozen observation005 bundle verifies. It runs 32 synthetic replicas for two PPO updates of 24 controls, using the actual frozen bounded residual core and 846/849 encoder. Fixed synthetic measurements and an artificial gradient-producing reward are deliberately not a physical locomotion task.

A fresh zero actor mean head and Adam optimizer are created. The early initial checkpoint succeeds with the known deferred-logger initialization. After 48 controls, actor weights have changed and all 17 optimizer state entries exist. Deliberately perturbing actor and Adam state, then strictly reloading the final checkpoint, restores actor/critic/normalizers/optimizer exactly and produces bit-identical deterministic actions. Wrong physical-source identity, wrong observation schema, an existing checkpoint and an orphan sidecar are rejected. The fixture retains raw reported joint rate 0.017 rad/s alongside zero angle-derived interval rate; the channels are not silently merged.

[The bound regression report](bound_cpu_regression/report.json), [full log](bound_rsl_tests.log), and initial/final checkpoints with exact sidecars are retained. They are labelled synthetic CPU lineage and cannot serve as physical training checkpoints. The earlier independent regression is also preserved under `cpu_regression`; it reached the same pass conclusion.

## Binding and scope

[run_bound_review.py](run_bound_review.py) hashes all six reviewed candidate files before and after running both suites and requires exact equality. It verifies the original observation005 inventory and the five installed-source parity receipts. All assertions pass; [bound_review.log](bound_review.log) records the result.

The candidate remains a new finite position-residual consumer with its own source/checkpoint identity. Frozen prototype flags and the 846/849 schema were not edited. The reviewed session admits only the demonstrated 0–0.005 m/s forward/stop command envelope; new directions, sustained quiet, exploration and longer optimization require the owner's separately implemented physical gates. This receipt launches no GPU and makes no Stage2, velocity-fidelity or walking-success claim.

Reproduction from the original repository layout (use fresh output paths or a copy to preserve these frozen results):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps .venv/bin/python -m unittest discover -s tmp/reference_residual_ppo_independent_review_001 -p test_session_seam.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps .venv/bin/python tmp/reference_residual_ppo_001/real_rsl_cpu_regression.py --observation-bundle tmp/reference_policy_observation_005_001 --output /absolute/fresh/cpu_output
```
