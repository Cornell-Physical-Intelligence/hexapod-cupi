# Forward-example PPO experiment

We compare two policies under the [frozen protocol](protocol_001/PROTOCOL.json).
Both use the approved direct-drive robot and a fixed 0.05 m/s forward command.
We give each policy 1,200 PPO updates, or 3,686,400 transitions, on 128 replicas.
We use seed 20260914 and the same reward, physics and PPO settings.

| Arm | Initialization before PPO |
| --- | --- |
| `scratch` | Keep the initial actor weights. |
| `example` | Fit the actor's action predictions from the recorded forward example with 1,000 Adam updates. |

We initialize both input normalizers from the first 700 example rows. We use
the next 300 rows to measure prediction error on the same recording. This
prediction check does not test whether the policy can walk. We preserve the
critic, action variance and PPO random state during the fit. We discard the
imitation optimizer before PPO. We use no imitation reward during PPO.

The [initialization audit](PAIRED_INITIALIZATION_001.json) binds the initial
checkpoint weights and optimizer states. The
[scratch receipt](SCRATCH_TRAINING_001.json) and
[example receipt](EXAMPLE_TRAINING_001.json) record both completed training
budgets and 29,491,200 force/torque samples per arm. Each receipt records its
own observation time; it does not describe current host resources.

## Comparison and scope

The [paired comparison](COMPARISON_001.json) establishes no benefit from this
example initialization. Both arms fail the forward screen at updates 0, 300,
600 and 1,200. Neither arm reaches a passing scheduled checkpoint.

| Final forward measurement | Scratch | Example initialization |
| --- | ---: | ---: |
| Mean forward speed, target 0.05 m/s | 0.004811 m/s | 0.000961 m/s |
| Planar tracking error, limit 0.025 m/s | 0.051045 m/s | 0.053322 m/s |
| Mean vertical ground support | 73.246 N | 73.227 N |
| Ground support, 95th percentile | 94.065 N | 87.808 N |
| Mean absolute applied motor torque | 0.488782 N·m | 0.405900 N·m |
| Highest joint RMS applied torque | 1.589046 N·m | 1.373609 N·m |
| Forward screen | Fail | Fail |

The final recordings show the [scratch policy](scratch_evaluate_update001200_001/run/standing/evaluation_00/rollout.mp4)
and the [example-initialized policy](example_evaluate_update001200_001/run/standing/evaluation_00/rollout.mp4).

Both final policies exceed the allowed fraction of motor demand above its
rating. The scratch policy also fails the native physical checks. The example
policy passes those native physical checks but fails forward tracking and
motor demand. We retain each original verdict in the comparison.

We can execute the prepared forward trajectory on this model. Fitting action
predictions from that trajectory, followed by this PPO budget, does not produce
a passing forward policy in this pair. The prediction fit fails the initial
walking test before PPO starts. This experiment does not identify the cause of
that failure or establish a result for other imitation methods or seeds.

We evaluate updates 0, 300, 600 and 1,200 from both arms. For each evaluation,
we request a 20-second forward trial, a 20-second quiet trial and a 21-second
forward-to-stop trial. We preserve early physical terminations as failed
prefixes with their recorded and requested lengths. We retain the 0.040 rad
per 20 ms limiter and the existing
physical and behavior gates. We use deterministic actions from each saved
checkpoint. Quiet and stop remain diagnostics because training uses a forward
command throughout.

Both final policies fail the quiet and forward-to-stop diagnostics. The final
example quiet trial ends at control 490 of 1,000 after a native termination.
Its forward-to-stop trial records all 1,050 requested controls and fails the
quiet-stop limits. We preserve the shorter quiet capture and its failure.

The reservation guard interrupted the first example-arm evaluation at update
600 after a competing CUDA process appeared. We preserve its completed forward
and quiet reports and unfinished stop capture. The [retry receipt](EVALUATION_RETRY_001.json)
binds attempt `002` to the same checkpoint, seed and source. We repeat all
scheduled cases. The interrupted allocation supplies no complete stop result.
We [verified the completed retry](EVALUATION_RETRY_VERIFICATION_001.json): the
first two control traces match the interrupted attempt byte for byte, and
their numeric verdicts and force summaries match. The retry supplies the
complete stop record. All three retry trials fail their existing gates.

We apply the protocol's benefit rule to the final forward results. A passing
example policy with a failing scratch policy supports benefit for this paired
seed. If both final policies pass, an earlier first passing scheduled checkpoint
for the example policy supports a learning-speed benefit. Other outcomes do not
establish the declared benefit. A pass at update zero establishes imitation;
later evaluations test whether PPO retains that behavior.

We record the extra 1,000 imitation updates and demonstration acquisition
apart from the equal PPO budgets. This experiment tests one initialization
method with one seed and one forward command. It does not reproduce AMP or
qualify Stage 2. The older mixed-command PPO run is outside this pair.

We also [summarize the training logs](TRAINING_DIAGNOSTICS_001.json) over the
first and last 100 updates. Both arms increase their mean task reward, while
late-training forward speeds stay below the 0.05 m/s command. We label these
post-hoc diagnostics because we selected the summary windows after training.
They describe stochastic actions from changing policies and do not replace
the scheduled deterministic policy tests or change the benefit rule.

## Force and torque records

Each trial has `force_metrics.json` and a 400 Hz raw capture. We report total
vertical support in newtons and contact-normal force per foot. A foot's
full-window average includes zero force during swing; the contact average
uses samples above 1 N. We report mean absolute and RMS motor torque in N·m,
with requested and applied torque kept separate.

For the forward comparison, we use the 18 seconds after the two-second
settling interval. The full-trial and startup windows remain in each summary.
We compare load with achieved speed. Low torque in a policy that fails to move
does not establish energy efficiency. The capture contains contact-normal
forces; it does not contain tangential friction forces or hardware load tests.

## Preserved files and reproduction

Each evaluation pack contains its binding, frozen source and checkpoint hash.
The `run/standing/evaluation_00`, `evaluation_01` and `evaluation_02`
directories contain the forward, quiet and stop records. Each record includes
the original MP4, report, force summary and control trace. Transfer receipts
bind the complete remote run inventory to SHA-256 hashes. Audit receipts bind
the video and report hashes to the saved policy.

We preserve complete raw runs on this workstation and on Spark under
`/home/orionh/HEXAPOD_runs/restart_20260914/forward_example_ppo_20260917/`.
We publish the two final forward MP4s in Git. We keep the remaining original
MP4s, large `native400hz/*.npz` arrays and `contacts.jsonl` files in both full
archives. We retain their hashes in each pack's `TRANSFER_VERIFICATION_001.json`;
the interrupted allocation uses `FAILED_TRANSFER_VERIFICATION_001.json`.
We preserve selected checkpoints at updates 0, 300, 600 and 1,200 in Git.
Intermediate checkpoints and duplicate learner files stay in the full archives.
Compressed logs preserve the raw bytes, with hashes in the archive receipts.

To audit a fresh checkout, restore the raw run from the remote path in its
transfer receipt, then check each file against that receipt's `files` mapping.
Use `rsync -a --ignore-existing` to preserve local files. Restore compressed
`native/native_readback.json.gz` to its original name if you use the published
payload without the full remote run. These commands audit the scratch policy
at update zero; use the declared arm and update for each remaining checkpoint:

```sh
uv run python -B artifacts/forward_example_ppo_20260917/review_001/verify.py \
  --arm scratch --update 0 --output /tmp/scratch_update000000_audit.json
```

The audit checks inference against the recorded actions, actuator output,
contact classification, original scoring and force summaries. It preserves
failed behavior verdicts. Each audit output needs a fresh path because the
verifier refuses to replace an existing result.

For the example-arm update-600 retry, use `review_001/verify_002.py` with
`--arm example --update 600 --attempt 2` and a fresh output path. The successor
checker adds allocation selection and provenance; it retains the numerical
and physical checks. Each successor audit records its selected evaluation pack.

The comparison script consumes the eight recorded audit files:

```sh
uv run python -B artifacts/forward_example_ppo_20260917/review_001/compare.py \
  --output /tmp/forward_example_comparison.json
```
