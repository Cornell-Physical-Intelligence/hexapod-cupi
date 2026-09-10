# Consumer003: recovery proof before a moving learner

This is a new, unlaunched consumer and checkpoint lineage. It preserves the
failed consumer001 profile and the frozen consumer002 reset-context repair.
No physical acceptance result is reinterpreted. The new phase tests whether the
learning infrastructure handles a finite task failure correctly; it is not a
claim that the controller already walks successfully in all replicas.

The admitted command interface remains forward/left/yaw, with this first pilot
restricted to forward 0.005 m/s and zero command. Actor/critic widths remain
846/849 and actions remain 18 finite position residual goals. Motor cap, position
gains, reference, quiet/retention gates, normalization, PPO loss and reward are
unchanged. New checkpoints use `c_wave005_recoverable_moving_residual_PPO_v2`
and exact source/plan/admission bindings. No predecessor checkpoint can resume.

## Admission group: zero PPO updates

The outer host runs three separate phases and then releases the GPU:

1. Original `standing`: fresh 32 robots, 1,000 controls, all original physical
   and quiet bounds.
2. `calibrate`, folder `calibration`: original zero-mean and sampled quiet
   calibration, 1,000 controls each after 200 startup controls. Initial standard
   deviation is 0.02; fresh weights, normalizers and Adam. Only passing trials
   can save immutable `initial.pt` and sidecar.
3. `learning_recovery_32`: 200 startup plus 512 zero-residual controls, allowing
   finite task failures to terminate affected training episodes. Each recovered
   row performs the original reset followed by the exact 200-control canonical
   recovery. At least one actual finite failure must complete that recovery.
   Every reset/history/critic ledger check must pass. No rollout storage,
   normalization or optimizer updates occur in this phase.

The new recovery phase does not have the old profile's no-event requirement.
The old `profile_32` and `profile_128` modes retain that rule unchanged and are
not initial learning prerequisites in this protocol. No numerical task-failure
rate threshold is invented from the incomplete failed prefix. Actual per-row
events, active/recovery counts, progress and timing require review.

Fatal sensor/data/source/actuator or failed recovery conditions still reject
the entire run. Finite task failures remain failed episodes with exactly one
event penalty, zero terminal bootstrap and no learning during recovery. Actual
400 Hz torque samples remain authoritative for substep motor events; a safe
50 Hz endpoint does not erase an interior spike.

## Compact actual evidence

`raw/learning_ledger/` contains immutable event NPZs and `summary.json` with
their hashes. Reset events store actual final actor/critic packets, episode,
validity, selected mask, zero post-reset actor packet, and all-row histories and
history-valid/interval-valid/step metadata before and after reset. Unselected
histories must be byte-exact through the reset call.

At completed recovery, the selected history is cleared before its first packet,
then contains exactly one valid frame, step zero, the new episode and an invalid
first interval-angle rate. Every actual finite recovery is linked to its reset
200 physical controls earlier. Raw SDK rates remain independently present.

The bootstrap record stores the actual final critic packet and the value used
by the shared computation. During admission it is explicitly labeled an
unoptimized probe, with no PPO storage claim. During training the record is
emitted after adding the actual transition to masked PPO storage. True terminal
value is zero; timeout uses the final valid same-episode critic packet, never
the reset actor packet. Per-update normalization counts must equal valid
learning-transition counts.

The reset context matches installed SDK inference semantics. Selected reset IDs
are int32, matching the inspected DirectRLEnv `.nonzero(...).int()` path. Actual
sensor clocks must match the source-bound reset kernel and subsequent eight
float32 updates; no clock freshness is invented.

The phase also writes top-level `learning_integrity.json`. It binds the exact
identity, ledger/hash, and an `infrastructure_passed` boolean. It always declares
`physical_admission: false` and `automatic_training_allowed: false`.

## Explicit allocation review

Both host and entrypoint call the stdlib-only
`campaign_contract.validate_learning_decision(args, identity)`. `args.output`
is the future `CAMPAIGN/train_10` directory. The required `--decision-receipt`
must be outside the campaign, mounted read-only, and contain:

```json
{
  "schema": "moving_PPO_learning_admission_review_v1",
  "identity": "the complete exact actual identity object",
  "accepted": true,
  "approved_updates": 10,
  "phase_state_sha256": {
    "standing": "actual SHA256",
    "calibration": "actual SHA256",
    "learning_recovery_32": "actual SHA256"
  },
  "learning_integrity_sha256": "actual top-level evidence SHA256",
  "event_and_recovery_review": true,
  "per_environment_progress_review": true,
  "timing_and_memory_review": true,
  "raw_evidence_integrity_review": true,
  "no_gate_relaxation": true
}
```

This is a documentation template, not an approval receipt. The verifier rereads
all completed phases and exact checkpoint/source/evidence bindings before
allocation. An admission pass cannot create this review automatically.

## Learning group after actual review

The second host group runs `train_10`, `evaluate_initial`, `evaluate_010`, and
`quiet_010`. Ten updates each collect 256 controls in 32 replicas, with recovery
rows excluded from normalization, GAE and minibatches. Immutable checkpoint
`decision_010.pt` must reload actor, critic, normalizers and all Adam state
exactly. Cold evaluations retain the original 2,400-control forward/stop and
32-replica quiet gates. No ten-update result completes Stage2 or proves useful
speed. Faster support-pattern work remains separate and necessary.

The entrypoint dependency flags are unchanged; the new mode uses folder
`learning_recovery_32`. `--decision-receipt` is required only for learning
allocation (`train_10`, or the distinct existing 10-to-25 review for `train_25`).
All outputs are immutable, source/assets remain read-only, and root owns every
Spark allocation and restoration.
