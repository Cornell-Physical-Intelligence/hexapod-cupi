# Actual moving PPO failure after nine updates

The tenth collection stopped at control 2,600 because **row 8's RM tibia requested torque remained above 1.6 Nm at the end of its 200-control timeout recovery**. All 32 rows had six distal contacts. The 26 rows completing timeout recovery had no nonfoot contact or native termination. No decision-010 checkpoint or matched policy screens were produced.

The last eight 400 Hz samples peaked at **1.62764847 Nm** on runtime joint `revolute_2_1` (index 17, resolved from the recorded name order). All applied torque stayed within the actual float32 representation of 1.6 Nm. Applied clipping does not make excessive requested demand pass the unchanged recovery check.

The failure is persistent, not one contact flicker or isolated sample. Over the final two seconds, the 50 Hz RM tibia demand stayed between 1.6274 and 1.6337 Nm while the executed target was exactly canonical and residual action/position were zero. At the final measured position, the Kp30 position-error contribution alone is approximately −1.61155 Nm; the Kd0.6 reported-rate contribution is approximately −0.01598 Nm. This endpoint decomposition is diagnostic, since actuator computation precedes endpoint integration. It nevertheless rules out treating the small velocity term as the sole explanation. Retain both raw SDK and angle-derived rates; do not replace the torque or quiet metrics.

At control 2,400, 26 rows reached their 2,200-active-control timeout and invoked the original inherited reset. The first following reference knot agrees with the existing two-second quintic from the recorded randomized reset target, followed by two seconds of canonical hold. Row 8's root position returns to the initial reset position within 0.24 µm; its XYZW quaternion is exactly equal. The final target is exactly the same canonical target that passed its initial recovery. These records do not show stale learned residual, target windup or a reset-frame error.

The supported inference is **sensitivity of the loaded canonical equilibrium to the randomized reset/contact history**. It is not proof of a native PhysX cause or proof that the nominal stance is infeasible under every initialization. The last RM normal force is about 26.19 N while RF carries about 4.06 N, despite all feet contacting. A small body-pose change accompanies a substantially different load distribution.

## Smallest useful next test

Do not rerun the unchanged ten-update job or merely lengthen settling: the last two seconds already show a nearly stationary over-limit equilibrium.

Prepare a separately named, zero-residual reset comparison. First reproduce the failing randomized joint target (recorded in report.json), with the same root pose, gains, actuator cap, scene and canonical recovery. Then compare an explicitly declared **canonical-joint reset distribution**, so both actual initial joint position and first emitted target equal the canonical stance. All reset writes occur only inside the selected-row reset; no body motion is prescribed during stepping. The quintic/hold machinery may remain, becoming a constant target for this branch.

Before learning, verify repeated full-32 and mixed-row resets, actual selected/unselected joint/root state, first-target continuity, all sensor epochs, exactly 200 recovery controls, per-row contact and all eight-substep torque checks. Preserve each failed attempt; do not quietly retry a row until it passes. This proposal changes training reset initialization, not the physical asset or the original evaluation gates. If a canonical reset still produces excessive demand, the test rejects it and motivates explicit load-transfer/control work rather than a higher limit.

After an admitted reset correction, any renewed ten-update allocation and checkpoint policy needs root review. The nine completed updates are evidence that PPO executes, not that useful walking improved. Consider preserving bounded intermediate diagnostic checkpoints in the next source, clearly marked unqualified, so a later infrastructure rejection does not erase all trained weights; this is separate from admitting them.

## Reproduce

Run the read-only analyzer against the fully fetched raw tree and write a new report path:

```sh
python3 tmp/reference_learning_train001_failure_review/analyze.py \
  --input tmp/reference_learning_ppo_train_results_001 \
  --output tmp/reference_learning_train001_failure_review/replayed_report.json
```

The report binds the exact state, trace, 400 Hz substeps and episode ledger hashes. It checks all torque/joint-position control endpoints against the substep stream, the final failure classification, canonical target equality and first reset quintic. Full source/input/cleanup/restoration audit belongs to root's separate terminal bundle. No frozen source, acceptance gate, GPU process or main file was modified.
