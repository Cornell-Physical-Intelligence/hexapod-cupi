# BC data and native/PPO boundary recommendation

Advisory only. Root owns dispatch and experiment adoption. This does not modify any source, checkpoint, dataset, simulator state or existing numeric gate.

## First measure the unchanged fitted actor

The useful next diagnostic is the existing BC checkpoint `f9d32d7f…` on the original canonical reset, actual measured history, deterministic mean action, admitted one-robot physics and original scorers. Do not run PPO, replace a latent, insert a gait phase or alter reset/normalizer behavior first. The existing entry reads Config from checkpoint and calls strict load; use the original accepted AMP prior/report, bc_steps argument zero, and no initial-std override. The checkpoint itself declares std 0.1 and 32-environment training configuration; evaluating the actor on one admitted robot is supported without changing that learning configuration.

Root's later allocation choice—forward 20 s, quiet 20 s, and forward 8 s followed by zero 13 s—is a sensible first subset of the 13 additional probes. Keep the other ten explicitly missing and aggregate diagnostic pass false. Broaden when informative. The full 13 additionally cover eight 0.05 m/s bearings, both yaw signs and the 32 s quiet case, but do not replace the full 96-case Stage 2 suite, arcs, higher commands, .025 starts or all-direction stop/continuous-transition cases.

Bind actual evaluation to checkpoint/source/model/prior hashes, bc_steps 1000, PPO updates/transitions zero, unchanged controller/limiter, original reset and selected/missing case IDs. Preserve any terminal prefix and exact original failures. Record actual video, signed motion, joint/target spectra, support, contact and requested/applied torque. Fresh D/style/critic outputs must be labeled untrained, and the estimator output an uncalibrated latent, not a measured velocity or gait score.

No maintained-code change is needed to load or execute this actor. Root's optional probe subset is an allocation change with its own source identity; it does not alter case duration, scorer threshold or checkpoint bytes.

## If initialization or stopping fails

A failed native start does not uniquely identify missing onset; distribution shift, feedback error and model fitting may also contribute. Compare recorded-reset/settle/onset outputs, clamp counts and the actual native traces before selecting the next intervention. Current off-distribution onset MSE is already much worse than steady fit, but is still only teacher-forced inference.

A data extension must get a new dataset identity. Keep the original accepted steady rows and immutable raw evidence. Add only complete, predeclared windows whose real environment/control/phase/physics-counter mapping can be reconstructed; verify recorded command, five-frame history, previous applied target, desired normalized action, actual target, finite values, model identity, no reset-crossing pairs, no termination and unchanged replay screens. Record excluded rows and reasons; do not append raw cycle 1 wholesale. Existing independent screening yields 15/21 command coverage for onset after four seconds of settling, not every command and not immediate reset-start coverage. Selecting these post-screened rows is explicit data selection, not independent validation.

Keep a row-provenance table and command/window counts. Extra settle rows are all zero command and can dominate uniform BC sampling; declare their selected time windows and resulting zero proportion rather than silently duplicating/downsampling rows. Keep cycle 3 out of fit; its weak same-trajectory scope remains unchanged.

There is no recorded walking-to-stop trajectory. Do not splice a walking observation into a zero-action label or describe stationary zero rows as stopping data. If chosen after native failure, root must record actual stopping under unchanged physics and the original command/quiet timing; failed attempts remain evidence. Neither a new dataset nor BC can substitute for those native gates.

Existing learner seams already separate construction with the accepted AMP prior from `pretrain_bc(other_bc_dataset)`. A future extended BC dataset must be bound separately from AMP through its own manifest/fit receipt/checkpoint SHA; the checkpoint's existing prior_sha256 identifies AMP data and must never be repurposed. Any production entry accepting an additional BC path must validate that binding, row provenance and hash. This is a proposed validation boundary, not implemented code.

## Before any PPO after useful BC behavior

The first native probe does not approve an automatic 200-update handoff. Preserve the demonstrated BC checkpoint as the reference. Current Config/strict load can retain its Adam moments, RNG and normalizers; this is the provenance default and is not proof of stability. The first collect also updates normalizers under existing rules, and policy sampling is stochastic even though the reference probe uses the mean.

Two measured risks need separate attention: the recent lower-noise PPO run had approximate KL 0.245647 and PPO clip fraction 0.665365 without target-KL control; BC's nominal velocity estimator has RMS error 0.774 m/s and subsequent coefficient-one velocity supervision would reshape the latent the actor uses. Neither proves the next update will destroy the gait.

If root chooses a minimal unchanged-learning check, bound it to one update first, save a checkpoint, inspect actual KL/clip fraction, actor parameter changes, estimator loss, normalizer changes and a matching native subset. At most five updates, with checkpoints per update and the same native comparison, is the outer diagnostic suggested by Fable; do not precommit to a long allocation. Retain the BC Adam moments explicitly unless root adopts a separately identified optimizer change. No PPO was run by this audit.

Any lower LR, target-KL guard, estimator freeze/coefficient schedule, critic warmup or velocity-supervised BC is a new learning intervention. It needs an explicit decision, matching source/config and honest checkpoint initialization/migration; never change optimizer parameters after strict load while leaving stale configuration, relabel the estimator silently, or bypass source/config equality. A calibrated BC successor may be preferable if the current actor works but its supervised-velocity handoff proves disruptive; that remains conditional.

Only one- and 32-robot admissions exist. Larger-batch admission is separate, and this Config32 checkpoint cannot silently become Config128 under the strict loader.

## Actual Fable consultation and independent caveats

Fable 5.1 with max effort completed a real, tools-disabled same-session review, `82e1442c-ccd4-4c27-94bd-c234a1b41893`, with actual fit, large-KL and measured latent/velocity facts. It supports the unchanged native BC diagnostic and a distinct bounded PPO handoff review. The subsequent first-three-probe allocation was root's later steering and was not submitted to Fable.

Keep its raw response intact but do not adopt its stronger causal wording. Low one-step MSE/increment agreement weakens copying as the sole explanation; it does not prove autonomous phase recovery. The 120 quiet examples contain measured steady histories, not five exactly repeated reset frames. Native quiet failure can reflect reset/feedback distribution shift even when those supplied zero rows fit. Missing stop examples do not make a stop failure known in advance. Counterfactual command substitution would be a synthetic policy query, not a genuine recorded first control; this audit used unchanged recorded inputs only.
