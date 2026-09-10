# Physical counterevidence and next bounded experiment

The improving raw quiet loss did not produce an accepted quiet controller. Both quiet-priority50 and normal CAPS50 pass **0/48** final quiet trials. Every replica's target-step p95 remains about **0.04000002 rad per 20 ms**. The recorded raw SDK rate channel also remains far above its unchanged bound; it is not replaced by an angle-difference estimate.

Independent read-only comparison of the raw final-stop JSON and traces gives:

| Result | Normal CAPS50 | Quiet-priority50 |
| --- | ---: | ---: |
| Median quiet planar excursion | 50.42 mm | 57.60 mm |
| Maximum scored excursion, retaining resets | 75.11 mm | 1,870.23 mm |
| Median mean requested saturation in quiet | 10.53% | 11.16% |
| Constant scenarios with saturation worse than original | 2/12 | 9/12 |
| Trial terminations | 2 | 2 |

The maximum needs a crucial qualification. Quiet-priority environment20 terminates at **25.12 s**, inside the scored quiet window, and its next-row position jumps **1.86541 m** after reset. Normal CAPS environment20 terminates at **18.82 s**, before quiet scoring, with a **1.82997 m** reset jump. Their environment44 terminations are also before quiet scoring: 17.52 s for quiet-priority and 20.86 s for normal CAPS. Consequently, 1.870 m versus 75.11 mm does **not** show 25-fold uninterrupted walking drift or a reset-free CAPS branch. Both failed trials and both original scored maxima remain failures. No row or unfavorable window is removed. Medians and saturation provide a less misleading comparison, while still not establishing the quiet coefficient as the cause.

The next useful experiment is a **fixed, fresh-original normal CAPS500 budget baseline**, subject to root's normal source/standing/launch review. Preserve the original checkpoint and initialize the optimizer explicitly as before. Keep geometry, physics, command curriculum, observation/action contract, gates and inference evaluations matched. Retain intermediate checkpoints, including a midpoint such as250, so a cold constant/stop screen can assess whether extra budget helps. A checkpoint is diagnostic evidence, not a promoted controller or permission to continue automatically.

This experiment answers a missing question: does substantially more optimization improve the existing normal-CAPS branch's measured behavior under the same gates? It does not isolate a weight effect from a budget effect against quiet-priority50, prove normal CAPS superior, or establish that a quiet objective cannot work. Neither current branch is qualified. A longer quiet-priority run is a plausible later budget comparison, but its smooth loss decrease alone is insufficient reason to prefer it now. A stronger quiet weight or architecture change adds a new intervention before the available budget response is measured; the sparse pre-Adam gradients do not yet establish that such a change will improve executed target steps.

This is an experimental-priority recommendation, not a GPU dispatch or new acceptance rule. Root owns the bounded allocation and midpoint decision. The Fable partner received the training diagnostics and physical reports; its final response is independently checked against the reset-timing correction above before any claim is adopted.

Fable independently chose the normal CAPS500 budget baseline, with a midpoint diagnostic. Its proposed literal native003 execution is not adopted: that frozen CLI admits only50 pilot updates, so a separately reviewed500-update source/contract is required. Its speculative sensor-floor concern is not a verified launch blocker. The exact stop evaluator and pinned RSL source use deterministic mean outputs. A nonreset quiet-priority row also has actual joint range0.300rad, adjacent-angle RMS1.492rad/s and targetp95.04, so the failure cannot be explained solely by an SDK reporting floor. These separate measurements do not relax the SDK-rate gate or establish a native simulator cause.

`physical_comparison.json` preserves the exact reset times, next-row jumps, same-gate/override equality, per-branch summaries and source-file hashes. The raw final-stop checkpoint identities are `ca0545760f81b8b8fb0cafde4b1ef89bfa5d8765a8b43b578e451f648a954415` for normal CAPS50 and `195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173` for quiet-priority50. The latter training receipt SHA remains `c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11`. This review does not repeat the complete terminal source/cleanup audit or modify any source, gate or GPU state.
