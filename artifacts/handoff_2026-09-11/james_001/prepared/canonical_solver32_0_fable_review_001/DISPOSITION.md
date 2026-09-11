# Independent disposition of the actual MAX response

The tools-disabled `claude-fable-5-1 --effort max` consultation completed successfully with `stop_reason=end_turn`, session `1e99e71c-ea97-4506-ade3-5f6d4aaca4e0`. The exact prompt, final JSON, launch/exit metadata and stderr are retained. This is a partner critique, not evidence of native behavior. The final standing005 source is independently frozen at `c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131` (109 payloads); the runtime did not change to accommodate speculative suggestions.

Accepted:

- A single controlled 32/0 comparison, followed by 32 only after an authentic same-source single pass, is a defensible next experiment. It preserves the original gates and tests one physical variable.
- Report signed SDK means beside RMS and actual angle-rate/position measures. Our separate hash-verified comparison of both actual 8,000-step single runs shows LM/RM post-settle 400 Hz mean SDK rates changing from −0.02295/−0.02302 to −0.03873/−0.03925 rad/s. Their centered SDK RMS stays about 0.001–0.00125 rad/s. All 8,000 angle-difference recurrences in each run match the recorded interval channel exactly. This supports a large persistent component; it does not identify its physical source.
- Scene clamps matter. The exact actual native5 resolved stage records TGS, scene minimum velocity iterations 0, maximum 255 and external-forces-every-iteration true. That rules out an authored positive scene minimum in that predecessor. It is not independent backend introspection; candidate native readback will remain evidence, not a guaranteed internal iteration count.
- The legacy coefficient getter can fail and must never become an inferred zero. Nonzero values remain unchanged. Link velocities and contact matrix may share backend state, so neither is independent ground truth.

Rejected or narrowed:

- “The quiet gate measures velocity-pass residual, not motion,” “phantom,” and “a 32/0 pass restores the gate's validity” overstate causality. The pose/SDK discrepancy is observed and documented as possible in constrained TGS; a passing numerical variant does not establish hardware fidelity or prove all reported velocity is fictitious.
- Identical trajectories would not prove authoring failed; a variable can have no observable effect in a given trajectory. Different trajectories would not identify the mechanism. Source/attribute proof and measured behavior remain separate.
- Add a hard sleep-signature rejection: this invents a new gate. The previous 32-run events did not show all-zero articulation velocity/contact; legitimate quiet sleep is not by itself a physical rejection. Preserve raw data and all existing gates.
- No predicted mean-angle shift proves the SDK bias never entered PD: false. The exact servo recurrence already verifies that the recorded native derivative entered the requested torque. Contact constraints, friction and geometry can absorb a torque difference without the proposed free-joint angle shift. The rough `kd·dq/kp` estimate is not a constrained-equilibrium proof.
- The matrix “gives tangential over normal per toe for free”: unverified. It is the native filtered aggregate in body sensor order, not the cap-classified patch set; this API alone does not independently identify tangential friction. No diagnostic claims such a calibration.
- A zero resultant plus inactive records proves a contact-generation failure: unsupported. The same event also contains actual angle motion; empty force records do not by themselves distinguish separation, reporting, constraint state or the underlying solver path.
- If 32/0 retains the mean, no iteration count could remove it: an unsupported universal conclusion from one additional setting. Likewise, increased/flipped means do not establish that 32/1 is optimal.
- Mandatory three-point legacy getter sampling or additional support-margin recording was not adopted. This source uses no friction setters, preserves the exact source and all detailed 400 Hz patches, and records the installed coefficient once before controlled physics. Raw support margins can be derived without changing the controller.

No additional native or controller edit followed this review. The next actual result may support or reject the candidate; no PPO admission currently follows from this receipt.
