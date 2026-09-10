# Independent paired-contact CPU review

No concrete CPU-controller blocker was found in the reviewed version. All **14 owner tests** and **3 additional independent tests** pass. This is an independent code and synthetic-interface review, not clearance to launch or a physical walking, torque, sensor, or Stage 2 admission.

The controller independently tracks each foot's contiguous force-free run, measured 2 mm lift, apex/descent, landing blend, contact reacquisition and three stable post-blend samples. A confirmed foot must remain in contact while its partner finishes. Both feet must complete before the pair advances; the retained four feet must remain measured supports with at least 50 mm projected COM margin. These are separate proposed paired-support criteria, not a waiver of the original single-leg five-support gate.

The added adversarial tests show that qualified flight followed by premature obstacle contact rejects, a six-sample contact loss exceeds the retained 100 ms landing bound, and interruption during an asymmetric landing reproduces reference P/V/A, zero-residual target state and all history frames exactly after JSON restoration. Missing-partner flight/landing, retained-support loss, nonfoot contact, terminal state and stop during flight are covered by the independently rerun owner tests. A latched failure emits no target.

Named runtime permutation, exact executed-target reset continuity, measured versus virtual preload separation and the geometry's 95% soft joint limits are preserved. The reference keeps the 0.02 rad joint reserve and 1.75 rad/s, 6 rad/s² discrete reference bounds; the unchanged residual core passes zero residual without changing the target. These are software target constraints, not measured motor speed limits. Measured body pose uses explicit XYZW plus a checked rotation matrix; the fixture alone follows virtual body motion by construction.

The ideal 0.01 m/s fixture completes 2,200 controls, 10 pairs and 20 landings, with maximum target speed 0.759201 rad/s and acceleration 1.786553 rad/s². Reference quiet begins 5.90 s after the requested stop. The 0.015 and 0.02 m/s fixture cases still reject on retained-support loss and have no final quiet result. Those outcomes are preserved; their illustrative equal force shares are not predictions of physical load distribution.

The new **1,014-feature controller state fragment** includes independent foot and landing states. It is explicitly incompatible with the existing 846/849 actor/critic interface and is not an adopted actor input. Exact checkpoint source/configuration bindings and explicit history epochs prevent silent schema reuse or stale episode history. This remains one-replica CPU code; there is no new batched device or PPO adapter.

The caller must supply authoritative, timestamp-aligned distal-contact and valid contact-point classifications. This generator does not itself infer the 1 N force threshold, sensor freshness or torque from the provided force arrays. A future physical adapter must enforce those independently, including full 400 Hz requested/applied torque evidence and real contact/pose motion. The review does not create a physically admitted stop or recovery trajectory.

[REVIEW.json](REVIEW.json) binds the exact reviewed owner payloads, runtime/oracle hashes and test results. The owner’s final documentation/manifest was still pending when this receipt froze; later runtime changes require separate review. [test_independent.py](test_independent.py) is read-only and accepts the owner path through `PAIR_MOTION_OWNER`; it does not edit the owner or launch physics:

```sh
PAIR_MOTION_OWNER=/path/to/frozen/reference_pair_motion_001 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B -m unittest discover -s /path/to/this/review -p test_independent.py
```

NumPy, SciPy and CPU PyTorch are required. The original frozen scalar oracle and earlier failed prototype evidence remain unchanged.
