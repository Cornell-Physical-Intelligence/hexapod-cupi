# Detailed robot: native coordinate and torque response

The canonical 7.466088235 kg direct-drive robot completed all **36 joint-coordinate checks and 36 signed torque-response cases** in Isaac. All ten native checks, 2,064 explicit 2.5 ms steps, host post-exit validation, immutable inputs, exact owned-container absence and weather restoration pass. Root independently audited and fetched **all 29 raw files / 14,842,939 bytes**. This is a successful small-motion diagnostic, not supported standing, controller, hardware or Stage 2 admission.

Each named motor received ±0.005 N·m for 20 ms, followed by 20 ms of zero input, after a bounded zero-input baseline. The full floating rigid-body mass matrix predicts the coupled response. Actual actuated-joint displacement is 0.960416–0.961332 times that prediction: within about 4%. Maximum joint excursion is 0.00862122 rad. Maximum SDK rate is 0.287020534 rad/s; the separate interval-angle maximum is 0.287020206 rad/s. These are aggregate maxima, not proof of equality of every sample or feedback fidelity under ground contact. The declared limits and response discriminator were unchanged.

Native generalized inertia matched the explicitly reconstructed root-COM velocity matrix; the alternative root-origin form is retained in the raw result. Native external-force input readbacks match every commanded pulse. Implicit drive gains and maximum remain zero: external torque is separately applied and software bounded. Projected reactions and incoming child-frame wrenches are recorded separately from commands. The versioned experiment explicitly enables external forces every solver iteration; it does not rewrite the passive inspection configuration or establish a cause for historical gait jitter.

There is no floor, gravity, standing servo, torque-speed actuator envelope, policy, checkpoint or automatic continuation in this run. State resets occur at recorded case boundaries; no hidden reset is used inside a pulse/coast interval. Seventy-four resets and all raw clocks are replayed by the source-bound result validator. Source screw/tibia overlaps, contact accuracy, support, loaded dynamics, hardware calibration and the unchanged Stage 2 gates remain open.

## Reproduction and provenance

- `source/` preserves actual source002, including its original 38-payload freeze, the source001 failure-evidence-only delta and corrected pulse-design identity. Source001 was never launched.
- `host/`, `guard/` and `auditor/` retain their complete original freezes. The unchanged ownership supervisor is independently hash-bound and lives in its separate prior source bundle.
- `root_checks/` retains actual Spark Python 3.12.3 `-B -S` setup, identical ownership CodeTypes, exact fresh transfers and dispatch. Root independently passed 9 source, 12 host, 21 guard and 8 auditor tests. Producer test logs are preserved.
- `terminal/` includes the independent terminal audit, complete raw inventory, original campaign, logs, native data and pause/restore evidence. Invocation `c6d287881452428d90e8e432ae739c86`; both recorded name and ID are absent. Pause004 restored at Unix 1789074862.1635034.
- The exact nine-file canonical asset and completed inspection003 admission are separate immutable dependencies, not represented as new actuation output. Their complete identities were checked before and after acquisition and again by root's auditor.

Run `python3 -B -S verify_bundle.py` from this directory for payload and raw-result replay. Re-executing physics additionally requires the exact asset, original admission, source-bound host environment and a newly authorized unique output allocation. The verifier never launches a simulator.

The next step is source-shape/contact verification and a separately admitted provisional supported standing controller, followed by a fresh two-update PPO integration on this detailed model. No historical C-study checkpoint or task is substituted. Current execution belongs in `STATUS.md`; this evidence stays unchanged.
