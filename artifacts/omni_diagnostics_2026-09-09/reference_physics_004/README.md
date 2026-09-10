# Reference004: adequate fourth-foot lift, landing bound still rejected

Fresh standing passed all 32 replicas. The full-C zero-residual wave again completed three confirmed steps, then rejected RF landing after 625 controls because the original endpoint error exceeded the unchanged 12 mm correction bound. No full six-leg cycle, completed progress/quiet-stop check or PPO admission is claimed. Stage 2 remains incomplete.

The new 7 mm swing reference produced **3.442 mm measured RF lift**, versus 1.445 mm in reference003. RF had 45 consecutive flight samples and returned after the apex during descent. Its original endpoint/preload error was **12.2345 mm**, while the proposed landing correction itself was 10.4936 mm. These are different checks: satisfying the latter cannot waive the former. Actual forward displacement over the partial moving interval was 37.311 mm; at least five other-foot supports, no unwanted contacts, no resets and no settled requested saturation were recorded.

The standing control trace is byte-identical to references002/003, whose separate frozen quiet replay passed all 32 undisturbed holds. This does not qualify learned standing or stop-from-motion.

## Every physics substep was measured

The observer captured 8,001 initial-plus-substep samples for standing and 5,001 for the rejected wave. Every eighth sample exactly equals the existing control record, with correct counters/timestamps and restored hooks. It introduced no extra physics steps or state writes. The standing trace identity is direct evidence that its control-boundary trajectory was unchanged.

The velocity/displacement discrepancy persists at 400 Hz: worst standing environment 3 differs by 9.706 mm over 16 seconds, and the wave by 9.949 mm over its 8.5-second moving prefix. The original 50 Hz discrepancies are 9.862 mm and 10.553 mm. Neither COM/link conversion nor shifting samples by one or two substeps explains this. The existing 5 mm complete-run consistency bound remains unchanged and was not reached as a final gait check. [Independent replay and sourced solver analysis](substep_review/REPORT.md) preserve all environments and methods.

Settled requested torque at 400 Hz peaked at 1.46309 N·m standing and 1.14622 N·m during the wave, with no above-rating substeps. Startup is different: initial retained actuator buffers read 63.7035/63.6596 N·m before the first observed update, and actual early settling reached 3.98354/3.02281 N·m requested at 40 ms, with applied torque clipped at approximately 1.6 N·m. These remain in the raw data and detailed review; the declared four-second settling exclusion is unchanged. This is not hardware-startup qualification.

## Reproducible evidence

- [Standing admission](results/run/standing/admission.json), [rejected wave state](results/run/wave/state.json), [raw control trace](results/run/wave/trace.npz), [reference states](results/run/wave/reference_states.json).
- Raw substeps and per-control integrals are stored beside each phase's trace; [wave substep summary](results/run/wave/physics_substep_review.json) records their scope.
- [Exact landing diagnosis](landing_review/report.json), [independent observer review](independent_observer_review/REVIEW.md), and [compact source reconstruction](preparation/README.md).
- [Remote source/asset/ownership audit](results/remote_audit.json), complete logs, and [pause030 restoration](results/forecast_pause/restored.json).

Root independently matched all 30 remote payload hashes, all 925 source files and all 550 admitted assets, and verified both exact owned container IDs/names absent. Source manifest SHA: `a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63`. The original preparation and reviews are preserved unchanged inside this terminal wrapper. The primary manifest covers every publication payload. No claim about GPU occupancy after later jobs is implied.

The next comparisons remain distinct: an isolated standing solver-option test investigates measurement consistency, while a timing-only swing successor completes horizontal motion earlier without changing the endpoint, lift, command speed or physical gates. Neither is adopted or admitted by this failed run.
