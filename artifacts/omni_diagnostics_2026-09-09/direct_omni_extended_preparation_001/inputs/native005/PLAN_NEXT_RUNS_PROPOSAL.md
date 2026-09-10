# Proposed documentation updates — root-owned integration only

## STATUS.md

Native005 CPU preparation adds an explicit 500-update direct-position PPO allocation from the preserved original checkpoint. The previous 50-update quiet-priority result failed all 48 quiet trials; its largest stop-window displacement includes a reset and is not continuous drift. No new source smoke, extended training result, or Stage 2 completion is claimed by this preparation.

## docs/PLAN.md

Keep the existing objective, physics, observations/actions, adaptive learning rate, command curriculum and formal evaluation gates. Compare a single selected CAPS branch using a fixed longer budget rather than treating the previous 50-update result as proof of infeasibility. Native005 supports caps and quiet_priority explicitly; root selects one after reviewing actual evidence. Fresh original initialization is required because current checkpoints do not preserve all RNG/simulator/history/schedule state needed for exact continuation.

## docs/NEXT_RUNS.md

After root reviews and source-builds native005 and the corresponding bound host: fresh 32×24×2 same-source quiet-priority smoke with standing, strict reload, final constant and stop acquisition. Only a successfully completed terminal smoke admits one selected 1024×24×500 experiment, with fresh standing → original constant/stop → train → final constant/stop. Relative decisions and gradient milestones: 1/10/25/50/100/250/500. Save every native update unchanged; inventory all remote files while curating local copies. Explicit extended/train bound1800 seconds, other phases600 and AppReady90; no unreviewed continuation or simultaneous branch allocation. Compare physical tracking/stop/quiet/contact/motor behavior, resets, full LR/KL and sparse component gradients; losses or acquisition completion alone cannot promote the model.

## Central framework

Append a versioned evidence record and link the native005 CPU preparation while keeping latest measured checkpoint, latest video and qualified benchmark separate. Update the locomotion next-experiment text without marking omnidirectional Stage 2 passed. Include STATUS.md, PLAN.md, NEXT_RUNS.md and a new site/updates record covering the bounded artifact paths. Run project-site check/build and any applicable runtime-lineage checks; no frozen prior artifacts are rewritten.
