# Campaign 007 launch record

These immutable launch records document the fresh fd34f66 campaign started
2026-09-05 at22:09:44UTC after independent review of the matched32-world
standing comparison. They are not a final campaign report or training admission.
At22:11:24UTC, the fresh startup probe had passed and full nominal validation
was running. No physical-model PPO had started at that observation.

The bootstrap verifies the completed standing reports, their common source,
exact placements/runtime and unchanged torque/height bounds before invoking
the ordinary full campaign. This does not reuse short standing evidence as
training admission: the new source runs its own complete fresh gates.

Campaign state lives on Spark at:
`/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1/campaigns/fourbar-campaign-20260905T220944Z-78b3e50d/campaign.json`.
Campaign PID1667897; reservation PID1667860, maximum expiry
2026-09-06T08:09:43.812781Z, with release on this campaign's terminal state.
The launch record binds both short report hashes, the final pair record and
this exact bootstrap script. The script is preserved for review; its paths
and prior-guard identity refer to this one operation and must not be blindly
replayed. The remote source remains frozen and read-only to each container.

The sequence is fresh1×100 startup, full32×1000standing+2400driven nominal
and refined qualification,64×3 scratch PPO with checkpoint verification,
then separate512×1000 resumed PPO only if all gates pass. This launch does
not predict policy quality or qualify terrain/hardware. Inspect the live
campaign and per-phase reports for subsequent outcomes.
