# Reference005: improved base measurement, quiet regression

This isolated standing-only comparison completed all 32 physical standing checks, but passed only **30/32 unchanged quiet checks**, so the solver setting is **not promoted**. No wave or PPO ran. The sole physical intervention from reference004 was enabling external forces at every TGS position iteration; all geometry, startup targets, gains, motor limits, timestep, 16/4 solver iteration counts and acceptance gates stayed unchanged.

Over the declared 16-second settled window, the worst link displacement/velocity discrepancy fell from 9.706 to 2.935 mm using all 400 Hz samples, and from 9.862 to 2.929 mm using the original 50 Hz method. Twenty-six matched environments improved and six worsened; all candidate errors remained below 5 mm. The [complete paired report](comparison/REPORT.md) retains every environment, both link/COM frames and every declared quadrature.

Environments 6 and 29 failed quietness at `revolute_2_1`: reported velocity RMS 0.03507/0.03436 rad/s exceeds 0.03. Sampled position ranges were only 0.000562/0.000927 rad, suggesting a reported-velocity bias; sampled pose differences cannot exclude all faster motion and do not replace the failed metric. Root independently reproduced the same 30/32 verdict. All other quiet checks passed, with no resets, unwanted contacts or settled requested saturation.

The settled 400 Hz requested torque peak was 1.48507 N·m. Initial retained telemetry read 63.7035 N·m before observed physics; actual early settling reached 3.96182 N·m requested at 40 ms and was clipped to approximately 1.6 N·m applied. Original settling exclusions remain unchanged. This is not an all-time requested-torque or hardware-startup qualification.

Evidence:

- [Standing state and gate](results/run/standing/state.json), [actual scene-setting readback](results/run/standing/solver_comparison.json), and full control/substep traces beside them.
- [All-environment comparison](comparison/matched_comparison.json), [quiet failure details](comparison/quiet_failures.json), and [independent root quiet replay](ROOT_QUIET_REPLAY.json).
- [Compact exact source reconstruction](preparation/README.md), [remote source/asset/ownership audit](results/remote_audit.json), and [pause031 restoration](results/forecast_pause/restored.json).

Root independently matched all 20 raw remote payloads, 926 source files and 550 admitted assets, verified the exact owned container absent, and checked restoration. Source manifest: `c557b71636c8b1e10883ab8a6b2b49f40b661988f9b09faeaefe4821a72844b4`. Every original source, failed attempt, comparison and preparation remains immutable. A later solver-iteration diagnostic is separate; no walking/default adoption or Stage 2 completion follows from this result.
