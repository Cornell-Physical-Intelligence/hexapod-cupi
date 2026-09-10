# Exact reference004 → reference005 standing-only comparison

This is the immutable source preparation for the external-force-timing comparison. Terminal results and independent review belong in a separate wrapper. It contains four reviewed runtime overlays, source lineage/map, owner tests and an exact no-write reconstruction recipe, without duplicating the 925-file parent.

Parent source map: `a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63`. Target 926-file source map: `c557b71636c8b1e10883ab8a6b2b49f40b661988f9b09faeaefe4821a72844b4`. The sole physical setting change is TGS external forces applied every internal position iteration, False→True. Startup/targets, iterations16/4, physical asset, torque limits, contact sensors, observer and all original metrics/gates are unchanged. Separate host/runtime restrictions allow only32×1000 standing and reject any wave or PPO continuation.

`integration_review/README.md` and the protocol preserve the exact predeclared all-environment comparison, scene readback requirements, original5mm diagnostic and older-policy negative history. The CPU preparation does not claim a favorable physics outcome or adopt the solver as a default.

Verify the exact virtual source:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B reconstruct_source.py --parent-source /path/to/reference_source004
```

Add `--output /fresh/reference_source005` only when materialization is required. The recipe checks the full parent, no extras, every overlay and the target map, and refuses overwrite. All owner copy hashes and the verified reconstruction result are included. No simulator or job is launched by this recipe.

The independent PPO receipt is a separately frozen sibling (`reference_solver_review_005`, SHA61d10a27dc6d23322a6b5d07ef627b0e1b5c4a1d58bed5dd2a41be26333f662b); root can append it with the terminal raw result, pause031 restoration and matched quiet review. This preparation stays immutable.
