# Measured LF lift: explicit baselines

Both reported lift values refer to the same measured link reference point in the unchanged `results/run/wave/trace.npz`. They use different, explicitly named baselines:

| Definition | Trace control index / time | Lift to peak |
| --- | --- | ---: |
| Settled startup / before the motion command | 199 / 4.00 s | 2.719058 mm |
| Immediately before the observed continuous flight | 222 / 4.46 s | 2.690834 mm |
| Measured peak | 250 / 5.02 s | — |

Observed distal contact is absent over indices223–272 inclusive (50 samples); it returns at273 / 5.48 s. The original gate's 2.690834 mm event metric uses the sample immediately before observed flight. Root's independent 2.719058 mm uses the settled startup baseline. Neither figure is an estimate of whole-pad or shaft clearance, and neither establishes a completed, confirmed landing.

The values are independently reproducible by taking the peak of `reference_point_world_m[223:273, 0, 0, 2]` and subtracting the same array's value at index199 or222. Leg index0 is explicitly LF in this trace; `legs` and runtime joint names remain recorded. No raw result or original preparation file was edited for this note.
