# Actual008: seven touchdowns, then incomplete wave rejected

Fresh32×1000 standing passed physical and all32 unchanged quiet gates. Both its raw50Hz trace and400Hz substep NPZ are byte-identical to007; adding the explicit quiet admission did not change that recorded physical trajectory. The raw source008 standing receipt now includes quiet scoring; that stronger admission was declared before launch.

The wave stopped at1037controls with `Reference rejected before target emission: Returned contact lacks measured2mm flight, passed apex and actual descent`. It completed seven confirmed touchdowns across all six leg identities, then rejected RR's second swing. This is a failed/incomplete bounded reference trial, not a walking policy or Stage2 pass. It did not finish24s forward travel or reach measured quiet stopping. Sensor-owner contact/phase diagnosis is a separate frozen review; this report focuses independent time, displacement and torque evidence.

During the16.74s post-settle moving prefix, actual root-link forward displacement was78.533883mm. The original50Hz trapezoid progress function integrates75.124451mm forward and reports3.522591mm3D disagreement. Independent400Hz trapezoid reports3.504569mm disagreement. Every eighth substep exactly equals the original control sample, counters/timestamps cover all8297samples, and the root-link/COM relation is checked. The prefix is under the old5mm threshold, but an incomplete prefix is not full moving-window or stop admission. No diagnostic substitutes for the original gate.

The actual wave's settled400Hz requested torque peaks at1.44310176Nm at11.4225s on `revolute_4`; the50Hz sampled maximum is approximately1.25420Nm. The substep peak is retained rather than hidden by control-rate sampling. Every settled400Hz sample remains below1.6Nm. Reported support remains at least five with no post-settle nonfoot contact, resets or sampled saturation, but these preserved physical bounds do not override the failed flight/landing requirement.

Startup remains unqualified for hardware: standing initial retained actuator telemetry contains63.7035446Nm requested, and real early-settling updates reach3.96188927Nm; wave initial retained telemetry contains63.6596222Nm and early updates reach2.88378310Nm. Applied torque is clamped near1.6Nm. The initial stored getter value is not proof of a newly applied63Nm impulse; the subsequent sampled overloads are distinct real measurements before the unchanged scoring boundary. All raw intervals remain available.

The observed joint position/rate disagreement from007 is unresolved.008 still records joint angles only at50Hz; its400Hz observer records rates without joint angles. A separately reviewed observer successor will collect actual angles at every existing update without altering physics or any gate. Neither the root-link improvement nor this one bounded waveform establishes native joint-rate accuracy, large-replica scalability, production defaults or PPO readiness.

Compared with actualwave004, both solver settings and horizontal foot timing changed. No improvement in008 is attributed to only one of them. The frozen preparation/source and raw result are unchanged.

Reproduce the diagnostic arithmetic:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B substep_replay.py --run /path/to/reference008/run --output /fresh/substep_review
PYTHONDONTWRITEBYTECODE=1 python3 -B extra_review.py --run /path/to/reference008/run --output /fresh/extra_measurements.json
```

All32 raw result/restoration files matched the root audit. That audit recorded926source/550asset files unchanged, exact owned container absence and pause034 restoration. These are historical terminal checks, not a current GPU-idle claim. Root's quiet replay notes tiny numerical heading differences in three rows versus Spark while preserving every verdict; this report does not claim bit-exact cross-platform quiet arithmetic.
