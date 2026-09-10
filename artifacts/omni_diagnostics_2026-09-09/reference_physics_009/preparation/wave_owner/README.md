# Wave005: qualify measured liftoff before interpreting a landing

This is a CPU-tested scalar successor to frozen wave004. It changes the measured-contact state machine only. Geometry, commands, body prediction, 7 mm lift, 2 s swing, 0.80 horizontal duration, joint target equations, preload, P/V/A bounds and every post-qualification landing limit are unchanged. No GPU execution or physical qualification is claimed.

Actual008 completed seven measured landings, including every leg, then rejected RR's second swing at 16% phase. Two zero-force samples produced only 8.407 micrometres of measured lift before contact returned. The target was still unloading its below-ground virtual anchor. The parent treated those two samples as flight and the return as a landing attempt. The immutable diagnosis is `../reference_wave008_review_001`; the raw trace remains a failed physical run.

The successor starts each swing in `unloading`. `flight_seen` becomes true only after both at least two consecutive absent-distal-contact samples and the existing minimum 2 mm measured rise within that same run. The existing planned apex is the finite deadline; failure to qualify by then rejects. No new tuning variable is introduced. An earlier unqualified contact return resets consecutive count, height baseline and peak, so separate blips cannot accumulate flight evidence. That return records an unloading event, never a landing or successful step.

After qualified flight, every previous obstacle/early-return rejection remains active. Apex and actual descent, measured lift, speed, 12 mm original endpoint/correction/excursion limits, bounded contact reacquisition, three stable post-blend samples, support geometry, nonfoot collision, target limits and finite stop behavior are retained. The physical adapter's independent measured-flight, torque, collision, support, progress and quiet-stop gates remain unchanged. This prototype does not alter the independent scorer or simulator sensors.

The fields named `raw_force_free_*` count absence of the provided distal-contact flag, which uses the existing contact-point and force classification. They do not assert that every such sample is physically airborne or exactly zero force. Actual008's two reported forces happen to be zero. Forces and raw contact points stay in the separate measured trace.

## Serialized state changes

The new mode is `unloading`. `flight_seen` now means qualified measured flight, and `flight_count` always reports the current consecutive absent-contact run, resetting on any contact. The qualified latch survives contact until the next swing/reset. Seven fields are added:

- `flight_qualification_semantics`: constant contract identifier.
- `raw_force_free_samples` and `raw_force_free_runs`: per-swing raw counts.
- `unqualified_contact_returns`: per-swing count of returns before qualification.
- `last_unqualified_return_time_s`: optional time of the last such return.
- `last_unqualified_run_samples`: its consecutive sample count.
- `last_unqualified_lift_m`: its measured peak rise, preserved before resetting the baseline.

These fields and semantics intentionally require a separately bound tensor and policy-observation schema. Frozen wave004 tensor/observation contracts are not compatible. This source is a scalar physics-screen candidate, not a policy or checkpoint migration.

## Verification and scope

Twenty-eight CPU tests pass: 20 inherited tests plus eight targeted qualification cases. They cover separated blips, insufficient micrometre clearance, qualified-flight obstacle rejection, finite unloading deadline, a stop during unloading, and four historical prefixes. The synthetic full-forward test still completes a wave and finite quiet stop. All common-prefix joint position, velocity and acceleration targets in the four recorded replays are exactly equal to the immediate parent.

| Recording | Immediate parent versus successor counterfactual replay |
| --- | --- |
| 002 | Both retain the earlier reviewed provisional landing correction; zero confirmed touchdowns at the recorded end. The actual older wave001 run remains rejected. |
| 003 | Both reject the insufficient 1.445 mm clearance. The successor rejects at 12.08 s near the apex, earlier than the parent's 12.34 s returned-contact rejection. |
| 004 | Both retain the original 12 mm endpoint rejection at 12.50 s. |
| 008 | Parent rejects at 20.74 s. Successor records the unqualified rebound, remains in unloading with seven confirmed landings, and produces only a next **unexecuted** target at 20.76 s. |

These are existing measured inputs replayed through two controllers. There is no physical continuation past any recorded terminal result. Historical002 uses its 5 mm/full-duration horizontal profile; 003 uses 7 mm/full-duration horizontal; 004 and008 compare the immediate parent's 7 mm/0.80 profile. No historical physical outcome is relabeled as a pass.

The runtime dependency list remains exactly `wave_reference.py`, `wave_math.py`, `serial_geometry.py`, `geometry/candidate_c_reference.json`, and `geometry/f050_t060.urdf`. Only the first changes from wave004. Source hashes and copied historical inputs are in `SOURCE_ORIGIN.json`; the independent review must bind the final freeze before root assembles a new physical source.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_reference_wave_005 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 tmp/omni_reference_wave_005/replay_histories.py
```

`initial_full_tests_harness_error.log` preserves an initial test-fixture baseline mistake: the fixture sampled after its first synthetic target had already risen 0.435 micrometres. The corrected test uses the recorded launch baseline. Runtime code did not change for that correction. The old 20-test initial output is preserved separately. Benign scalar tensor-construction performance warnings are retained.
