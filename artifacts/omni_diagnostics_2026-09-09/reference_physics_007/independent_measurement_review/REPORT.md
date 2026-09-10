# Reference007: standing and unchanged quiet gates pass; velocity bias remains

All32 fresh standing replicas pass the original physical gates and all32 pass the unchanged post-settle quiet scorer. This is bounded undisturbed full-C holding, not walking, terrain, PPO or production solver qualification. Source007 corrects the failed006 link-metadata readback; its physical settings are the same16/1 TGS configuration with external-force timing enabled. The exact32 authored roots and608 owned bodies passed the readback.

The independent replay checks every environment, both root-link/COM frames and all four original velocity integrations over control boundaries200–1000(16s). Each400Hz eighth sample exactly matches the old50Hz control sample. Worst root-link displacement/integral discrepancy is2.373838mm at400Hz trapezoid and2.365839mm with50Hz last-substep right integration, versus005's2.935324/2.929059mm. Every replica remains under5mm. Root-link discrepancy improves25/32 replicas and increases7; COM improves26/32 and increases6. Full rows retain the unfavorable cases; no integrator was selected to replace the original wave gate.

Quiet admission remains the original measured50Hz scorer: worst joint RMS0.025609206rad/s, position range0.004444944rad, planar excursion0.236256mm, heading excursion0.00751321degrees, target-step p95zero, saturationzero.005 passes30/32 under this exact same replay;007 passes32/32. The byte-preserved scorer hash matches source007.

**Reported joint-velocity bias is unresolved despite passing the stated0.03rad/s quiet limit.** All32×18 joints are compared individually. The largest007 discrepancy is env6 `revolute_2_1`: measured16s position delta−0.000697136rad, reported400Hz velocity integral+0.409699616rad, discrepancy−0.410396752rad. Reported50Hz RMS is0.025609203rad/s while the position range is0.000716448rad.005's same joint discrepancy was−0.561684848rad. Control-position finite differences are diagnostic only; they neither replace the raw velocity gate nor rule out unobserved faster motion. Root-link agreement below5mm does not prove accurate joint velocities or scalability to larger replicated scenes.

Post-settle requested torque peaks at1.48390329Nm across all400Hz samples, with zero samples above1.6Nm. The initial pre-observed-step actuator telemetry still contains63.7035446Nm requested and1.60000002Nm applied, and actual early-settling updates reach3.96188927Nm requested at0.04s. All are preserved. The initial getter is retained actuator telemetry, not proof of a newly applied63Nm impulse; the early-settling overload is real sampled demand. This is not hardware startup qualification.

The fresh008 proposal combines these solver settings with separately reviewed earlier horizontal swing completion.007 proves no dynamic touchdown, progress, final walk-stop or sensor behavior. A future combined result cannot isolate solver versus timing causality. No old/frozen data, threshold, metric or production default is changed by this review.

Reproduce without simulator dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B compare.py --baseline-run /path/to/reference005/run --candidate-run /path/to/reference007/run --output /fresh/comparison
PYTHONDONTWRITEBYTECODE=1 python3 -B joint_bias.py --baseline-run /path/to/reference005/run --candidate-run /path/to/reference007/run --output /fresh/joint_bias.json
```

The20 raw candidate files were checked against root's terminal audit; it independently verified926 source and550 asset files unchanged, exact owned container absence and pause033 restoration at the recorded time. That audit is historical evidence, not a claim about current GPU availability. This review contains source-bound input hashes,64 quiet rows, all32 frame/integrator comparisons,1,152 joint rows and torque intervals; raw results/restoration belong in the terminal wrapper.
