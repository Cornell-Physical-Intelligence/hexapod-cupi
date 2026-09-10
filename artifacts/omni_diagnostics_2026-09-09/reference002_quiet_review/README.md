# Reference002 standing: independent quiet review

All32 standing replicas pass the existing quiet criteria over the16-second settled window (steps200–999). This scores the original complete trace without another simulation or changed gates. It qualifies only this undisturbed reference hold: the same campaign still rejected the first step, has no walking-to-stop result, and does not complete Stage2 or admit a learned policy.

Across all replicas, worst planar excursion was0.107mm, heading excursion0.0337°, joint velocity RMS0.01070rad/s, joint position range0.003460rad, and target-step p95zero. There were no terminal events anywhere in the trial, no settled requested saturation, and maximum sampled applied/requested torque1.46298N·m. Torque samples are control-step measurements; this analysis does not certify every physics-substep peak. SDK rawXYZW and explicitly convertedWXYZ remain in the trace.

[All32 results and unchanged bounds](report.json) bind the exact trace and scoring source hashes. The original [reference002 campaign](../reference_physics_002/README.md) remains unchanged. The frozen scorer is preserved byte-for-byte; replay executes its exact QUIET_GATES assignment and quiet_metrics function through AST extraction, avoiding unrelated simulator imports without changing its calculations.

Run `uv run python -B artifacts/omni_diagnostics_2026-09-09/reference002_quiet_review/replay.py` from the repository. It verifies both input hashes and exact equality of every recorded result row. This does not mutate the evidence.
