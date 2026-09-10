# Independent reference004 substep telemetry review

All six focused tests passed independently. The observer calls the original scene update first and records the exact eight 0.0025-second updates, including link/COM position and velocity, torques, and timing. Endpoint samples must equal the existing pre-reset control sample. The existing 50 Hz progress and 5 mm discrepancy gate remain unchanged. This is additional diagnostic evidence, not a substituted acceptance velocity.

The reviewer identified an initial-entry exception gap: assigning a recorder only through `with ... as` lost access to an initial failing sample. The owner fixed it by constructing the recorder before context entry and retaining the entry error. The new test confirms raw initial nonfinite evidence can be exported while the original scene method remains untouched. Missing/extra updates, wrong dt, endpoint mismatch, decimation handled by the backend, nonfinite samples, aliased final velocity samples and hidden torque spikes are also covered.

No remaining concrete first-run API blocker was found against the recorded installed SDK definitions. Actual Isaac execution must still confirm this hook, and all physical admission gates remain required. No runtime file, GPU job, main branch or Git state was changed by the reviewer.
