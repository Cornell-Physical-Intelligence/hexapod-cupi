# Local TGS one-iteration fallback candidate

**CPU-tested candidate only. No Spark sync, GPU launch or physical/PPO admission.**
This branch, `codex/mkii-velocity1-candidate`, starts at `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683` and selects
`mkii_fourbar_tgs_external_forces_1600hz_position64_128_velocity1_v7`.
The active v6 campaign remains a separate source release.

The complete runtime behavior diff is exactly two assignments in
`packages/hexapod_core/hexapod_core/fourbar_v1.py`: the unique numerical recipe ID
and velocity iterations **16 → 1**. Nominal/refined position counts remain
64/128, physics remains 1600 Hz with 32 substeps per 20 ms policy step, and external
forces are applied every iteration. The v5 robot, motor model, target schedule,
31 native contact sensors, every-substep metrics and before-reset training guard
are unchanged. Physical gates were not modified.

The matching [NVIDIA 110.1 solver guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/simulation_control/simulation_control.html#physics-solver)
recommends around one TGS velocity iteration. This motivates a controlled
fallback experiment; it does not establish a root cause or promise stability.
The current 1600 Hz / 16-iteration experiment must complete before an owner decides
whether this fallback should run.

## Validation

- Focused physical-fourbar CPU tests: **139 passed**, 64.088 s.
- Complete repository CPU suite: **938 passed**, 91.275 s.
- Runtime AST comparison: only the two declared core assignments differ.
- Validator, trainer, qualifier, asset binding, environment/config, motor JSON
  and launch/campaign entrypoints are byte-identical to the base release.
- `git diff --check` passed. The fresh local virtualenv was built from the
  existing lockfile offline; dependencies and lockfile were not changed.

The strengthened numerical-recipe tests reject old v6 reports with internally
consistent 16-iteration settings, a new nominal mixed with an old refined result,
and Python `True`/`1.0` masquerading as integer 1 in all three evidence locations.
Historical 800 Hz recipes, stale actual backend settings and timing divergence
remain rejected. Test configuration starts with a deliberately wrong velocity
count so the assignment check remains meaningful. Existing negative-CLI tests
print usage/errors in otherwise passing suite logs.

An independent read-only review by `/root/capture_recovery` found no concrete
blocker and confirmed the two-assignment runtime delta, exact scalar typing,
stale-v6 rejection and mixed-recipe admission failure. It did not rerun the suite.

## Freeze and next use

Functional source identity: `328efb07dded31e871d0b6dad3c50d949a7af798bacd8f8e6f3451ccbb7ca340` across 241 files.
`source_contract.json` is the complete admission identity; `source.SHA256SUMS`
checks its files from the worktree root. `candidate.patch` preserves the exact
runtime, test and documentation change. `validation.json` binds the tests and
selected recipes to changed/protected file hashes. The enclosing commit ID must
be recorded by the owner when staging a future release; this artifact does not
contain a self-referential commit claim.

From this directory run `python3 verify.py` and
`shasum -a 256 -c SHA256SUMS`. Source checks are CPU-only and import no Isaac/GPU
runtime. The candidate requires fresh complete 32-env, 1000-standing + 2400-driven
runs at 64/1 and 128/1, followed by the unchanged solver convergence/admission
checks before scratch or full PPO. Each full validation captures 108,800 physical
substeps per environment. No current-v6 admission or checkpoint can be reused.

No artifact from a previous runtime was relabeled as this candidate's result.
The branch contains historical evidence inherited from its base commit, which
retains its original meaning. The updated training guide marks this fallback as
local and unqualified; the root worktree's live `STATUS.md` was not edited.
