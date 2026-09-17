# Source release verification

Source identity establishes which files were checked. Robot admission and policy
acceptance require matching physical/evaluation evidence under [ARCHITECTURE](../ARCHITECTURE.md).
The checker's historical `mkii_fourbar_v1` label does not qualify the canonical model.

## Verify a checkout

```sh
python3 tools/check_pipeline_lineages.py historical
python3 tools/check_pipeline_lineages.py current
python3 tools/c_study_runtime.py
```

The historical check verifies `stage2_pipeline.sha256` against the immutable Git
revision `81d7c6f2a43c7de99f32cd6bb1b7efb0f54874df`. Fetch repository history first.
The current check selects `repository_ppo_reference_20260917_v10_pipeline.sha256`.
It covers current runtime/source paths, tests and retained release inputs,
including the inventoried top-level modules and CPU tests in
`experiments/paper_walk/` and `experiments/trajectory_optimization/`. Nested experiment
outputs and frozen attempt copies are not scanned as maintained source. CI runs
the prototype's CPU checks separately; its native Isaac allocations retain their
own exact source, model, standing-admission and result identities. The two explicitly
allowed historical packaging changes remain in the checker; frozen package guides
and all other historical contracts retain their original bytes.

## Publish a source change

1. Preserve all published manifests. Give the successor a new filename.
2. Update the checker's current manifest and the CI selection to that filename.
   Include the preceding release manifest in the retained source inputs.
3. After edits and focused checks, generate the new manifest with
   `python3 tools/check_pipeline_lineages.py generate --manifest <new-path>`.
   The command rejects overwriting an existing manifest.
4. Verify historical and current identities, run required suites, and add an
   append-only site update identifying the change and its qualification limits.
5. Commit the coherent release. Use its pinned checkout for reproduction;
   source relocation does not transfer old admission or checkpoint compatibility.

The canonical-restart v6 release retains the earlier canonical-restart
manifests and every preceding release unchanged. It adds explicitly labeled
cold and settled BC startup diagnostics, preserving separate whole/prefix/policy
physical evidence and all original evaluation cases. It also checks exact
evidence-path capitalization across macOS and Linux before paper publication.
Explicit hash-pinned correction records repair historical evidence links while
retaining original record bytes and visible provenance. The v3 rollback and actor
normalizer controls remain unchanged. Cross-schema checkpoints require an explicit recorded migration;
the runtime loader remains strict. Source inclusion does not admit a robot,
accept a policy or transfer historical checkpoint compatibility. The checker's
historical header remains a lineage identifier, not a native run's robot model.
The canonical-restart v7 release retains v6 and every preceding manifest
unchanged. It binds the prototype guard to the pause coordination hash, adds the
maintained `refit_bc.py` helper with its CPU tests, and records the executed
paired BC refit under protocol 002. Source inclusion of a fitted candidate does
not evaluate it natively.
The canonical-restart v8 release retains v7 and every preceding manifest
unchanged. It adds the RS05 direct task on the approved model: its asset spec,
the paper-walk actuator, the task modules, the standing capture runner, the
scratch-only trainer and their CPU tests. Source inclusion registers a new task
ID; it records no capture, no training and no admission.

[Archived release history](archive/README.md) preserves earlier explanations and
failed-attempt references. New release explanations belong in site updates.

The trajectory-optimizer v8 release retains v7 and adds the approved-model
dynamics solver, native replay packer and independent dynamics checks. It pins
the saved feasible cycle used by the CPU regression checks. The source release
does not accept the native replay: the original tracking gate fails.

The locomotion-force v9 release retains v8 and adds normal-contact load and
motor-torque summaries to native evaluations. It preserves the physics source
and acceptance gates. The new report also supports separate reanalysis of
existing captures without changing their bytes.

The PPO/reference v10 release retains the preceding v9 manifest and the PR 22
port manifest. It adds the standard RSL-RL PPO adapter and native load summaries,
and optional speed/curvature objectives for the reference optimizer. It also
repairs per-patch contact classification and replica-coordinate handling in the
RS05 direct task. That task still needs native admission; the PPO comparison
uses the existing admitted prototype physics. Source identity does not qualify
omnidirectional behavior.
