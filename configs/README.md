# Named experiment and intervention configs

A training run is a Python configclass tree with a list of Hydra-style
`env.*=` / `agent.*=` overrides applied on top of it. Until now the effective
Stage2C baseline existed only as bash arrays inside a deploy launcher, which
made it invisible to review and impossible to diff. These files promote it to
a named, reviewable artifact.

## Composition model

Three layers, each owning exactly one thing:

1. **Configclasses** (`packages/hexapod_env/hexapod_env/env_cfg.py`,
   `phase2_cfg.py`; the `isaaclab/hexapod_rl` paths are compatibility shims) own
   the *schema*, the *defaults*, and the *validation*. A key that does not
   exist there is not a key. Nothing in this directory may introduce a new
   field, change a default, or relax a check.
2. **`experiment/`** owns the *named baseline*: the exact delta that defines a
   line of work, promoted verbatim from the launcher's bash arrays. One file
   per experiment. `stage2c_accel.yaml` is the Stage2C probe baseline, pinned
   together with its run identity (task id, parent run, parent checkpoint and
   its SHA-256, env count, iteration count).
3. **`intervention/`** owns *one causal probe's delta*: the smallest set of
   keys that distinguishes the probe arm from its baseline, in its own file so
   it shows up as a few reviewable lines in a pull request rather than as an
   argv string in a shell history. `probe21_bilateral.yaml` is two keys.

The effective config for a probe is therefore
`configclass defaults <- experiment/<name>.yaml <- intervention/<probe>.yaml`,
plus a seed.

## Enforcement

`isaaclab/tests/test_experiment_config_contract.py` pins all three
representations to each other:

- exact bidirectional key/value equality between `experiment/stage2c_accel.yaml`
  and the launcher's `baseline_env_overrides` / `fixed_agent_overrides` arrays
  and its pinned run variables;
- every `env:` key in this directory must exist as a configclass attribute
  (a typo guard standing in for Hydra's struct mode, which the launcher's
  argv path does not give us);
- the baseline is checked against the Probe20 zero-intervention resolved
  config recorded under `artifacts/`, which was launched from this exact
  baseline and is therefore ground truth for what the trainer actually saw.

The launcher and the configclasses are the reality; these files adapt to them,
never the other way around.

## Front end: `hexctl`

`ops/hexctl` is the front end for these files. `hexctl compose` turns
`experiment + intervention + seed` into the hardened launcher's exact argv, and
`probe` and `screen` supervise that launcher as a child process; `doctor` gates
the host first. See [`ops/README.md`](../ops/README.md).

The launcher still carries the baseline in its own bash arrays, and
`isaaclab/tests/test_experiment_config_contract.py` pins the two to each other
so neither can drift; retiring those arrays in favour of these files is separate
work. `doctor`, `probe`, and `screen` have not yet been validated in vivo on the
Spark — treat the first real run as a supervised experiment.

## Why not Hydra multirun

Hydra's `--multirun` would sweep interventions automatically, and it stays off
for causal probes on purpose. Each probe attempt gets an immutable one-attempt
label, its own artifact leaf, and a fail-closed record of whether it produced a
valid run. That labeling is scientifically load-bearing: it is what lets an
infrastructure failure be recorded as an invalid attempt instead of silently
retried into the same result slot, and what keeps a probe's evidence
attributable to one exact configuration and seed. A sweep that generates and
reuses run directories on its own would erase that guarantee.
