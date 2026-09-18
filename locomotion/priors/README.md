# Optional motion-prior preparation

You can optimize the retained forward tripod cycle and replay its motor targets
through the shared locomotion kernel. Directional datasets and AMP remain
unimplemented. This package supplies no paper-reproduction claim.

`model.py` owns inverse dynamics. `optimize.py` solves and audits the cycle.
`prepare.py` packages native replay; `replay_native.py` executes it with the
kernel's contact and torque measurements. The completed action-initialization
comparison remains in Git with its original checkpoint identities.

```sh
uv run python -m unittest discover -s locomotion/priors/tests
uv run python -m locomotion.priors.optimize --help
uv run python -m locomotion.priors.prepare --help
```

Use a fresh output directory. Native replay needs matching standing admission.
Read [TRAINING](../../docs/TRAINING.md) before changing this experiment.
