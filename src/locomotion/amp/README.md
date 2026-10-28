# Motion priors and AMP

You generate periodic motor targets, audit their native replay and compare
PPO with an adversarial motion prior. The manifest of the admitted bank in
`src/locomotion/priors/datasets/amp_demonstrations_001` binds 18,000 transitions to the
approved model and reviewed native clips. The admission records and the learner
identity cite that bank by its pack path, `locomotion/priors/datasets/amp_demonstrations_001`.

| Module | Purpose |
| --- | --- |
| `model.py`, `optimize.py` | Full-body inverse dynamics and constrained periodic trajectory optimization through CasADi/Ipopt. |
| `prepare.py`, `replay_native.py` | Pack and replay solved targets through the native `OmniEnv`. |
| `dataset.py` | Audit the command bank, require motion review, export and verify raw transitions. |
| `features.py` | Pack 61 motion-feature values from measured state. |
| `amp.py`, `discriminator.py` | Extend stock RSL-RL PPO with the style reward, discriminator and checkpoint state. |
| `networks.py` | Keep the history encoder, velocity estimator and privileged critic comparison. |

## Generate and admit demonstrations

```sh
uv run python -m locomotion.amp.optimize --help
uv run python -m locomotion.amp.prepare --help
uv run python -m locomotion.amp.dataset --help
uv run python -m unittest discover -s src/locomotion/amp/tests
```

Give each attempt a fresh output directory. The optimizer retains failed
attempts, trajectory hashes and model identity. Its saved fixture preserves the
recorded force, torque and 0.040 rad target-step checks. A finite failed iterate
can seed a fresh attempt through `--initial-trajectory`; it grants no admission.

Replay each solved command at start phases 0 and 0.5. Follow
[OPERATIONS](../../../docs/OPERATIONS.md) for standing admission and GPU ownership.
`dataset inspect` checks telemetry, hashes, tracking and feature reconstruction.
The program lead reviews direction and tripod motion and supplies the bound
review to `dataset export`. The exporter crops controls 100:1000 from phase zero;
half-phase clips stay as audit evidence. You cannot join clip endpoints
or export reset/terminal transitions.

## Feature and learner contracts

Each raw state gives 61 values: 18 joint positions, 18 joint velocities,
body-frame root-origin linear velocity, angular velocity, root height and six
body-frame toe positions. Adjacent states span 0.02 s. The discriminator reads
122-value pairs and standardizes them using the admitted bank. It learns
least-squares prior/policy targets and a gradient penalty; its style reward lies
in [0,1]. The online penalty differentiates standardized prior inputs.

You select `--learner amp` in the native scripts or `spark/prepare.py`. The MLP
comparison supports the gait clock and wrapper settings. `--networks
paper` selects the retained encoder comparison and requires raw inputs without
a gait clock or velocity noise. AMP preserves terminal next states before reset,
uses stock PPO updates and saves discriminator, optimizer and bank identities.
The learning-rate cap and command-boundary bootstrap apply to AMP too.

RSL-RL 5.0.1 provides PPO, storage and model interfaces. This package adds the
AMP discriminator and style reward. Native runs
remain the acceptance evidence. CPU regression and surrogate runs cannot admit
a policy or establish simulator parity.
