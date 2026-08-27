# hexapod_env

This package owns the Isaac Lab side of the hexapod: the `HexapodEnv` direct RL
environment, the per-stage configclasses (`env_cfg.py`, `phase1_*_cfg.py`,
`phase2*_cfg.py`, `ppo_cfg.py`), the command-sampling and curriculum helpers,
the pure reward/gait primitives under `rewards/`, and the gym registration in
`register.py`. It is consumed today via `PYTHONPATH` (the Spark container adds
the repo's `packages/` directory), and `isaaclab/hexapod_rl/` remains as a set
of re-export shims so the historical `hexapod_rl.*` import path and every gym
entry-point string keep resolving. Don'ts: never rename or renumber a gym task
ID, and never repoint an entry-point string away from `hexapod_rl.<module>`;
never import from `hexapod_train` or `hexapod_eval` (dependencies point inward,
toward this package only); keep every module under `rewards/` free of Isaac Lab
imports so it stays testable with `torch` alone; and any change to a reward,
curriculum, or termination term needs a matching test under `isaaclab/tests/`
before it lands.
