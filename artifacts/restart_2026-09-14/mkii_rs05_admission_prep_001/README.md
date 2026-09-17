# Prepared RS05 direct-task standing capture

Status: **PREPARED_NOT_EXECUTED**. No capture ran, no scorer graded a result,
no compute was allocated and no admission exists for the new task. This
directory records the exact commands a successor runs on the Spark host and the
source and model identities they are bound to.

The Spark host was unreachable from the executing machine on 16 September 2026.
`ssh -o BatchMode=yes spark 'echo ok'` exited 255 because the `spark` alias did
not resolve; the Tailscale client is absent from that machine. Every native step
below therefore stayed unexecuted. `PREPARED.json` records the check, the
commands and the bound hashes; `SHA256.json` records this directory's bytes.

## What is prepared

`packages/hexapod_env/hexapod_env/tasks/mkii_rs05/` holds the direct task on the
model `robot/active_model.json` selects. `isaaclab/admit_mkii_rs05.py` records a
standing capture in the layout the unchanged
`experiments/paper_walk/env.py:score_diagnostic` reads. CPU tests compare the
action path, observation layout, reward terms, terminations and the capture
layout with that frozen source. CPU agreement is not native behavior.

## Exact commands

Run them inside the project's workload-gated launcher, after verifying the
reservation, the GPU lock and process ownership on the host.

```sh
ssh -o BatchMode=yes spark 'echo ok'
# one replica
python3 /workspace/hexapod/isaaclab/admit_mkii_rs05.py \
  --num-envs 1 \
  --geometry /workspace/hexapod/artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json \
  --geometry-extrema /workspace/hexapod/artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry_extrema.npz \
  --output /workspace/hexapod_runs/mkii_rs05_admission_001/one
# the declared replica count of the intended allocation
python3 /workspace/hexapod/isaaclab/admit_mkii_rs05.py \
  --num-envs 128 \
  --geometry /workspace/hexapod/artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json \
  --geometry-extrema /workspace/hexapod/artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry_extrema.npz \
  --output /workspace/hexapod_runs/mkii_rs05_admission_001/batch128
# grade each capture with the unchanged scorer
python3 -c "import json,sys;sys.path.insert(0,'/workspace/hexapod');\
from experiments.paper_walk.env import score_diagnostic;\
print(json.dumps(score_diagnostic('/workspace/hexapod_runs/mkii_rs05_admission_001/one/standing'),indent=2))"
```

Add `--dry-run` to print the plan without any simulator import.

## Limits

The runner writes evidence. It changes no gate and grades nothing. A capture at
one replica and a capture at 128 replicas are separate results with separate
identities; neither transfers the historical paper-walk admissions, which keep
their own source, model and layout. Passing this standing screen would admit
standing for this task alone, not walking, stopping, terrain or hardware.
