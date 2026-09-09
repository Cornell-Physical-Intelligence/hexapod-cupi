# Prepared corrected learner release

Source commit `5d476d42546bf4f5c84ca39b8c224c8f68f1454b` contains the tested RSL-RL configuration and checkpoint-reload fixes. Functional identity is `d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54`; the separately versioned 298-path manifest and archive are recorded in [source_release.json](source_release.json). The full repository suite passes 931 tests. Actual CPU synthetic learner/save/reload/continue checks pass, without claiming robot PPO.

All 298 manifest paths were verified again on Spark at2026-09-06 23:09:32UTC. [staging.json](staging.json) and [refined_dry_run.json](refined_dry_run.json) record the exact prepared command with `execution=not_started`. No GPU job, reservation or shared-control change occurred.

## Next bounded GPU action — prepared, not launched

The shared Spark file records a newer user pause at13:33UTC and prohibits a long-lived exclusive reservation. Its canonical control remains `REQUESTED`. This release does not clear that control, launch a workload or create a reservation. An explicit user resume and coordinated scheduling are required. Use ordinary per-job collision/resource locks only; do not restart `reserve_workflow.py`.

After the source is staged and the user explicitly resumes GPU work, the first useful job is a complete refined physics comparison. It retains the existing robot, 800Hz timing, Kp30/Kd0.30, coincident flat layout and all acceptance bounds, using128/16 solver iterations:

```sh
python3 /home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/source/isaaclab/deploy/run-mkii-fourbar validate \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/source \
  --source-commit 5d476d42546bf4f5c84ca39b8c224c8f68f1454b \
  --asset-model mkii_fourbar_v5 \
  --environment-layout coincident_flat_origin_v1 \
  --num-envs 32 --steps 1000 --solver-multiplier 2 \
  --timeout-seconds 7200 \
  --output-root /home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/refined_diagnosis_001
```

The validator includes2,400 additional driven controls, totaling54,400 physics substeps per robot. The timeout is a maximum, not an ETA. The preceding nominal run took approximately29minutes; refined runtime is unmeasured. The launcher rechecks coordination/resources, captures exact source, and owns only its new container.

A refined pass alone does not admit PPO. The previous nominal failed its closure bound, and the new source has a different identity. Use the refined result to choose the next numerical/controller correction; establish complete matching nominal/refined reports and convergence on that exact selected source. Only then proceed to64×3 scratch PPO, separate512×1,000 resumed PPO, strict checkpoint verification, and actual learned-policy capture using capture-tools v2. A completed training run is distinct from a demonstrated useful gait.

The original source, failed reports, checkpoints, retired manifests and removed progress automation remain unchanged. The camera follower is prepared but undeployed and respects the same shared control.
