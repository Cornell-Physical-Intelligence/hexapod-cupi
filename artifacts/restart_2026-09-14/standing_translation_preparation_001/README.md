# Single-robot standing placement preparation

**Prepared and CPU-checked; no native execution, standing admission or PPO.**
This is the smallest saved-analysis proposal for investigating the detailed,
motor-corrected 7.466088235 kg robot. Two fresh single-robot arms compare world
XY `(0,0)` and `(14,4)` using the same dynamics, reset, floor and original gates.
The latter coordinate is batch environment23, which recorded seven support losses.
One run per arm can establish event recurrence only, not a cause or failure rate.

`OUTCOMES.json` freezes the proposed interpretation before dispatch. The ground
is an80m square made from two triangles, with seam `y=x`: origin is on that seam,
while `(14,4)` is inside a triangle. This changes location relative to tessellation
as well as numerical coordinates. No result should be called a distance effect.

## Exact preparation

- `parents/source005`, `parents/host007`, `parents/guard005` preserve complete,
  byte-identical archived inputs. Their old notes and launch names stay historical.
- `source/` is a new standalone replay source, not a production-package import.
  `--num-envs 1 --placement origin|xy14_4` is mandatory. The selected placement is
  part of the new source identity and both authored and physics-view reset evidence.
- `host/` is a pinned successor preserving the original host interfaces. A fresh
  external binding must match source, host, placement, output and current compute
  coordination. Both example bindings are deliberately disabled.
- `analysis/` provides an all8000-step event readout. Its FP32-product/FP64-sum
  aggregation functions are exact copies of the saved analyzer; the existing
  strict `>1 N` support detector now also reports settling samples and isolated
  one-step zero-force events. The128-record pattern remains an observed signature,
  not an inferred internal contact capacity or proven separation mechanism.

Only source entry placement/metadata, initial reset declaration, non-admitting
identity, and host bindings change. The robot and153 SDF colliders, solver32/0,
400Hz servo,50Hz control, geometry, friction assumptions, motor envelope,
standing scorer and quiet thresholds retain their original bytes. Both arms
append the same transform operation, including a zero offset at origin. Reset
preserves source005's float32 root write and existing2e-6 native readback tolerance.
No first-step zero-motion or between-arm equality gate is introduced.

The original live floor/scene fields and native body/joint order are compared
with source005's recorded single-robot values. The new scene-attribute census
is read-only. Fields absent from old evidence cannot retrospectively establish
backend-default equivalence, and USD solver readback is not internal solver
instrumentation. The full existing per-step force, pose, rate and contact capture
is retained, including failed prefixes.

## Proposed calls and ownership

Root's selected fresh remote parent is
`/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001`, containing
`source/`, `host/`, `guard/`, `bindings/`, `origin_001/` and `xy14_4_001/`.
The exact actual external bindings and guard live outside the frozen inputs.

The following are **host CPU preflight examples**, not native launch authorization:

```sh
/usr/bin/python3 -B -S /home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001/host/launch_standing_spark.py \
  --source /home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001/source \
  --asset /home/orionh/HEXAPOD_runs/canonical_direct_20260910/asset_001 \
  --admission /home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_actuation_001/actuation \
  --supervisor-source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009 \
  --output /home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001/origin_001 \
  --num-envs 1 --placement origin \
  --bindings /home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001/bindings/origin.json \
  --preflight-only
```

The second call uses `--placement xy14_4`, output `xy14_4_001` and binding
`bindings/xy14_4.json`; all other arguments are identical. Bindings must be fresh
and valid even for preflight. A dedicated root-owned guard then invokes the host's
`verify_inputs`, `command(args,name,'standing')` and terminal contract while owning
both locks and the exact container lifecycle. It must not nest the old supervisor's
lock acquisition. Necessary guard changes include the new exact source/host hashes,
current coordination and reservation identity, arm/output binding,1200-second cap,
AppReady timeout, exact owned-container cleanup and post-exit source verification.
The old guard's previous-owner checks and output/unit names cannot be reused.

A completed acquisition with failed standing gates remains a failure. Preserve
its report and raw evidence, run the event analysis, and allow the other independent
arm after verified cleanup. Missing or corrupt evidence never counts as absence.
Even two passing arms admit neither batch standing nor training.

After terminal audit, the stdlib analysis call is:

```sh
/usr/bin/python3 -B -S /path/to/analysis/readout.py --phase /path/to/origin_001/standing
```

It writes JSON to stdout and rechecks each consumed sealed input after analysis.
Root should redirect into a fresh result directory, then repeat for `xy14_4_001`.

## Validation and review

`TESTS_source_002.txt` records28 passing inherited source tests: exact servo rows,
contact rules, clock/counter/partial-failure handling, raw scoring and rejection.
The new copy's fixtures add explicit origin and diagnostic identity; the original
fixtures remain unchanged under `parents/`. Two stale temporary test paths were
resolved to their existing repository homes. `TESTS_translation_003.txt` records10
additional passing fixtures, including actual translated physics-view reset with
no extra step, inconsistent placement/drift/USD-readback rejection, old-source
binding rejection, all8000-step detector timing and exact leg naming, unchanged
parent/gate bytes, native-source compilation and standalone stdlib host loading.
The inherited positive terminal fixture verifies that every gate passing still
returns `standing_admission=false`, `batch_admission=false` and no training.

Initial preparation errors and their corrections are retained in
`TEST_ATTEMPT_001.json`. Independent code review found and helped fix a helper-name
collision and a standalone host import problem before native execution. The Fable
5.1 MAX conceptual review, session `6511ce0d-8efa-420a-83b2-7902f2e1a2e7`, approved
the placement pilot conditionally; root owns its full review/disposition receipt,
final independent code review, live admission, allocation and publication.

Run `python3 -B -S verify_bundle.py` here for portable byte verification. It launches
no process or simulator and makes no remote call. `READINESS.json` holds exact
source/host freezes. The separate operational guard and actual runs remain root-owned.
