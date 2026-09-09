# External first-policy evaluator — implementation candidate

This directory is outside the training identity's hashed code paths. **Isaac/RSL
execution has not been exercised.** Pure scenario/scoring and preflight tests are
local CPU tests; they do not validate native simulator integration. Root review,
an owned bounded launcher, and one actual small execution are required before
using the evaluator's results. No policy or physical admission is supplied here.

Copy this complete directory to an evaluation location outside the chosen frozen
training snapshot. Mount that snapshot read-only. The driver rejects execution
from inside it and records separate training and evaluator hashes. Use the same
admission and selected asset as the checkpoint; unrelated source changes cannot
be excused by the evaluator. Do not add this code to frozen `tools/`, `isaaclab/`
or package directories, or rewrite a checkpoint sidecar.

Preflight (stdlib only; no Isaac launch):

```sh
python3 /evaluation/evaluate.py \
  --source /frozen/training-source --admission /evidence/admission.json \
  --checkpoint /evidence/checkpoint.pt --report /new-output/preflight.json
```

For a zero-action baseline, use `--controller zero` and omit `--checkpoint`.
For reviewed live execution, use Isaac's Python launcher and add
`--execute --viz none --device cuda:0`, with a fresh output directory. **There is
currently no integration with `run-mkii-fourbar`'s ownership/timeout supervisor.**
Do not run the live command directly on a shared Spark. A host wrapper must acquire
the shared lock, check unrelated producers, bind this driver and its frozen source,
enforce a finite wall timeout, archive exact bytes, and clean up only its owned
container. The evaluator itself never starts Docker, other processes or training.

The protocol is fixed: 14 parallel scenarios, 500 control steps each, repeated with
seeds 101/202/303. Each controller gets 42 trials and 1,500 batched control steps.
Signed forward/left/yaw, diagonals, standing, stops and reversals use the amplitudes
and windows in `evaluation_math.py`. These deterministic flat-ground repeats are
not independent terrain trials. Run the earliest verified scratch checkpoint,
selected full checkpoint and zero-action baseline separately with identical inputs.

`--execute` verifies lineage before Kit, selects the admitted USD, compares the
complete loaded runtime manifest, loads the actual RSL5.0.1 actor/normalization,
switches inference to evaluation mode and checks that learner state is unchanged.
Commands are installed before observations and are checked in the84-element policy
input. Random command refresh is suppressed through a separately recorded external
adapter; gains, actions, motor scheduling, resets and physical parameters are not
changed. Inactive trial slots use zero actions after their first terminal event.

The frozen trainer's physical guard runs on every physics substep. The observer
captures terminal state before automatic reset; scoring excludes later respawns.
Isaac Lab 3 returns `root_quat_w` in **XYZW** order. Both native XYZW and explicitly
converted WXYZ values are archived; the latter is used only by the pure scoring
math. Each actual reset must have the frozen task's identity orientation before
rollout. Reviewed SDK excerpts/hashes are in `sdk_contract_evidence.json`; loaded
quaternion/reset/RSL implementation files must match those hashes. The installed
explicit reset calls `sim.forward()` but not `scene.update()`. The driver also
rejects any reset that unexpectedly generates guarded physics samples.
Motor demand/applied/clipping/overload/current-proxy extrema and headroom minima
cover every physics substep. Trajectory, support/slip, acceleration and reward
components are sampled at50Hz and labeled accordingly. Contact/closure checks in
the physical guard retain full substep coverage. Zero loaded feet yields zero
loaded-foot slip and must be interpreted with the support distribution.

Use an empty output directory. Outputs include `report.json`, `progress.json`, a
copy of the separately hashed evaluator source, and three hashed per-seed JSON
traces with initial poses, executed commands, trajectories and terminal reasons.
Per-trial summaries include directional tracking, path/cross-track error, stops,
reversal delay, roll/pitch, support/contact, motor usage and reward components.
Trace data supports plots; this version does **not** render trajectory plots.
The report always has `pass=false`, both admission flags false and
`policy_quality_pass=null`. `evaluation_complete` means the fixed protocol was
observed; `physical_integrity_pass` means its existing numerical guard passed.
Falls remain failed trials, not hidden resets. No mission performance threshold
is introduced. Critical evidence is saved before `env.close()`/Kit shutdown.

Tests, run from this directory:

```sh
python3 -m unittest discover -s . -p 'test_evaluation.py' -v
```
