# bm0 u1000: distilled policy, nylon body + 0.75 kg (surrogate diagnostic)

A single actor (278 inputs -> 18 outputs, no delay predictor) distilled on 2026-11-12 from
"predictor B (2 controls ahead) + pfit1 u2000". It was trained on a surrogate body whose centre piece is
nylon plus 0.75 kg, giving a robot of 7.280 kg (nominal model 7.466 kg). Source of record: the
"bmass1" entries of `hexapod-surrogate/runs/curriculum/decisions.md` (2026-11-12).

**Status: diagnostic only.** It fails the former 96-case gate (19 and 20 of 96 under the fitted axes) and
is not in the lineage. Main's walking-requirements judge passes it 26 of 26 in all six surrogate runs, with
and without the fitted axes (see "Results under main's walking-requirements judge"). Every result here is a
surrogate result.

To run, judge, view, fine-tune or export it, see [GUIDE.md](GUIDE.md).
Its scripts and chain run on the pre-merge code, commit `41d3e243`; GUIDE.md says how. The one
exception is `scripts/requirements_fitted_judge.py`, which runs main's judge on the merged code.

## Files

| file | sha256[:16] | what |
|---|---|---|
| `checkpoint_update001000.pt` | d84cb9285a1312da | the policy (actor + critic) |
| `checkpoint_update001000.json` | | its declaration (source hashes, config, seed) |
| `body_mass.json` | | the body change as applied in training (mass, centre of mass, inertia before and after) |
| `scripts/body_mass.py` | 074b57d6a0bfe922 | wraps any surrogate script so the body is the nylon + 0.75 kg body |
| `scripts/body_center_parts.json` | b42fe28f86a00265 | the nine printed body parts (CAD ledger volumes and densities) |
| `scripts/fitted_eval.py` | 3c6f4b0314a54d41 | the fitted-dynamics judge (all five fitted axes) |
| `scripts/requirements_fitted_judge.py` | ab7d1176fa2909d4 | main's walking-requirements judge with the five fitted axes drawn per case (merged code) |
| `full96_suite.py` | a4f5949ee1c039ab | the 96 cases; `fitted_eval.py` imports it from one folder up, so keep this layout |
| `scripts/distill_train.py` | fd9b35e3497c4d15 | the DAgger distillation trainer (reproduction only) |
| `scripts/delay_predictor.py` | 55d8c8151c21221e | predictor model code that `distill_train.py` imports (reproduction only) |
| `scripts/bmass1.sh` | f0eefa70ded1b49f | the exact training and judging commands (laptop paths hard-coded) |
| (on Spark) | | the two recorded fitted-check logs, reduced case by case into `site/assets/curriculum_bm0_20261119_001/results.json` (`bm0_fitted_checks`); the raw logs are on Spark under `/srv/cupi/hexapod/runs/julian/pr108_raw_results_20261119/` |

## Training (2026-11-12, about 22:21-22:54 EST, laptop CPU MuJoCo surrogate)

- Teacher: pfit1 u2000 (e2ec2f0c…) with predictor B (1dff0e80…) in front, 2 controls ahead. Labels are
  the teacher's deterministic (tanh-mean) action.
- Student: starts from pfit1 u2000's weights. A fraction of robots is driven by the teacher: 0.5 at
  the start, falling to 0 by update 200. 5 epochs of MSE per update, Adam 5e-4, no anchor. 1000 updates.
- Task: destination_v70 (`terrain_v2_odom` destination input, speed-scaled gait clock 60), flat ground,
  128 robots, 36 steps per update, seed 20261110. Exploration noise 0.05, velocity noise 0.5.
- Dynamics: all five leg-stand fitted axes drawn per episode (action delay 1-2 controls, kd scale
  0.13-0.37, toe friction 0.45-0.6, joint friction, rotor inertia). Touchdown model off.
- Body: `body_mass.py`. The nine printed body parts are 583.6 cm^3; at nylon 6 (1140 kg/m^3) they
  weigh 0.665 kg instead of 1.602 kg at the CAD's aluminium densities. 0.75 kg is added as a point mass at
  the old body centre. Body link 1.909 -> 1.722 kg, robot 7.466 -> 7.280 kg.

## Results (fitted check, same body, all five axes, motor check drawn)

| test seed | passes of 96 |
|---|---|
| 20261110 | 19 |
| 20261111 | 20 |

Reference, judged the same way: sister seeds bm1 18, 17 and bm2 16, 19; the same recipe trained on the
old body, judged on this body: distill1 17, 18; dc1 18, 20; dc2 16, 21. Training on the new body made
no measurable difference.

## Results under main's walking-requirements judge (2026-11-20, merged code `d5b3c897`)

Main replaced the 96-case scorer with the walking-requirements judge (`src/locomotion/evaluation/requirements.py`,
set `stage2_flat`: stand, two speeds on eight bearings, two turns, two arcs and five routes; 26 cases). Each
case checks drift from the commanded path over a 20 s hold (0.10 m), stopping distance (0.05 m), distance to a
route's last goal (0.12 m), tilt while standing and after stops (1.7 deg) and the largest applied motor torque
(1.6 N m).

| run | fitted axes | body | seed | passes of 26 | worst drift | worst tilt | worst stop position |
|---|---|---|---|---|---|---|---|
| standard_body | off | standard | 20261110 | 26 (pyramidal 26, friction 0.9 26) | 0.071 m | 0.95 deg | 0.066 m |
| own_body | off | nylon + 0.75 kg | 20261110 | 26 (pyramidal 26, friction 0.9 26) | 0.074 m | 0.88 deg | 0.065 m |
| fitted_standard_20261110 | on | standard | 20261110 | 26 | 0.048 m | 0.73 deg | 0.065 m |
| fitted_standard_20261111 | on | standard | 20261111 | 26 | 0.050 m | 0.83 deg | 0.065 m |
| fitted_own_20261110 | on | nylon + 0.75 kg | 20261110 | 26 | 0.059 m | 0.71 deg | 0.065 m |
| fitted_own_20261111 | on | nylon + 0.75 kg | 20261111 | 26 | 0.059 m | 0.81 deg | 0.064 m |

Stopping distances stay at or below 0.014 m. The applied torque equals the 1.6 N m bound in every case,
standing included: on the surrogate the motors' rated cap is the bound, so that check cannot fail here.
"Fitted axes on" draws the five ranges of `scripts/fitted_eval.py` per case (each run drew delays of 1 and 2
controls and kd scales from 0.14 to 0.37), with the touchdown model off and the motor-law check against each
case's drawn damping; it skips the contact consensus, which varies friction. The old scorer failed this
policy mostly on speed and turn-rate tracking inside short windows; the new judge bounds drift over a
whole hold and has no turn-rate bound.

Raw summaries and judgements: `/srv/cupi/hexapod/runs/julian/pr108_bm0_judge_20261120/` on Spark
(read-only, SHA256SUMS), reduced into `site/assets/bm0_requirements_judge_20261120_001/results.json` by the
`build_results.py` beside it. The 400 Hz captures were not kept. Commands, from a checkout of the merged code:

```bash
# without the fitted axes (add the body wrapper for the nylon body, as below)
.venv/bin/python -m locomotion.surrogate.evaluate --checkpoint records/distilled_bm0_u1000_20261112/checkpoint_update001000.pt --output OUT --cases all --video none
# with the fitted axes, on bm0's own body
.venv/bin/python records/distilled_bm0_u1000_20261112/scripts/body_mass.py run records/distilled_bm0_u1000_20261112/scripts/requirements_fitted_judge.py --checkpoint records/distilled_bm0_u1000_20261112/checkpoint_update001000.pt --output OUT --cases all --video none --seed 20261110
```

`evaluate.py` on main ends with `KeyError: 'metrics'` after it writes `surrogate_summary.json`: its closing
printout reads a field the judge no longer writes. The summary is complete; `requirements_fitted_judge.py`
catches the error.

## Running it on another computer

From a hexapod-cupi checkout of this branch, with the project environment (`uv sync`):

```bash
uv run python records/distilled_bm0_u1000_20261112/scripts/body_mass.py run records/distilled_bm0_u1000_20261112/scripts/fitted_eval.py OUT_DIR bm0_u1000=records/distilled_bm0_u1000_20261112/checkpoint_update001000.pt --motor-check drawn --seed 20261110
```

Add `--match stand` for a 30-second smoke test (expect 3/3). Without the `body_mass.py run` prefix, the
same command judges the policy on the nominal 7.466 kg body.

## Rerunning the whole chain from the v70 odometry policy

`pipeline/run_chain.sh` reruns everything from od1 u1200 (`records/curriculum_20261109/od1`) to
bm0, then judges the result: predictor B, then pfit1, then bm0, then the judge. It takes about 2 h on
CPU. `pipeline/record_run.sh` files the finished run under `records/` for a commit. Steps, recorded
results and push rules are in [pipeline/PUSHING_A_RUN.md](pipeline/PUSHING_A_RUN.md).

`lineage/` holds the two intermediate models as recorded, so the chain can start partway through:
`lineage/predictor_B/` (predictor.pt 1dff0e80c45da086, its data record and held-out report) and
`lineage/pfit1_u2000/` (checkpoint e2ec2f0c433d21d8, declaration, predictor_in_loop.json, train
log). `scripts/predictor_train.py` (f7c892af13edef8d) trained pfit1; `scripts/fitted_report.py`
(cf6cad06c08dfdc8) is imported by `delay_predictor.py`.
