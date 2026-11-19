# bm0 u1000: distilled policy, nylon body + 0.75 kg (surrogate diagnostic)

A single actor (278 inputs -> 18 outputs, no delay predictor) distilled on 2026-11-12 from
"predictor B (2 controls ahead) + pfit1 u2000". It was trained on a surrogate body whose centre piece is
nylon plus 0.75 kg, giving a robot of 7.280 kg (nominal model 7.466 kg). Source of record: the
"bmass1" entries of `hexapod-surrogate/runs/curriculum/decisions.md` (2026-11-12).

**Status: diagnostic only.** It fails the 96-case gate and is not in the lineage. It has not been run
on the normal (no-delay) simulator, where its teacher's recipe (distill1) scored 18/96 and failed both gates.

To run, judge, view, fine-tune or export it, see [GUIDE.md](GUIDE.md).

## Files

| file | sha256[:16] | what |
|---|---|---|
| `checkpoint_update001000.pt` | d84cb9285a1312da | the policy (actor + critic) |
| `checkpoint_update001000.json` | | its declaration (source hashes, config, seed) |
| `body_mass.json` | | the body change as applied in training (mass, centre of mass, inertia before and after) |
| `scripts/body_mass.py` | 074b57d6a0bfe922 | wraps any surrogate script so the body is the nylon + 0.75 kg body |
| `scripts/body_center_parts.json` | b42fe28f86a00265 | the nine printed body parts (CAD ledger volumes and densities) |
| `scripts/fitted_eval.py` | 3c6f4b0314a54d41 | the fitted-dynamics judge (all five fitted axes) |
| `full96_suite.py` | a4f5949ee1c039ab | the 96 cases; `fitted_eval.py` imports it from one folder up, so keep this layout |
| `scripts/distill_train.py` | fd9b35e3497c4d15 | the DAgger distillation trainer (reproduction only) |
| `scripts/delay_predictor.py` | 55d8c8151c21221e | predictor model code that `distill_train.py` imports (reproduction only) |
| `scripts/bmass1.sh` | f0eefa70ded1b49f | the exact training and judging commands (laptop paths hard-coded) |
| `judge_results/*.log` | | the two recorded fitted-check runs, case by case |

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

## Running it on another computer

From a hexapod-cupi checkout of this branch, with the project environment (`uv sync`):

```bash
uv run python records/distilled_bm0_u1000_20261112/scripts/body_mass.py run records/distilled_bm0_u1000_20261112/scripts/fitted_eval.py OUT_DIR bm0_u1000=records/distilled_bm0_u1000_20261112/checkpoint_update001000.pt --motor-check drawn --seed 20261110
```

Add `--match stand` for a 30-second smoke test (expect 3/3). Without the `body_mass.py run` prefix, the
same command judges the policy on the nominal 7.466 kg body.

Reproducing the training needs the teacher (pfit1 u2000) and predictor B, which are not in this folder.
They are on the original laptop under `hexapod-surrogate/runs/curriculum/` (`pfit1/`,
`predictor2_20261110/merged4/`).
