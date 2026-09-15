# Cold BC onset distribution diagnosis

The measured pattern is **a close initial observed context followed by rapid
departure from demonstrated states**, plus a specific fitting error at the first
teacher onset action. The evidence does not support saying that every cold
input is absent from the dataset, that local target averaging explains the
failure, or that a warm-start trial will succeed.

Frozen source 017 exactly restores the fit 003 actor. Fit 004's full model and
normalizer tensors are bitwise identical; both checkpoints contain zero PPO
updates and zero native training transitions. CPU actor outputs reproduce the
recorded evaluation 012 actions to a maximum absolute difference of 4.18e-7 in
normalized action units. No fitting or native simulation is performed here.

The BC dataset contains 3,760 actual native rows. The +0.05 m/s forward subset
has 120 accepted onset rows from raw controls 200–259 and 120 steady rows from
controls 260–319, each from replicas 9 and 30. These are raw control indices, not
dataset row indices 200–259. The onset follows four seconds of native neutral
settling; evaluation 012 starts its policy immediately from a physical reset.

Distances use the checkpoint's actual saved normalization, including
sqrt(var+1e-6) and the actor's [-10,10] normalized-feature clamp. Each value is
RMS over the included scalar features, not a probability or qualification gate.

| Nearest +0.05-forward observation distance | First 100 controls | Later 900 controls |
|---|---:|---:|
| All 231 features, either onset or steady |2.03374|1.57982|
| All 231 features, onset only |2.12786|1.77451|
| Joint-position history 90, independent nearest |1.41166|2.25498|
| Joint-velocity history 90, independent nearest |2.18473|0.198677|
| Gyro history 15, independent nearest |2.02399|0.148246|
| Gravity history 15, independent nearest |1.76975|0.286056|
| Previous held target 18, independent nearest |1.14004|1.72566|
| Command 3 |0|0|

For descriptive context, each forward demonstration's nearest *different raw
control* has mean distance 0.373902, p95 1.02560 and maximum 1.17972. Ninety-four
of the first 100 evaluation controls, and all later 900, exceed that empirical
maximum. This is not an invented OOD cutoff: the reference trajectories and
adjacent controls are correlated, and the distance has no calibrated
generalization or safety interpretation. Per-group nearest rows can differ;
`RESULT.json` also records every group's distance to the same jointly selected
neighbor, avoiding a claim that separately close components form one close row.

The exact first observation is an important exception. It lies within every
dataset coordinate's observed range and is only 0.217287 from the nearest
forward-onset row (dataset 1884, replica 30, raw control 200). Command and previous
held target match exactly. Its current joint-position difference is 0.02218 rad
RMS, maximum 0.02771 rad. The pre-hold root heights are 0.1028023 m cold versus
0.0957616 m at demonstration onset, a 7.04 mm physical-context difference. Root
height and explicit contact state are not actor input channels, so observed
feature closeness cannot establish identical initial physical conditions.

At evaluation control 4, the nearest onset's distance is already 2.07171,
dominated by joint-velocity history at 3.24967 normalized RMS. Later, the robot
is relatively quiet in velocity while its pose and held targets remain distant.
Across all later 900 controls, exactly six current joint-position channels lie
outside the entire dataset's marginal range: `lf_coxa_yaw`, `lf_tibia_pitch`,
`lm_coxa_yaw`, `lr_coxa_yaw`, `rf_tibia_pitch`, `rr_tibia_pitch`. This corresponds
to one-third of all 90 q-history scalars. Previous-held-target scalars are outside
their dataset ranges 38.43% of the time. The command itself is represented.
These are directly observed distribution differences, not proof of their cause.

The actor fits its own forward-steady teacher inputs with mean target RMS
error 0.00173144 rad and forward-onset inputs with mean 0.00256590 rad. In contrast,
the recorded cold actor differs from the nearest same-command teacher's
requested target by mean 0.0762202 rad in the first 100 controls and 0.100097 rad
in the later 900. Actual held-target differences, after the existing limiter,
are 0.0758150 and 0.100097 rad respectively. Teacher inputs and cold states are
different, so these comparisons are not supervised prediction-error labels for
the cold state and do not define an executable replacement controller.

There is also a localized onset fitting error *on actual training inputs*.
Both replicas have identical first-onset requested labels. Their teacher change
from the previous held target is 0.0730585 rad RMS; fitted actor changes are only
0.0419328/0.0420739 rad RMS, leaving 0.0320930/0.0319157 rad teacher error. The
cold first actor change is 0.0436524 rad RMS. However, the original actuator
limiter reduces the cold first held-target difference from the teacher to
0.00516133 rad RMS, maximum 0.0111717 rad. A large requested-label discrepancy
does not transfer unchanged into actual actuation or prove the later failure.

Local averaging is not established. Over the later 900 states, the eight nearest
same-command teacher labels have mean within-neighborhood spread 0.0105616 rad,
while the actor is 0.100860 rad from their mean—roughly the same discrepancy as
the nearest single teacher. Results for 1/4/8/16 neighbors are all retained;
these neighborhood sizes are descriptive sensitivity comparisons, not gates.
Identical first-onset labels across the two forward replicas also do not provide
evidence of conflicting labels at that initial command. A neural network need
not implement nearest-neighbor averaging, and these observations cannot exclude
other regression or closed-loop mechanisms.

The narrow actionable finding is to keep two issues separate: the original
cold physical context differs from the settled demonstration context, and the
fitted policy under-reproduces the first teacher onset request even on its
training inputs. An independently recorded context-controlled trial can test
the former; this CPU diagnosis supplies no result for that trial. Later posture
departure and action disagreement are measured, but which difference initiates
the divergence remains unproven. All physics, source, checkpoints, raw
measurements and Stage 2 gates remain unchanged.

`analyze.py` reproduces the normalized nearest-neighbor, coordinate-range and
teacher-action comparisons. `initial_action_detail.py` reproduces the initial
training-input and later per-joint findings. `INITIAL_PHYSICAL_CONTEXT.json`
records exact raw array selectors and hashes for root-height and held-target
comparisons. `per_row.npz` retains every nearest row/distance and the recorded
and CPU actor actions. `RESULT.json` pins 13 source/checkpoint/data inputs and
checks that their bytes remain unchanged. No warm evaluation is read.
