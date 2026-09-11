# Mock geometry length study

49 generated URDFs: **femur and tibia independently take 50, 60, 70, 80, 90,
100, or 110% of the archived mock lengths**. `f100_t100` is the baseline.
Coxa geometry, yaw mounts, hip transforms and their inertials never vary.
The accurate CAD asset, archived mock, existing training defaults and checkpoints
are not modified.

| Dimension | Baseline | Study range |
|---|---:|---:|
| Femur: hip to knee | 145 mm | 72.5–159.5 mm |
| Tibia: knee to distal mesh tip | 210 mm | 105–231 mm |
| Coxa: yaw to hip, horizontal | 47.718 mm | fixed |
| Coxa: yaw to hip, 3D including vertical offset | 55.454 mm | fixed |

Regenerate from the repository root with:

```sh
python3 robot/tools/generate_length_study.py
python3 -m unittest discover -s robot/tests -p 'test_*.py'
```

Edit `study.json`, never the generated URDFs. The two tuning fields are
`femur_scales` and `tibia_scales`; there is no coxa scale. Each choice applies
symmetrically to all six legs. Original STLs are copied into `meshes/` so each
URDF resolves inside this package. Visual and collision meshes, their origins,
the knee offset and link COMs all follow the longitudinal length change about
the proximal joint. Thicknesses and transverse offsets stay fixed. This is
simulation mesh deformation, not a manufacturing CAD edit; holes along the
length also deform. Finalists need detailed CAD/linkage/clearance redesign.

## Mass, moments and motors

Each physical link receives its anatomically corresponding **mass, COM and full
inertia tensor** from `hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf`.
Total mass is **8.26081134 kg**, including the current motor-mass overrides;
motors are not added a second time. The current serial model's approximation of
the four-bar dynamics is inherited.

The source and mock use different local frames. Each coxa is registered by
positive yaw axis and yaw-to-hip direction. Each femur is registered by raising
axis and hip-to-knee direction; each tibia by opening axis and knee-to-distal-pad
direction. Right-handed orthonormal bases define a rotation Q. Transfer uses
`COM_mock = Q COM_source` and `I_mock = Q I_source Q^T`, with inertia measured
about the COM. The body uses aligned bottom planes and unchanged XYZ directions.
The complete transforms, anatomical leg mapping and source hashes are in
`manifest.json`. No tensor is blindly copied across incompatible frames.

**This sweep holds each transferred COM-centred tensor and link mass fixed.**
The femur/tibia COM's longitudinal offset scales with length. Intrinsic inertia
is not rescaled and structural mass is not reduced for shorter parts. This is
an explicit geometry ablation: it isolates leverage and geometry while retaining
the accurate model's intrinsic moments. It is not a prediction of a rebuilt
shorter leg's mass properties. CAD-derived inertials must be recomputed for
final hardware candidates before treating the result as a physical optimum.

URDF effort is the current source's **5.5 N m peak**; velocity is
50.26548246 rad/s. The manifest separately snapshots all current `DCMotorCfg`
parameters, including **1.6 N m continuous effort**, armature 0.0007 kg m²,
torque/speed saturation, PD gains and friction. URDF alone cannot encode the
continuous torque controller or motor curve. The supplied Isaac runner uses
that complete actuator configuration.

## Joint and contact conventions

The original **mock names, joint zero, signs, limits and body frame are kept**.
Femur zero is horizontal and positive raises it. Tibia zero is straight and
positive closes it. These differ from the current CAD angles. Current CAD
angle limits/stance must not be copied numerically into these models.
`body_mock` is the top of the mock plate; its bottom is at -0.023 m.
The seven empty export scaffolding links/fixed joints are removed, leaving
19 physical links and 18 independent revolute joints.

`link_joint_mapping` maps `lf lm lr rf rm rr` to original `revolute_*` names.
Joint/action order is unobserved until import and must be resolved by name.
These assets cannot be selected solely by changing the current CAD task's
`HEXAPOD_USD_PATH`: that task has different names, limits, reset coordinates
and contact thresholds. The standalone comparison runner does not use its
task configuration or archived policies.

Each tibia has its original mesh collider; the distal 7% of its +Y extent is
the study's provisional foot region. `metadata/*.json` records the resulting
length-dependent threshold and tip offset. This is not the accurate model's
silicone-pad collision geometry. Contact behavior must be checked after import.

## Static screening and Isaac comparison

`variants.csv` and `metadata/*.json` contain a common-posture static screening
at femur 35°, tibia 125° in mock coordinates. The six normal reactions solve
vertical force and roll/pitch moment balance; each joint's holding torque sums
those foot reactions and the gravity of its descendant links. This is neither
a learned gait nor a dynamic torque prediction. It has no controller transients,
slip, friction/compliance model or alternate-support gait. Torque figures should
not be used to rank dynamic performance alone. Reactions, balance residuals,
foot-height spread, non-foot vertex clearance and signed torques are retained.

Some long-femur/short-tibia combinations intersect the ground at this posture.
They are explicitly flagged, even if their static torque is low. Reset height
clears both feet and non-foot mesh vertices by 6 mm. An ineligible posture must
be changed and validated before training that variant; it is not automatically
a rejection of that segment length.

`experiments/c_length_study/tools/simulate_length_study.py` imports the new assets, authors their source
mass properties explicitly and verifies all 19 tensors numerically using
OpenUSD's own rotation transform. This is necessary because the saved September
4 review found an inverse-principal-axis defect in the deployed importer.
The repair touches only these newly generated study USDs. A CPU-only synthetic
USD test is `experiments/c_length_study/tools/test_length_study_usd.py`; it checks authoring/round-trip
mathematics, not an actual URDF import or PhysX simulation.

The finite live comparison uses one independent floating articulation per
variant in a 7 × 7 scene (columns: tibia, rows: femur, both ascending). All 49
advance together, with the 1.6 N m actuator, for 1,000 physics steps at 400 Hz.
It captures an actual Isaac render at step 500 and records observed joint order,
peak raw/applied torque and final state. It does **not** start PPO, certify a
standing acceptance gate, or manufacture an image when simulation is blocked.

On the Spark, use a separate frozen source snapshot and a new output path:

```sh
python3 experiments/c_length_study/tools/launch_length_study_spark.py \
  --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/source \
  --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/smoke_001 \
  --limit 1
# After the smoke test passes, use --limit 49 and a fresh output directory.
```

Reach the Spark over Tailscale SSH before running the launcher. It uses
nonblocking per-job GPU locks, refuses unrelated CUDA/Isaac jobs, keeps
16 GiB memory headroom, and has a 40-minute deadline. It stops only its own
container if another CUDA workload appears. It creates no persistent reservation
or hidden queue and never starts the legacy training service.

Before a training campaign: validate imported inertias/contact geometry, solve
and validate a stance for every admitted variant, verify motor loading under
motion, then compare policies trained from scratch with matched rewards,
commands, budgets and multiple seeds. Evaluate tracking, energy/duty, slip,
clearance, stability and failures. An insect-like appearance is not sufficient
evidence that a gait or leg length is optimal.
