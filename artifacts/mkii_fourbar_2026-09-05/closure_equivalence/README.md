# Redundant four-bar closure rows — 2026-09-05

**The planar articulation tree already enforces the closure's axial separation
and two axis-alignment conditions.** Two hinge-local transverse position
constraints supply its remaining independent closure conditions. This supports
testing a D6 closure with local X/Y translation locked and local Z/all rotation
free, while retaining the original six full pin-position and axis checks.
This audit reads the existing revolute-closure v3 asset; it neither changes that
asset nor admits a replacement for training.

[audit.py](audit.py) derives an all-angle geometric bound and independently
checks grids against both the double-precision kinematic contract and the actual
USD local joint frames. [report.json](report.json) records source SHA-256 hashes,
the complete geometry used, bounds and sample results. Authored float quaternions
are normalized when interpreted as rotations.

## Results

The bound applies to **all independent A, B and D angles**, including states
outside the closed branch. It does not assume `qD=qA` or `qB=-qA`, and is not
limited to a sampled workspace. The independent grids contain 78,732 total
configurations: every leg in both source representations, using a hard-limit
grid that includes the nominal stance, plus a full-rotation grid.

| Leg | All-angle axial-gap bound from authored v3 USD |
|---|---:|
| lf | 0.0214 nm |
| lm | 0.0565 nm |
| lr | 0.2201 nm |
| rf | 0.2698 nm |
| rm | 0.2822 nm |
| rr | 0.2975 nm |

The largest authored-USD axis-chord bound is approximately 1.12e-16; the
double-precision contract's largest axial-gap bound is 5.30e-17 m. A separate
1e-12 numerical comparison margin covers floating-point evaluation of the
formulas and grids, giving a worst axial bound below **0.299 nm** with that
margin. These authored-frame discrepancies are far below the unchanged live
0.1 mm full-position and 0.1-degree axis gates. They do not bound floating-point
error in live body-pose reporting or establish solver convergence.

The closest ideal parallelogram toggle lies approximately 0.1634 rad beyond the
lower hard motor limit; neither flattened configuration is inside any leg's
current hard interval. This does not replace runtime branch checks, measured
hardware stops, collision/workspace qualification or checks of the candidate's
full position/axis residuals.

## Why the three omitted rows are implied by the tree

All quantities below are expressed in the femur frame at the raw CAD zero.
`A` is femur→pushlever, `B` is pushlever→rod, `D` is femur→tibia, and `C0/C1`
are the two sides of the rod→tibia closure. Let `n` be the A axis; `nB`, `nD`,
`n0`, `n1` are the corresponding B, D and C-side unit axes. Rotating the coxa,
femur or floating root applies a common rigid transform to the entire mechanism
and therefore preserves these distances and relative-axis conditions.

For arbitrary independent angles `a`, `b`, `d`, the tree gives:

```text
p0(a,b) = pA + Rn(a) [pB-pA + RnB(b)(pC0-pB)]
p1(d)   = pD + RnD(d)(pC1-pD)
n0(a,b) = Rn(a) RnB(b) nC0
n1(d)   = RnD(d) nC1
```

In the ideal planar geometry every axis equals `n`, and all relevant pin
displacements lie in its perpendicular plane. Rotation about `n` preserves
the axial component, so axial separation and axis alignment are independent
of all three angles. The original closure's five constrained directions
therefore have only two independent effects on this articulation tree. Local
X/Y coincidence supplies those two conditions. Rotation around the shared axis
was already free in the original revolute closure.

The calculation also bounds imperfect authored frames without assuming exact
parallelism. Define:

```text
delta0 = ||nC0-nB|| + ||nB-n||
delta1 = ||nC1-nD|| + ||nD-n||
L      = |AB| + |BC| + |DC| + |AD|

axis chord <= delta0 + delta1
axial gap in C0's frame <= |n·(pC0-pC1)|
  + 2|BC| ||n×nB|| + 2|DC| ||n×nD|| + delta0 L
```

These follow from rotation preserving lengths and from the maximum change of
a vector's projection under rotation about a nearly parallel axis. They hold
for every angle, rather than estimating an unsampled maximum from a grid.
If local X/Y closure error is zero, the full pin separation is precisely the
bounded remaining axial component. A nonzero transverse solver error still
counts against the original full-position gate.

## Scope and necessary dynamic experiment

The result establishes the redundant rows of this regularized rigid planar
geometry. It does not identify physical bearing compliance, uniquely determine
redundant bearing reaction loads, or establish dynamic equivalence for an
arbitrary nonplanar linkage. The test deliberately rejects a 1 mm axial offset
and a tilted closure axis; this reduction must not be applied to other assets
without re-derivation.

The motivation is the 800 Hz v3 run's continued driven closure error. Removing
redundant solver rows is a candidate numerical correction, not evidence that
they caused the failure. Compare a separately versioned candidate against the
same complete standing/driven sequence, motor envelope, gains and masses.
Record pre-step motor position/velocity, P and D torque terms, delivered torque,
post-step velocities, contact impulses and each C-pin's transverse/axial error.
Keep all six original full pin/axis checks, passive branch checks and admission
limits. Neither the geometric audit nor a short probe grants training admission.

NVIDIA documents that an excluded ordinary joint can close an articulation,
while the internal tree uses reduced coordinates. See
[articulation topology](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html)
and [configurable D6 constraints](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/Joints.html).
The equivalence calculation above is this project's derivation, not a vendor
guarantee for this robot.

## Reproduce

From the repository root, with NumPy and the standalone USD library installed:

```sh
.venv/bin/python artifacts/mkii_fourbar_2026-09-05/closure_equivalence/audit.py --out /tmp/fourbar-closure-equivalence-new.json
.venv/bin/python -m unittest discover -s artifacts/mkii_fourbar_2026-09-05/closure_equivalence -p 'test_*.py'
```

Output must be a new file; existing evidence is never overwritten. Four tests
check exact planar geometry, reject an axial offset and tilted axis, and verify
the all-angle inequalities against independent Rodrigues rotations with
deliberately imperfect axes. The report's grids separately verify the source
contract and authored v3 USD frames.
