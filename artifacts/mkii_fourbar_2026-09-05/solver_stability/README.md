# CAD inertia and explicit-PD stability diagnostic — 2026-09-05

The small pushlever inertia helps explain sensitivity to an incompletely solved
four-bar constraint. **This calculation supports testing 1.25 ms physics; it
does not establish that PhysX is stable or identify the live failure's cause.**
The unchanged standing, driven-reversal and closure gates must decide that.

The physical asset, motor gains and inertia inputs are unchanged by this audit.
[report.json](report.json) records their exact SHA-256 identities and all numerical
results. [audit.py](audit.py) is a CPU-only calculation using the nominal CAD pose,
all 31 body masses, COM offsets and full inertia tensors, and the configured
0.0007 kg·m² armature on each of the 18 active motors.

## Result

The table reports the lightest mode seen by the active motors when the body is
free. The first model enforces the ideal parallelogram's passive relations;
the second deliberately omits the six external C-pin constraints while keeping
all 30 articulation coordinates. The latter represents a diagnostic limit in
which the closure has not transferred the tibia's load to the active pushlever.
It is not a replacement robot model or a reconstruction of a PhysX iteration.

| Nominal model | Minimum active-mode inertia, kg·m² | Semi-implicit critical timestep | Constant-acceleration/ZOH critical timestep |
|---|---:|---:|---:|
| Ideal closed four-bar, free body | 0.00217498 | 6.268 ms | 7.250 ms |
| Unclosed tree limit, free body | 0.000724673 | 2.285 ms | 2.416 ms |

With the body fixed, those minimum inertias are respectively 0.00275974 and
0.000725014 kg·m². Individual floating-body pushlever effective inertias are
0.002834–0.002849 kg·m² with ideal closure, versus approximately 0.00072499 kg·m²
without the external closure. The light unclosed mode is dominated by the
pushlever coordinates and their provisional armature.

For that light unclosed mode, the negative discrete pole changes as follows.
Magnitude above one predicts a growing alternating response in the stated
linear model; all poles, including the other pole, are recorded in the report.

| Physics timestep | Semi-implicit pole | Constant-acceleration/ZOH pole |
|---|---:|---:|
| 5 ms | −3.9664 | −3.4233 |
| 2.5 ms | −1.2116 | −1.0745 |
| 1.25 ms | −0.0373 | −0.0028 |

Thus 5 ms is not intrinsically unstable in the **ideal closed** stance
linearization. However, 2.5 ms remains marginal for the light **unclosed**
diagnostic under both integration assumptions. Testing 1.25 ms with decimation
16 preserves the 50 Hz policy interval and gives that diagnostic more margin.
Torque clipping can bound delivered effort while leaving an oscillating
response; an applied-torque envelope pass alone does not certify stability.

## Method and checks

Centred finite differences at 1e-6 rad form each link's translational COM and
angular Jacobian. The kinetic-energy matrix sums
`m JvᵀJv + Jwᵀ(R Icom Rᵀ)Jw` over all links, then adds active armature. Floating
base translation/rotation are included. For the motor coordinates, the inverse
of the selected block of `M⁻¹` gives effective inertia with unforced base and
passive coordinates free to respond. With equal gains on every active motor,
its eigenmodes reduce the local system to scalar PD examples.

The independent finite-difference step check at 1e-5 rad changes matrix entries
by at most about 1.53e-11. Projecting the independently differentiated 30-joint
tree matrix through the ideal passive-coordinate mapping reproduces the
directly differentiated closed matrix; the exact residual is in the report.
All matrices must be finite and positive definite before results are emitted.

For each mode, `I q̈ = −Kp q −Kd q̇`, with `Kp=30 N·m/rad` and
`Kd=0.6 N·m·s/rad`. The semi-implicit update advances velocity first and position
using the new velocity. The ZOH comparison holds torque constant through the
step and integrates constant acceleration exactly. Their stability boundaries
are, respectively:

```text
h < (sqrt(Kd² + 4 Kp I) − Kd) / Kp
h < min(2 I / Kd, 2 Kd / Kp)
```

Neither update reproduces the complete TGS solver. The diagnostic omits contact
constraints and impacts, gravity stiffness, changing geometry, velocity-dependent
inertia terms, joint friction, torque-speed clipping and burst-budget dynamics.
It assumes the nominal pose and gains, not all reachable configurations. The
unclosed case is a limiting comparison, not a proven bound on the partially
converged constrained solver. Hardware gains and armature still need leg-stand
identification.

NVIDIA recommends reducing timestep for difficult closed loops and checking
mass/inertia and drive stiffness. Its TGS documentation also distinguishes
internal solver iterations from complete smaller simulation steps. These support
the proposed experiment, not a predicted pass. See the
[articulation stability guide](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/guides/articulation_stability.html)
and [PhysX simulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html).

## Reproduce

From the repository root, with Python and NumPy installed:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/solver_stability/audit.py --out /tmp/fourbar-inertia-new.json
python3 -m unittest discover -s artifacts/mkii_fourbar_2026-09-05/solver_stability -p 'test_*.py'
```

The output must be a new file; the script refuses to replace existing evidence.
Omit `--out` to print JSON. Source identities may change in later commits;
compare the report's hashes before comparing results. Four analytic tests check
both integration boundaries, the ZOH update, passive-coordinate inertia
reduction and the distinction between 2.5 ms and 1.25 ms for the light mode.
