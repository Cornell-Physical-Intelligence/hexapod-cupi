# Native physical coupling candidate (v5)

The complete v4 D6 diagnostic failed: 0.505577 mm pin separation versus the
unchanged 0.1 mm bound, 0.0289924 rad passive error versus 0.005 rad, and brief
missing foot support. Its maximum raw demand was 42.0734 N·m; applied torque
remained capped at 5.5 N·m. It is not admitted for PPO.

The v5 candidate retains the same 31 bodies, all CAD inertias/masses, 1,927
visual instances, 171 collision primitives, 18 motors, 30 tree coordinates,
reset, joint limits and motor settings. Geometry and kinematics file bytes
match v3/v4. Only loop formulation changes. The six external cut joints are
replaced by twelve native PhysX bilateral coordinate constraints:

- knee angle minus push-lever angle equals zero;
- rod angle plus push-lever angle equals zero.

These are the exact existing parallelogram branch relations in the CAD frame.
The six cut endpoints remain defined in the unchanged hashed kinematics JSON.
Their full 3D separation, axis error and passive-angle residual continue to be
measured from live link poses on every physical substep. No passive coordinate
is copied after reset, and no passive drive, extra inertia, or softer acceptance
bound is introduced. The original v3 task default remains unchanged.

PhysX explicitly documents mimic constraints as two-way impulse interactions,
including reaction on the reference coordinate. The associated constraint
Jacobian annihilates the allowed velocity tangent [1, 1, -1] for
[lever, knee, rod], so its ideal reaction does no work on permitted mechanism
motion. Keeping every body's original inertia retains the physical kinetic
energy on that constrained branch. Contact competition can still cause
numerical residuals; the candidate must pass the same live gates.

Sources checked against Spark's installed PhysX 110.1.13 schema and demo:

- [PhysX articulations and mimic dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
- [Omni Physics articulation support](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/articulations.html)
- [Mimic API and bilateral interaction](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_mimic_joint_a_p_i.html)

Installed schema:
`/isaac-sim/extscache/omni.usd.schema.physx-110.1.13+110.1.2.la64.r.cp312.u7f4/plugins/PhysxSchema/resources/generatedSchema.usda`.
`PhysxMimicJointAPI:rotZ` authors gearing -1/+1, offset 0,
referenceJointAxis rotZ, naturalFrequency 0 and dampingRatio 0. These values
select a hard physical constraint. The API is marked deprecated in this SDK
in favor of NewtonMimicAPI; its existing PhysX schema and native demo are
present. Live parsing and motion remain acceptance requirements.

CPU validation passes. This is a candidate, not a walking policy or hardware
qualification. The unchanged campaign sequence provides the live startup,
32-environment full nominal/refined comparison, then scratch/checkpoint and
full PPO gates. Self-collision and accurate rough-terrain contacts remain open.
