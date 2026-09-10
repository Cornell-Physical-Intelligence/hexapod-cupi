Verdict: the scope is sound, but four stated facts are unproven from the excerpts and one is wrong as worded.

**Must-fix semantic issues**

- **Step accounting is unproven.** Warmup advances PhysX at `physx_manager.py:768` and `physx_manager.py:786`, then `play()` pumps the app with playSimulations left True by initialize. On CPU the stage is never attached through IPhysxSimulation, per `physx_manager.py:762`, so whether that pump also steps PhysX is device-dependent and undecidable from this code. Label records as "authored USD", "post-reset", "after explicit step k", never "t=0". Call step with render disabled, since the render body is not in the excerpt.
- **"2.5 ms" needs three values to agree.** The cfg dt, the manager physics dt, and the scene timeStepsPerSecond must all be 400 Hz. A 2.5 ms simulate call against a 60 Hz scene does not produce one 2.5 ms step per call.
- **"No drive authoring" is not "no drives."** URDF import normally authors PhysicsDriveAPI with target zero, plus armature and joint friction. Those torques act during the eight steps. Without readback, zero-gravity motion cannot be attributed to self-contact.
- **"Self-collision as authored" is three attributes.** Root enabledSelfCollisions, per-joint collisionEnabled, and any filtered pairs. If the root flag is False, "no motion" proves nothing.
- **SDF view count is necessary, not sufficient.** A fallback convex shape silently drops out of the view. Close it with a USD count of mesh prims whose approximation token is sdf, plus a grep of the PhysX log for cooking warnings. Compare limits in one unit: USD authors degrees, the tensor view returns radians.

**First-run API uncertainties, fail closed if absent**

- An omni.physx physics-step subscription is the only read-only step counter. Prove it fires exactly eight times across the explicit steps before trusting its count during reset.
- SDF view factory signature, what check() asserts, and query frame and sign.
- Bodies of initialize_visualizers and the fabric resync after resume are not shown. Both simulation views are created with the warp backend at `physx_manager.py:777`, so CPU readback returns warp arrays.
- No known tensor getter for joint local frames. Read the USD joint attributes.

**Optional later**

- SDF probe at a far point and at a known mesh vertex to pin sign and frame.
- Contact offset, rest offset, and max depenetration velocity readback, since these shape overlap-driven motion.
- GPU re-proof. The CPU lifecycle differs at attach time, so CPU counts do not transfer.

**Actionable checks**

1. Subscribe a dt-logging step counter before reset. Expect at least two events during reset, exactly eight after, every dt 0.0025. Any surplus is timeline stepping.
2. Before stepping, read all 18 joints' stiffness, damping, max force, armature, friction, and scene gravity. Nonzero stiffness invalidates contact attribution.
3. Read the self-collision flag, per-joint collisionEnabled, and filtered pairs.
4. Count sdf-approximation prims and compare to view count. Both must equal the authored shape count. Grep the log.
5. Sum per-link masses against the canonical mass, read fixed_base from the view metatype, and compare limits in radians.
