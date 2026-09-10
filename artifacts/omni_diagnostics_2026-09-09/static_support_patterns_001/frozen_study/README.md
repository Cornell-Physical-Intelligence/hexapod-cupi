# Nominal C support-pattern feasibility001

10 September 2026. **Nominal alternating tripod has conditional static torque headroom; it is not ruled out by this study.** It depends on lateral contact-force allocation and friction. An opposite-leg-pair sequence retaining four supports has substantially more reserve and is a sensible intermediate physical test toward faster, smooth locomotion. No new gait is selected or launched here; the present slow single-leg wave remains a feasibility experiment, not a speed ceiling.

This uses the exact frozen C serial study geometry and stored CAD-derived link masses/COMs:8.26081134kg, current nominal stance, unchanged coxa/femur/tibia lengths and current1.6N·m motor contract. It is not the production four-bar model, a manufacturing validation or a new payload configuration. The pose and every model parameter remain unchanged.

## Results at the fixed nominal stance

Ranges below bracket the **ideal static minimum peak motor demand**, with the worst of the listed symmetrical/support choices shown. They do not predict the current position controller's actual torque.

| Available/load-bearing pattern | Vertical forces only | Assumedμ=0.3 | Assumedμ=0.6 |
| --- | ---: | ---: | ---: |
| Alternating tripods LF+LR+RM / LM+RF+RR |2.159N·m|1.303–1.320N·m|1.194N·m|
| Three opposite-pair four-support sets |1.129N·m|0.670–0.679N·m|0.622N·m|
| Worst of all six five-support choices |1.155N·m|≈0.679N·m feasible witness|0.637N·m|
| All six supports |0.686N·m|0.376N·m|0.376N·m|

The tripod'sμ0.3 feasible solution leaves approximately0.280N·m (17.5%) below1.6; atμ0.6 the ideal optimum leaves0.406N·m (25.4%). Its support-polygon margin is approximately115mm. At theμ0.6 optimum the load is not evenly divided: the middle supporting foot carries about36N, and the two corner feet about22.4–22.6N. Horizontal reactions are nontrivial; the saved witness uses friction ratios up to approximately0.50. A naïve total-mass/three-feet load estimate misses those moments.

For both nominal alternating tripods, the ideal circular-cone friction threshold for staying at or below1.6N·m lies approximately between **μ0.196 and0.200**, bounded by the outer/inner approximations. This is not a measured terrain friction estimate or a safe operating threshold; it leaves essentially no torque margin for motion or uncertainty.

The three balanced four-support sets correspond to swinging **LM+RM**, **LF+RR**, or **LR+RF**. Their projected support margins are186,115 and116mm, respectively. At the same swing duration a three-phase opposing-pair sequence could progress through all six feet with fewer phases than the present six-phase single-leg wave. These static results justify a bounded investigation; simultaneous swing inertia, load transfer, contact forces and actual target continuity still need proof.

All42combinations of3/4/5/6declared supports were enumerated. Requiring every declared supporting foot to carry at least1N leaves2/20tripods,9/15four-foot sets,6/6five-foot sets and1/1six-foot set with nominal static equilibrium. Several otherwise mathematical combinations balance almost exactly on an edge or assign essentially zero load to an alleged support; they are excluded by the1N requirement. Arbitrary four-foot patterns are not equivalent to the balanced opposing-pair sets.

## What the optimization actually proves

For each joint, downstream link gravity is explicitly included:

`tau_gravity,j = Σ axis_j · ((COM_link − pivot_j) × mass_link g)`.

The actuator demand is `−(tau_gravity + JᵀF_contact)`. Whole-robot force and moment balance use every leg and body mass/COM. Non-support feet have exactly zero contact force but their gravity torques remain in the18-motor objective. Each declared supporting point has normal force≥1N, an explicit positive-load convention for this study, not a new simulator gate; no adhesion or contact moment is allowed.

A linear program minimizes the largest absolute motor torque. The **outer16-facet friction polygon** relaxes the circular Coulomb cone, giving an optimistic lower bound on this ideal fixed-pose optimum. The **inner16-facet polygon** gives an exact-Coulomb-feasible force witness and an upper bound on that same ideal optimum. Both include unrestricted ideal contact-force control and remain optimistic compared with real hardware/control. The outer polygon can over-admit friction by up to1.96%; this is why both bounds and exact witness friction ratios are reported.

Vertical-only tripod demand above1.6N·m means that particular fixed-pose, vertical-force-only model cannot meet the torque bound. It does **not** prove dynamic impossibility, reject a different body posture, or rule out an accelerating/turning tripod gait. Conversely, a1.194N·m optimized allocation does not establish that the current PD controller can produce those forces without slip, saturation or body motion.

## Missing data before a faster physical policy trial

1. Measured full-C load transfer and normal/tangential forces when unloading balanced pairs or a tripod, with requested/applied torque captured over physics substeps. The existing reference's passive PD load distribution can differ considerably from the ideal allocation.
2. Actual body pose and joint deflection throughout stance/swing. The recent RF clearance failure showed that a small pose difference can consume roughly3mm of the planned lift; this study holds the nominal pose fixed.
3. Swept joint limits, foot/body collisions, swing inertia, acceleration and stop feasibility at the desired cadence and body-twist envelope. Gravity-only static reserve must also pay for these dynamic demands.
4. Ground friction/compliance, controller force response, motor thermal/electrical limits and the final sensor/payload mass/COM. Currentμvalues are declared scenarios, not measurements; no electrical energy or thermal claim is made.

A useful next discriminator is a short full-robot load-transfer screen progressing from six supports to a balanced four-support pattern, then a bounded tripod condition if measured torque/slip/body-pose margins permit. Preserve the current physical gates and compare at matched commands/views with immutable Benchmark1. Static support analysis alone cannot choose a PPO architecture or establish all-direction walking quality.

## Artifacts and checks

`report.json` contains all336LPsolutions (42patterns×4friction cases×2cone approximations), forces, joint torques, gravity terms, exact source identity and residuals. `summary.json`, `friction_thresholds.json` and `static_headroom.png` expose the comparisons. Three tests verify all18FKJacobian/gravity derivatives by finite differences, force/moment/torque readback and cone bracketing, and rejection of a same-side tripod. No frozen earlier evidence was edited.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/c_support_pattern_feasibility_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 tmp/c_support_pattern_feasibility_001/static_patterns.py
PYTHONDONTWRITEBYTECODE=1 python3 tmp/c_support_pattern_feasibility_001/summarize.py
```
