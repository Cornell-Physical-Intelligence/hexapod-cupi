# Campaign 006: scheduled motor targets

The full nominal validation **passed** on 5 September 2026 UTC: 32 robots,
1,000 standing + 2,400 driven control steps and all 54,400 physics substeps per
robot measured. All independent and grouped motor direction checks passed.
The refined solver comparison is a separate phase and was still running when
this evidence was first published. This nominal pass alone is not training or
hardware admission.

| Nominal metric | Result |
| --- | ---: |
| Maximum driven pin separation | 0.0819092 mm (limit 0.100 mm) |
| Maximum passive-coordinate residual | 0.000996530 rad (limit 0.005 rad) |
| Minimum driven foot support | 3 |
| Non-foot ground contacts / resets | 0 / 0 |
| Peak raw requested torque | 14.32394 N·m |
| Peak delivered torque | 5.5 N·m |
| Torque-envelope excess | 0 |
| Minimum motor burst headroom | 0.456704 |
| Settled peak torque | 0.668155 N·m |
| Settled mean plate height | 0.13570454 m |

The preceding v5 full campaign failed. This revision changes only active motor
position-target delivery: the same 50 Hz endpoints are delivered over sixteen
physics updates, with zero velocity feedforward. Gains, mass/inertia, collision
geometry, mechanism constraints, action limits and physical acceptance bounds
are unchanged. Raw requested and delivered torque are intentionally separate:
the bounded motor model clipped transient demands rather than supplying the
14.3 N·m request.

Source `9cd8d4c`, functional identity
`c53071afdf320f6a9d6f91de09ddc6102de74df6f45a203e3a8166a8686548ca`,
selected asset `mkii_fourbar_v5`, TGS 64/1, 1.25 ms physics and 50 Hz policy.

Spark campaign:
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/campaigns/fourbar-campaign-20260905T185810Z-c36097a7/`.
Full source archives remain beside the reports on Spark; compact primary
reports, logs, audits and source hash lists are retained here. The `admitted`
file is only the host resource barrier, not physical training admission.
