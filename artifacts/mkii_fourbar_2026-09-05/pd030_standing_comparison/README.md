# Kd 0.30 standing comparison: solver consistency still fails

Both short standing checks completed 32 environments × 600 control steps
(9,600 physics substeps per robot) without individual physical gate errors.
They do **not** constitute full validation or training admission: neither
includes the required 1,000 standing + 2,400 driven steps.

The matched standing comparison still fails the unchanged torque convergence
bound. The named controller experiment reduces the previously observed spikes
but does not resolve the solver sensitivity. No full campaign or PPO was
launched on the strength of these reports.

| Settled measurement | Nominal 64/1 | Refined 128/1 |
|---|---:|---:|
| Peak delivered torque (N·m) | 0.6656231880 | 1.9252085686 |
| Mean plate height (m) | 0.1357025828 | 0.1357222823 |
| Maximum pin gap (µm) | 1.5090691 | 14.8984191 |
| Minimum loaded feet | 6 | 4 |
| Minimum motor burst headroom | 0.5007101893 | 0.4314651787 |
| Non-foot contacts / episode resets | 0 / 0 | 0 / 0 |

The [derived comparison](comparison.json) records identical runtime settings
apart from solver position iterations. Both reports use the same 2.4–12.0 s
settled window (physics samples 1,920 through 9,599), zero commands, zero reset
jitter and seed 0. The torque difference is **1.2595853806 N·m**, against
`max(0.05, 0.05 × refined peak)` = **0.0962604284 N·m**. The 19.6995 µm
mean-height difference is within its separate 1 mm bound.

Source `a3081dd` selects `mkii_pd_damping_030_v1`: Kp 30 and Kd 0.30, with the
same v5 physical asset, 800 Hz dynamics, sixteen-step target ramp, provisional
48 V motor envelope and all prior physical gates. Functional identity is
`6d81d3117af6d39d703b68ecbeb6a410806a0f6931d0f059863e4081769e25c7`.
Runtime readback confirms actual damping 0.30 and stiffness 30.0. The base RS05
JSON/default profile remains 30 / 0.60.

Primary reports:

- [Nominal](nominal/hexapod-fourbar-validate-20260905T195746Z-14ead46a/report.json).
- [Refined](refined/hexapod-fourbar-validate-20260905T194825Z-e5e2e098/report.json),
  SHA-256 `f7f16751bd9f3514245a050af0a9a09662a40e25d5885a6fb99be55df2ee7fa6`.

Both supervisors report clean exact-container removal, exit 0 and unchanged
source identity at finish. Their reports, logs, CPU audits, source hash lists
and launch records are preserved here; full source archives remain on Spark.
The next investigation tests the observed relationship between simulation
placement and standing noise. That relationship does not yet establish its
cause or a successful remedy.
