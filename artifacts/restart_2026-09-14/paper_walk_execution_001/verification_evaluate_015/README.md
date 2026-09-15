# Independent native evaluation015 verification

`RESULT.json` completes the independent audit of all three actual checkpoint320
trials: 24,400 native physics steps and369,749 saved contact patches. Every
original gate verdict is reproduced. All three trials fail: forward tracking,
quiet six-toe support and stop six-toe support, respectively. No full Stage2
qualification is granted. All96 required cases and ten unselected learning probes
remain missing.

The bound checkpoint is actual train008 **update320**, SHA-256
`7a694caeab00bc6c23a7724b99c88e6de8cfdbfd88e60d97288df4440c8f0a2a`.
`BINDING_REVIEW.json` verifies frozen source018, the exact19-body/18-joint/153-shape
robot, unchanged physics relative to014, native recipe/readback/error records,
contact completeness and owned-container cleanup. All71 transferred files and
all audit input hashes remain verified. The source018 freeze is
`0baf8869effbee16253927363d41e315b43be27444493ac75e648801c7923c3a`.

## Complete native physical outcome

The acquisitions retain1,000/1,000/1,050 controls and8,000/8,000/8,400 physics steps.
Every float32 PD request, speed-dependent effort ceiling and actual native torque
input is recomputed. Pre/post joint continuity and every control endpoint match.
All369,749 contact patches are transformed through the recorded body/shape frames
and reclassified, with force aggregation and the1N toe-support predicate exactly
reproduced. The full native sequences/counters and raw-file hashes are verified.

There are no joint-position or SDK speed violations, terminations, truncations,
internal resets, active nonfoot events or applied-torque cap breaches. Maximum
SDK joint speeds are21.8409,11.2020 and21.8409 rad/s, below50.2655 rad/s. Maximum
applied torque is1.600000024Nm, within the unchanged1.60001Nm numerical allowance.
Whole-trial requested torque peaks are3.05243,2.06047 and3.05243Nm; the actual
applied torque remains capped. No requested-saturation gate fails.

Recorded non-toe clearance and plate height pass their original bounds. The
artifact checks these saved channels rather than independently transforming
every mesh vertex again. This clean physical acquisition does not establish
successful commanded movement or acceptable six-foot standing.

## Fixed-window comparison with014

Forward uses2s settling then18s measurement. Quiet uses4s settling then16s
measurement. Stop commands forward for8s, then zero, followed by the original2s
settling allowance and11s quiet measurement. No window or gate is changed.

| Metric |014 |015 |
|---|---:|---:|
| Forward signed mean COM velocity, commanded+0.05m/s |+0.00000831 |−0.000799517 |
| Forward planar error, limit0.025m/s |0.0499917 |0.0515597 |
| Quiet joint RMS, limit0.03rad/s |0.0113285 |0.00616294 |
| Quiet joint range, limit0.02rad |0.000271439 |0.00168452 |
| Quiet target-step p95, limit0.002rad/20ms |0.000111312 |0.000144395 |
| Quiet missing-six-toe count, limit0 /6,400samples |0 |6,400 |
| Stop joint RMS, limit0.03rad/s |0.00661340 |0.0123926 |
| Stop joint range, limit0.02rad |0.00582314 |0.000898659 |
| Stop target-step p95, limit0.002rad/20ms |0.0000922769 |0.000259954 |
| Stop missing-six-toe count, limit0 /4,400samples |0 |4,400 |

**The quiet/stop result is fixed three-foot support, not complete loss of support.**
LM,RF andRR support the robot at every scored quiet and stop substep. LF,LR andRM
are unsupported throughout both windows. Exactly three toes support the robot
in every one of those10,800 scored substeps; there are no scored zero-toe samples.
Their minimum supporting force norms are approximately28.46,22.27,22.12N in quiet
and28.57,22.47,21.56N after stop. This is not an alternating tripod gait.

The forward scored window also lacks six-toe support in all7,200 samples,
predominantly the same tripod. Moving feet are allowed in that profile; the
original failed gate remains planar tracking alone. A quiet-recovery or stop
success is not claimed: both fail support, and preceding forward walking never
tracks the command.

Mean root height increases from0.068424 to0.106558m in the quiet window and from
0.069932 to0.108183m after stop. Fixed-window raw action clipping fractions are
43.20% forward and33.33% in both quiet windows. These are recorded posture/action
changes, not proof of the learning cause. The completed training and its bounded
update metric do not guarantee retention of the previous quiet-support behavior.

## Arithmetic and source-version limits

The verifier inherits the explicitly documented014 compatibility treatment.
Native float32 percentile rank/interpolation is reproduced without changing
maintained source or gates. Local NumPy2.5.2 differs slightly for descriptive
vertical-velocity percentiles and the quiet target percentile; the latter differs
2.91e−11rad before matching the native interpolation. Stop target p95 is already
identical. Both local and native values retain the original gate decisions.

The quiet heading metric is not bit-exact across runtimes: native
0.0151698496193 degrees, local float32 0.0151766799390 degrees and independent
float64 `math.atan2`0.0151720930359 degrees. Their native/local difference is
6.83e−6degrees. All three independently pass the unchanged2-degree gate; no
compatibility tolerance substitutes for that gate and no native value is replaced.
Every other score matches after the declared percentile arithmetic. This015 CPU
audit completes on its first run; earlier013/014 exact-comparison failures remain
preserved in their original artifacts.

The evolving maintained startup orchestrator is not imported here. The actual
source018 acquisition orchestrator is hash-verified as data; the invoked scorer,
servo, rotation and analyzer helpers match their frozen018 hashes. This artifact
modifies no maintained source, raw trace, registry, policy or requirement. Root's
analysis and video review remain separate. `SHA256.json` excludes itself.
