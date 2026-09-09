# Live body-pair collision controls — attempt 002

The filtered and deliberately unfiltered controls both completed on the Spark
in separate owned GPU containers. The measured **root-body pair** was isolated
with filtering and interacted when the diagnostic enabled reciprocal
cross-environment collision links. This is a successful diagnostic control pair;
the case, pair and supervisor reports keep their training-admission,
hardware-admission and general `pass` flags false. `pair_report.json` records `pair_control_success=true`.

Evidence is under `overlap-20260905T211506Z-1c8fc7b2/`. Each case ran two
co-origin v5 robots for **256 physics steps × 0.00125 s = 0.32 s**, using the
frozen `d863663` source and **128 position / 1 velocity** TGS iterations. This
records the executed recipe, independently of later solver candidates.

| Recomputed measurement | Filtered | Unfiltered negative control |
| --- | ---: | ---: |
| Samples | 256 | 256 |
| Samples with body-pair contact | 0 | 256 |
| Maximum native contact count, per direction/sample | 0 | 340 |
| Maximum body-pair force-vector magnitude | 0 N | 19,690.294921875 N |
| Ground support observed for each robot | yes / yes | yes / yes |
| Peak ground force per foot, by robot | 139.076 / 139.076 N | 430.080 / 1,066.513 N |

The negative control starts the two bodies in exact overlap; its large contact
force demonstrates an effective positive contact detector and is not a robot
load estimate. Neither native contact-count buffer reached its 512-contact
capacity. Both supervisors recorded owned GPU PIDs, exit status zero,
`diagnostic_complete`, unchanged input identities and `removed_exact_id` cleanup.

## Independent verification

The downloaded reports have identical source identity, fixture hash, v5 asset
bundle and dependencies, numerical recipe, initial roots and mimic-reference
records. Runtime manifests differ in exactly four fields, all inside
`resolved_collision_isolation`: mode, topology verification status,
cross-environment allowlinks and the environment allowlist. The negative-control
report records the two reciprocal group relationships added before physics
initialization; no other runtime difference appears.

Both trace hashes match their reports. Each NPZ contains finite `float64`
`values[256,50]`, exact `int64` `pair_contact_counts[256,2]`, matching column
names and scalar `physics_dt_s=0.00125`. The observed native dtype is
`torch.uint32` for both contact views. The archived fixture converts it to
`int64` before count operations; the integer array and mixed-trace count columns
match exactly. Recomputing count maxima, force norms and ground-support metrics
from the arrays reproduces the reports, retaining the native float32 precision
for the reported force norms.

All **24 composed USD mimic references** are unique and point from each robot's
six `tibia_pitch` and six `tibia_rod_pivot` joints to that same robot and leg's
`tibia_lever_pivot`. This verifies composed relationships; it does not directly
query the native solver's internal remapping.

## Scope and archived inputs

The two directional native contact views bind only
`/World/envs/env_0/Robot/Geometry/body` and the corresponding `env_1` body.
They do **not** enumerate all 31 links against all links, test 32 or more
overlapping environments, verify rendered/LiDAR sensor isolation, or establish
long-duration stability, locomotion quality or hardware readiness. Shared-ground
foot forces were observed separately for both robots.

The production source identity is
`a1eaf8e5411519c7b8fe70edff45acfc5e4f217c5cff8904c5cd0ca3e607b20f`;
the executed fixture SHA-256 is
`bd96b63f2b6cbfd7ff578aa574fe0035312ce77d44f99599edd8f7bf0c191b99`.
All 240 contract-file hashes and selected USD dependency hashes agree with the
preserved 1,229-file `source.SHA256SUMS`. The 236 MB production `source.tar.gz`
is not included in this compact download; its hash and remote run location are
recorded in `inputs.json` and launch metadata. The included `fixture.tar.gz`
and every archived fixture member were verified against their recorded hashes.
Pair-report input hashes also match the two preserved reports.

`SHA256SUMS` covers every file in this attempt directory except itself,
including this README, logs, traces, reports and the fixture archive. Verify
from this directory with `shasum -a 256 -c SHA256SUMS`. Earlier attempts and
production/fixture sources were left unchanged by this review.
