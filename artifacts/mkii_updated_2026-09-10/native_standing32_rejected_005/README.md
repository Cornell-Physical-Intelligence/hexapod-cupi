# Canonical32 standing: completed acquisition, rejected admission

**All32 detailed robots completed1,000 control holds /8,000 physics steps. Standing admission failed:10/32 combined passes,22 support rejections and8 quiet-rate rejections.** No PPO training was admitted. Unlike the earlier operational failures, this allocation had no timeout or foreign-workload interruption: the native container exited0 after completing its measurement, and the host then correctly rejected the original standing report.

This is the detailed direct-drive7.466088235kg canonical model, source005's explicit32 position/0 velocity solver recipe, unchanged full geometry/materials/servo and original physical/quiet gates. The matching single-robot native6 pass remains separate historical evidence; it does not substitute for a32-robot pass. No gate or threshold was changed by this publication.

## Actual report

| Measurement | Saved result |
|---|---:|
| Combined / physical-only / quiet-only passes |10/32 /10/32 /24/32|
| Replicas with post-settle missing six-toe support |22|
| Missing-support samples per affected replica |1–8|
| Missing-support replica-substep samples summed across rows |73|
| Largest quiet rawSDK joint-rate RMS |0.1060787514rad/s; gate0.03|
| Largest quiet joint position range |0.0107462620rad; gate0.02|
| Largest quiet planar excursion |0.0001612219m; gate0.01|
| Peak requested/applied effort over all substeps |1.1257334948Nm|
| Maximum requested saturation fraction |0|
| Nonfoot-contact substeps |0|

All22 physical rejections name `six_toe_support`. Eight of those replicas additionally fail only `max_joint_velocity_rms_rad_s`: environments2,8,9,19,20,23,25,27. `PER_ENVIRONMENT.csv` retains all32 report rows. Seventy-three is a count of replica-substep support failures, not73 independently qualified flight/landing events. These aggregate report values do not establish a contact-loss cause or prove SDK bias caused it. RawSDK rates and measured angle changes remain separate evidence; no alternative rate metric replaces the original gate.

`SUMMARY.json` is reproducibly derived from the saved report by `summarize_report.py`. This is **report-level calculation, not local raw-trace replay**. The copied original report and native state preserve exact precision and failure labels.

## Completion and cleanup

Actual unit: `hexapod-canonical-native-standing32-005-20260910.service`, invocation `c5397bd5a8dd4640899ad3dec2c30f47`, under pause018. The job records completed acquisition, exit0 and exact container cleanup. The host campaign/unit records failure because `Standing physics/quiet rejected; acquisition is not admission`. Native state is completed with8,000 explicit steps, no native error events and `standing_pass=false`.

The audit is verified with no audit errors. Its `raw_acquisition_completed=true` and `standing_completed=false` are intentionally different: the latter is an admission result, not evidence of an unfinished acquisition. Both exact owned container name and ID were absent at terminal audit. All original input trees were verified unchanged at audit start/end. Authored solver attributes read32/0 after authoring, reset and completion on all32 roots; the auditor labels this USD attribute readback, not independent backend solver introspection.

The phase deadline was1,200s with90s AppReady, separate from quality gates. Native measurement wall time was599.213s, including initialization; the session's controlled portion was577.554s. The job completed in about606.073s, below its1,200s bound. There was no deadline or foreign-process rejection.

Persistent automation reservation,31 masks and the strong reconstruction entry/held queue lock passed reservation checks. Per-job restoration retained the persistent reservation and released no external schedulers; its timer list was empty. The earlier failed reservation probe and successful successor remain preserved under `root/` and the separate operations001/002 artifacts. They are not relabeled as native attempts. Saved checks do not establish an arbitrary privileged/manual-CUDA security partition.

## Curated bytes and remote-only evidence

The immutable remote inventory contains **39 raw payloads totaling5,019,294,627bytes**. Root fetched27 small payloads totaling3,517,992bytes; every copied byte was independently compared with the full terminal inventory. Twelve large payloads totaling5,015,776,635bytes remain remote: the full contact JSON4,526,633,896bytes, control trace55,284,390bytes and ten400Hz NPZ chunks. Exact SHA-256, byte size and remote absolute path for each are in `REMOTE_ONLY.json`. No5GB download or reconstruction was performed.

The raw output remains at `/home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_standing32_005`; the pause files are in `forecast_pause_018`. No claim of a self-contained full raw replay is made. Remote backup or trace availability is not re-probed by this portable verifier.

The package includes complete exact frozen preparation payloads: source109, host63, guard116 and auditor24, plus root's dispatch/setup/progress/audit-transfer snapshot. Source009 remains the pinned ownership supervisor; the host's explicit1,200s deadline adapter is recorded without claiming its runtime function is unchanged. The current robot physics comes from the canonical source005, not the old task. The asset, original source009 and earlier admissions are hash-bound in the audit and existing artifacts; their whole external trees are not duplicated here.

## Verification

```sh
python3 -S /absolute/path/to/native_standing32_rejected_005/verify_bundle.py
python3 -S /absolute/path/to/native_standing32_rejected_005/summarize_report.py --output /tmp/standing32_005_summary.json
```

The verifier checks every public payload and all four source manifests,27 copied raw files against all39 recorded raw entries, explicit12-file exclusions, source/asset/admission identities, actual completion-versus-rejection semantics, solver/readback/cleanup/reservation metadata and exact summary reproduction. It does not import native code, execute the stored operator/auditor, invent missing NPZ files, rerun the physical scorer or contact Spark.

Root owns the shared status/plan/site update and Git publication under `docs/PROJECT_SITE.md`. This package changes no runtime, acceptance gate, old artifact, GPU state or physical qualification claim.
