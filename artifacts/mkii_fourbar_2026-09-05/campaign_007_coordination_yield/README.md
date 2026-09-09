# Campaign 007: coordination yield interrupted nominal validation

Campaign 007 stopped because the guarded launcher detected changed bytes in
the shared coordination file. The one-environment probe passed, but the
32-environment nominal validation has **no final primary report**. The
campaign did not reach refined validation, scratch PPO or the full PPO run.
This evidence does not establish a full nominal physics pass or failure.

The supervisor records `Coordination changed; owned job yielded compute`,
final container exit **137**, supervisor exit **1**, unchanged source at
finish and removal of the **exact owned container**. The campaign then
records `nominal: supervisor exited 1; no retry or subsequent phase`.

## Last demonstrated progress

The genuine probe report passes **1 environment × 100 controls**, with all
1,600 physics substeps observed. Its settled peak raw/applied torque is
0.701226115 N·m, maximum C-pin gap 0.674391 µm and minimum support six feet.

The nominal log's final progress record is **standing step 600/1000**,
timestamp **2026-09-05T22:18:24.289578780Z**. That demonstrates at least 600
completed controls and 9,600 physics substeps per environment. The exact
stop step is unknown: progress was printed only every 100 controls, and no
final report was written. No driven progress appears in the captured log.

| Nominal last logged settled metric | Value |
| --- | ---: |
| Peak raw / applied torque | 0.759236634 N·m |
| Mean / minimum plate height | 0.135700507 / 0.135664761 m |
| Maximum C-pin gap | 1.720294 µm |
| Maximum passive position residual | 19.073486 µrad |
| Minimum loaded feet | 5 |
| Non-foot contacts / invalid samples | 0 / 0 |
| Minimum non-foot ground clearance | 0.025229380 m |
| Peak / RMS passive velocity residual | 0.692813993 / 0.002539213 rad/s |
| Peak / RMS C-pin relative speed | 0.012707432 / 0.000110809 m/s |

These are cumulative settled metrics for **controls 201–600**, 6,400 physics
substeps per environment, simulation time **4.0–12.0 s**. This differs from
the earlier 600-control comparison's 2.4–12.0 s window: this campaign
requested 1,000 standing controls, so its first 200 controls were startup.
The startup maximum raw/applied torque through control 200 was 1.559164524
N·m. All captured progress records remain in `summary.json` and the original
timestamped container log. They are partial observations, not a substitute
for the missing final validator report.

## Stop chronology

All times below are UTC on **2026-09-05**. Filesystem modification times are
identified explicitly; they are not invented event timestamps.

| Time | Evidence |
| --- | --- |
| 22:09:42 | Original launch record and preflight |
| 22:09:44 | Campaign resource gate passed |
| 22:11:19.409845 | Probe supervisor completion record modification time |
| 22:11:21 | First nominal resource gate passed |
| 22:18:24.289578780 | Last flushed nominal progress: standing 600/1000 |
| 22:18:51 | Last successful nominal runtime resource gate |
| 22:18:56.719657 | `stop_requested` modification time |
| 22:19:22.900430 | Final nominal supervisor record modification time |
| 22:19:22.910692 | Campaign failure record modification time |

The stop marker says `Shared coordination file changed; checkpoint and
pause.` The frozen launcher, however, immediately leaves the validation
loop on a coordination change; its longer checkpoint grace applies only to
training. Cleanup uses `docker stop --time 25` on the exact owned container,
preserves its timestamped log, and removes that exact ID. The preserved
source excerpt verifies this behavior against the campaign's launcher hash.
The resulting exit 137 is recorded after that bounded cleanup; this artifact
does not reinterpret it as a native physics crash.

The baseline coordination SHA recorded by the nominal supervisor is
`bc4ed67cd624840d82a4b083bd707d6b06d8f6865692ea220b31a42be4fefbe1`.
The supervisor does not record the editor identity, changed file contents or
reason for the edit. Those cannot be inferred from the hash-change event.
The shared file was not edited during this evidence audit.

## Identity and preserved files

Source is **`fd34f661bd672df9f68e6d8568548ad092ef32cf`**, functional identity
`1fcab03b2c9f810a931e3d65fd057312d6dfb3b92003ca30058ccd821ff965e6`.
Probe and nominal capture the same 308-file source hash list, SHA
`5f99a880d346b230ed4c94898bd4fc22764f4755c9d0f4bf5e99b9ccc1f211ff`.
Every functional-contract file matches that list, and both supervisors
verify unchanged source and exact cleanup.

The [probe primary report](probe/hexapod-fourbar-validate-20260905T220944Z-7e259721/report.json)
has SHA `b76cabe0003016fbafb4dd350c966a16fba6214e9db274c2a6b3c59e00aa057a`.
**No nominal `report.json` is created in this artifact.** Its absence is
recorded in both the original supervisor and the read-only remote inventory.
Original campaign/probe/nominal reports where present, logs, CPU audits,
source manifests, supervisors, stop marker, launch/staging records and
remote modification times are preserved. Large source archives remain on
Spark and were neither downloaded nor independently rehashed.

Remote campaign:

```text
/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1/campaigns/fourbar-campaign-20260905T220944Z-78b3e50d
```

## Reproduction

From the repository root:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/campaign_007_coordination_yield/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/campaign_007_coordination_yield \
  --out /tmp/hexapod_campaign007_interruption_new.json
```

The analyzer verifies downloaded bytes against the captured remote inventory,
source/cleanup identity, probe report identity, missing nominal report and
the populations of the logged startup/settled windows. `summary.json` is
explicitly derived interruption evidence. No production source, model,
gate, shared note, GPU workload or previous artifact was changed.
