from pathlib import Path
import hashlib,json,datetime
p=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
old=p.read_bytes();sha=lambda v:hashlib.sha256(v).hexdigest()
assert sha(old)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
note='''# Current C-study GPU coordination — 10 September 2026 UTC

The user resumed the C-length study on 9 September, explicitly prioritized its Stage 2 omnidirectional PPO and subsequent Stage 3 terrain work, and authorized pausing Forecasting-Pipeline GPU work and deferring it until later. This current instruction supersedes the historical 7 September physical-model pause below for the C study only. The physical four-bar campaign remains separately qualified. Preserve all historical records and outputs.

HEXAPOD is preparing the exact direct315 CAPS smoke at `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_smoke_001`, then independently admitted curriculum and CAPS pilots. Root is deferring only the identified `stormscope-halo104-1010-20260910.service`, preserving its completed/partial outputs before any eventual recomputation and arranging a bounded restart fallback. The ordinary forecast timers retain their per-allocation pause/restore controls.

Do not start a competing GPU producer during these actual HEXAPOD jobs. There is no long-lived exclusive GPU reservation or MPS quota. Each job still holds the existing weather/project locks, checks actual competing processes and available memory, and owns only its exact containers. Other CPU work may continue with headroom. Record any new explicit sharing request here so the next bounded allocation can yield. Output directories and process/unit receipts determine whether a job is actually running; preparation does not mean training started.

The earlier physical `HEXAPOD_SHARE_STATUS` and dated pause records below remain historical to their identified lineage. This note does not authorize changing any checkpoint, physical acceptance gate, unrelated workload or global GPU setting.

---

'''
new=note.encode()+old
q=p.with_name(p.name+'.c_study.tmp');assert not q.exists();q.write_bytes(new);q.replace(p)
print(json.dumps({'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(p),'previous_sha256':sha(old),'new_sha256':sha(new),'historical_bytes_preserved':new.endswith(old),'GPU_or_service_changed':False},indent=2))
