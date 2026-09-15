"""One-use local preparation of source016's unchanged 128-replica diagnostic.

No SSH, container, native runtime, reservation or admission operations occur.
The source016 train entry is executed only through its stdlib CPU preflight.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parent
REPO = EXECUTION.parents[2]
REMOTE = "/home/orionh/HEXAPOD_runs/restart_20260914/paper_walk_001"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (HERE/name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def main():
    source = EXECUTION/"source_016"
    freeze = json.loads((source/"FREEZE_SHA256.json").read_text())
    assert {str(p.relative_to(source)):sha(p) for p in source.rglob("*")
            if p.is_file() and p.name != "FREEZE_SHA256.json"} == freeze
    assert not any(p.is_symlink() for p in source.rglob("*"))
    parent_path = EXECUTION/"bindings/batch32_001.json"
    parent = json.loads(parent_path.read_text())
    binding = json.loads(json.dumps(parent))
    binding.update(root_review_complete=False, source=REMOTE+"/source_016",
        source_freeze_sha256=sha(source/"FREEZE_SHA256.json"), output=REMOTE+"/diagnostic_batch128_001")
    argv = binding["command_args"]
    argv[argv.index("--num-envs")+1] = "128"
    argv[argv.index("--source-freeze-sha256")+1] = binding["source_freeze_sha256"]
    pairs = {
        binding["asset"]: REPO/"artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected",
        binding["prior"]: REPO/"artifacts/restart_2026-09-14/paper_tripod_prior_001",
        binding["geometry_source"]: REPO/"artifacts/restart_2026-09-14/standing_translation_preparation_001/source",
        REMOTE+"/diagnostic_one_002": EXECUTION/"results_one_002",
    }
    mapped = {}
    for remote, expected in binding["input_files"].items():
        roots = [(prefix, local) for prefix,local in pairs.items() if remote.startswith(prefix+"/")]
        assert len(roots) == 1, remote
        prefix, local = roots[0]
        path = local/remote[len(prefix)+1:]
        actual = sha(path)
        assert actual == expected, str(path)
        mapped[remote] = {"local_path":str(path), "sha256":actual}
    container_to_host = {"/asset":binding["asset"], "/prior":binding["prior"],
                         "/geometry_source":binding["geometry_source"]}
    def translate(arg, paths):
        for prefix, path in paths.items():
            if arg == prefix or arg.startswith(prefix+"/"):
                return str(path)+arg[len(prefix):]
        return arg
    host_argv = ["python3","-B","-S",binding["source"]+"/train.py","--preflight-only",
        "--output",binding["output"]+"/standing",*[translate(arg,container_to_host) for arg in argv]]
    container_to_local = {prefix:pairs[host] for prefix,host in container_to_host.items()}
    local_output = HERE/"UNCREATED_native_output"
    local_argv = [sys.executable,"-B","-S",str(source/"train.py"),"--preflight-only",
        "--output",str(local_output),*[translate(arg,container_to_local) for arg in argv]]
    completed = subprocess.run(local_argv, cwd=REPO, text=True, capture_output=True, check=True)
    assert not local_output.exists()
    identity = json.loads(completed.stdout)
    one = json.loads((EXECUTION/"results_one_002/standing/state.json").read_text())
    old = json.loads((EXECUTION/"admission_001/admission_host.json").read_text())
    for key in ("physics_source_files","physics_config","prior_metadata_sha256",
                "model_sha256","usd_sha256","geometry_sha256","geometry_extrema_sha256"):
        assert identity[key] == one["identity"][key] == old[key], key
    assert identity["config"]["num_envs"] == 128
    assert identity["config"]["episode_seconds"] == 60.
    assert identity["config"]["render"] is False
    assert one["status"] == "completed" and one["standing_gate_pass"] is True
    assert binding["input_files"] == parent["input_files"]
    for name in ("env.py","env_config.py"):
        assert sha(source/name) == sha(EXECUTION/"source_002"/name)
    approved = json.loads(json.dumps(binding)); approved["root_review_complete"] = True
    approved_sha = hashlib.sha256(encode(approved)).hexdigest()
    side = math.ceil(math.sqrt(128))
    origins = [[i%side*2.,i//side*2.,0.] for i in range(128)]
    write("binding.draft.json",binding)
    write("input_translation.json",mapped)
    write("cpu_preflight.json",{"schema":"local_translated_native_entry_preflight_v1",
        "argv":local_argv,"returncode":completed.returncode,"stderr":completed.stderr,
        "identity":identity,"output_directory_created":False,"native_started":False,
        "scope":"Exact hash-bound local counterparts; live host paths/resources are not checked here."})
    write("root_dispatch.json",{"root_review_pending":True,"draft_binding_sha256":sha(HERE/"binding.draft.json"),
        "reviewed_binding_change":{"root_review_complete":True},
        "reviewed_binding_sha256_if_approved":approved_sha,
        "reviewed_binding_remote_path":REMOTE+"/bindings/batch128_001.json",
        "source_already_existing_remote":binding["source"],
        "host_translated_cpu_preflight_argv":host_argv,
        "guard_preflight_argv":["python3","-B","-S",binding["source"]+"/launch_spark.py",
            "--bindings",REMOTE+"/bindings/batch128_001.json","--bindings-sha256",approved_sha,"--preflight-only"],
        "native_dispatch_argv":["python3","-B","-S",binding["source"]+"/launch_spark.py",
            "--bindings",REMOTE+"/bindings/batch128_001.json","--bindings-sha256",approved_sha],
        "source_or_guard_edits_required":False,
        "preconditions":["Root reviews binding and writes reviewed successor without changing draft.",
            "Previous allocation cleanup, current reservation, both GPU locks, empty GPU and no producer descendants rechecked.",
            "Remote source016 complete tree and all15 input hashes rechecked by guard; output does not exist.",
            "Run host-translated CPU preflight then live guard preflight, preserving both receipts.",
            "Root owns bounded dispatch, stop/cleanup and post-exit reservation/resource verification."],
        "stage2_complete":False,"physical_admission":False})
    write("PREPARATION.json",{"schema":"canonical_batch128_standing_preparation_v1",
        "prepared_at_utc":datetime.now(timezone.utc).isoformat(),"status":"prepared_not_native_admitted",
        "parent_binding_path":str(parent_path),"parent_binding_sha256":sha(parent_path),
        "source_path":str(source),"source_freeze_sha256":binding["source_freeze_sha256"],
        "draft_binding_sha256":sha(HERE/"binding.draft.json"),
        "unchanged_physics_source_files":identity["physics_source_files"],
        "unchanged_physics_config":identity["physics_config"],"one002_reusable_identity_verified":True,
        "local_cpu_preflight_passed":True,"live_host_preflight_performed":False,"native_started":False,
        "controlled_recipe":{"replicas":128,"controls":1000,"substeps_per_control":8,"physics_steps":8000,
            "control_dt_s":.02,"physics_dt_s":.0025,"duration_s":20.,"settle_controls":200,
            "quiet_controls":800,"quiet_duration_s":16.,"episode_timeout_s":60.,
            "actions":"Zero offsets; hold declared [0,-0.30,0.40] per leg through existing limiter/servo.",
            "resets_inside_trial":0,"native_video":False,"training":False},
        "expected_layout":{"side":side,"spacing_m":2.,"origins":origins,
            "root_paths":[f"/Robot_{i:03d}" for i in range(128)],"articulations":128,
            "rigid_bodies":2432,"dofs":2304,"sdf_shapes":19584,"contact_sensor_rows":2432,
            "contact_filters":1,"max_contact_data_count":131072},
        "binding_differences_from_batch32":["root_review_complete pending instead of historical true",
            "source002 becomes existing frozen source016 and bound source hash",
            "fresh diagnostic_batch128_001 output","--num-envs128"],
        "admission002":"Not created. Requires complete authentic all128 gates, independent raw/scorer/clearance review and root adoption.",
        "historical_admission001_unchanged":True,"physical_admission":False,"stage2_complete":False})
    print(json.dumps({"preparation":str(HERE),"source_freeze_sha256":binding["source_freeze_sha256"],
        "draft_binding_sha256":sha(HERE/"binding.draft.json"),"reviewed_binding_sha256_if_approved":approved_sha,
        "local_cpu_preflight_passed":True,"native_started":False},indent=2))


if __name__ == "__main__":
    main()
