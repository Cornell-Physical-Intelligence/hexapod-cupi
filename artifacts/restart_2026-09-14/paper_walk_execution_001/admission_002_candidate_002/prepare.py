"""Prepare a reviewed candidate; root adoption remains a separate decision."""
from pathlib import Path
import copy
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
A = HERE.parent
REPO = A.parents[2]
REMOTE = "/home/orionh/HEXAPOD_runs/restart_20260914/paper_walk_001"
sys.path.insert(0, str(REPO))
from experiments.paper_walk.train import require_admission
from experiments.paper_walk.env_config import EnvConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    audit_path = A / "verification_batch128_001/audit.json"
    assert sha(audit_path) == "025dadefb47833a1884930a59cce72218def5c3c4c8106ddc7a9c678bc9cbf07"
    audit = read(audit_path)
    assert audit["status"] == "completed" and audit["all_checks_passed"] is True
    assert audit["inputs_unchanged"] and audit["exact_recomputed_report"]
    assert audit["replicas_verified"] == 128 and audit["controls"] == 1000
    assert audit["substeps"] == audit["contact_packets_verified"] == 8000
    assert audit["contact_patches_reclassified"] == 24479073
    assert audit["separation_and_floor_checks_passed"] and audit["native_metadata_and_lifecycle_verified"]
    assert audit["audit_script_sha256"] == sha(A / "verification_batch128_001/audit.py")
    hashes = {}
    result = A / "results_diagnostic_batch128_001"
    for path in sorted(result.rglob("*")):
        if path.is_file():
            relative = path.relative_to(result).as_posix()
            digest = sha(path)
            assert audit["input_sha256"][REMOTE + "/diagnostic_batch128_001/" + relative] == digest
            hashes[relative] = digest
    assert len(hashes) == 14
    report_path = result / "standing/standing/standing_report.json"
    state_path = result / "standing/state.json"
    report, state = read(report_path), read(state_path)
    assert report == audit["recomputed_report"] and report["all_pass"]
    assert [r["env"] for r in report["replicas"]] == list(range(128))
    assert all(r["pass"] and r["quiet"]["pass"] and not r["failed_physical_bounds"] for r in report["replicas"])
    assert read(result / "standing/identity.json") == state["identity"]
    assert state["status"] == "completed" and state["mode"] == "diagnostic" and state["errors"] == []
    cleanup = read(result / "cleanup.json")
    assert cleanup["cleanup_checked"] and cleanup["inspections"] == [{"absent": True}]
    assert cleanup["resources"] == {"cuda_processes": "", "active_containers": ""}
    assert read(result / "standing/native/native_errors.json") == []
    drafts = {}
    for label, template in (("host", "admission_host.json"), ("container", "admission.json")):
        value = read(A / "admission_001" / template)
        value["num_envs"] = 128
        value["batch"]["report_sha256"] = sha(report_path)
        value["batch"]["state_sha256"] = sha(state_path)
        if label == "host":
            for key in ("report_path", "state_path"):
                value["batch"][key] = value["batch"][key].replace("diagnostic_batch32_001", "diagnostic_batch128_001")
        for key in ("physics_source_files", "physics_config", "model_sha256", "usd_sha256", "prior_metadata_sha256", "geometry_sha256", "geometry_extrema_sha256"):
            assert value[key] == state["identity"][key]
        for name, digest in value["physics_source_files"].items():
            assert sha(REPO / "experiments/paper_walk" / name) == digest
        drafts[label] = value
        write("admission_host.draft.json" if label == "host" else "admission.draft.json", value)
    local = copy.deepcopy(drafts["host"])
    for kind, root in (("one", A / "results_one_002"), ("batch", result)):
        local[kind]["state_path"] = str(root / "standing/state.json")
        local[kind]["report_path"] = str(root / "standing/standing/standing_report.json")
        for stem in ("state", "report"):
            assert sha(local[kind][stem + "_path"]) == local[kind][stem + "_sha256"]
    local_path = HERE / "admission_local_for_validation.json"
    write(local_path.name, local)
    for count in (1, 128):
        assert require_admission(local_path, state["identity"], EnvConfig(num_envs=count)) == local
    try:
        require_admission(local_path, state["identity"], EnvConfig(num_envs=32))
    except ValueError as error:
        assert str(error) == "Training replica layout lacks matching standing admission"
    else:
        raise AssertionError("Mismatched batch accepted")
    write("verification.json", {
        "schema": "canonical_paper_walk_standing_admission_candidate_review_v1",
        "status": "prepared_for_root_adoption", "root_adoption_recorded": False,
        "independent_audit_path": str(audit_path), "independent_audit_sha256": sha(audit_path),
        "audit_script_sha256": audit["audit_script_sha256"], "compact_native_files_sha256": hashes,
        "all_compact_bytes_match_audited_remote_inputs": True,
        "exact_recomputed_report": True, "all_128_original_gates_pass": True,
        "one002_reused_unchanged": drafts["host"]["one"],
        "batch128": drafts["host"]["batch"],
        "local_require_admission_passed_num_envs": [1, 128], "mismatched_num_envs32_rejected": True,
        "validator_source_sha256": sha(REPO / "experiments/paper_walk/train.py"),
        "physics_files_unchanged": True,
        "worst_joint_velocity_rms_rad_s": max(r["quiet"]["max_joint_velocity_rms_rad_s"] for r in report["replicas"]),
        "max_applied_torque_nm": max(r["physical"]["max_applied_all_substeps_nm"] for r in report["replicas"]),
        "minimum_sampled_sphere_gap": audit["minimum_sampled_sphere_gap"],
        "minimum_linear_swept_sphere_gap": audit["minimum_linear_swept_sphere_gap"],
        "minimum_floor_envelope_clearance": audit["minimum_floor_envelope_clearance"],
        "full_raw_local": False, "full_raw_retained_remote": REMOTE + "/diagnostic_batch128_001",
        "limitations": audit["limitations"], "stage2_complete": False,
        "physical_admission_created": False,
    })
    write("ROOT_HANDOFF.json", {
        "status": "draft_only_pending_root_adoption",
        "adopt_by_copying_without_content_changes": {
            "admission_host.draft.json": "admission_002/admission_host.json",
            "admission.draft.json": "admission_002/admission.json",
        },
        "new_binding_paths": {
            "standing_admission_host": REMOTE + "/admission_002/admission_host.json",
            "standing_admission_container": REMOTE + "/admission_002/admission.json",
            "standing_one": REMOTE + "/diagnostic_one_002/standing",
            "standing_batch": REMOTE + "/diagnostic_batch128_001/standing",
        },
        "training_num_envs": 128,
        "required_before_native": [
            "Root records adoption of completed audit and compact review; preserve admission001 and candidate001.",
            "Freeze fresh training source and confirm physics_source_files equal this admission.",
            "Choose a fresh128 learner configuration or exact compatible128 checkpoint; ordinary32 checkpoint resume is incompatible.",
            "Hash actual adopted host/container admission bytes into the new launch binding; mount one002 and batch128 read-only.",
            "Run exact host-translated stdlib CPU preflight and resource guard against the new binding.",
        ],
        "stage2_complete": False,
    })


if __name__ == "__main__":
    main()
