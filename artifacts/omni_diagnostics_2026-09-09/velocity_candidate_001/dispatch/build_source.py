"""Create one immutable candidate source from published code and admitted study inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_COMMIT = "bf7a5e84294d0e80394c666f1d3d8581c7b96bfa"
ASSET_SOURCE = ROOT / "tmp/terrain_robot_smoke_003_preparation/source/robot/hexapod_mkii_length_study"
ASSET_HASHES = ROOT / "artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_003/run/inputs/study_before_flat.sha256.json"
BASE_PLAN_SHA = "6a234f2b1ffd4f30ba4470b5ebb806cb2964fa0cdc85f6e4433976749c98af2a"
CANDIDATE_FILES = ("velocity_action.py", "candidate_env.py", "candidate_runner.py",
                   "candidate_diagnostics.py", "candidate_asset_audit.py", "train_velocity_candidate.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Fresh source directory required")
    candidate = args.candidate.resolve()
    frozen = json.loads(args.manifest.read_text())
    frozen = frozen.get("files", frozen)
    for name in CANDIDATE_FILES:
        if frozen.get(name) != digest(candidate / name):
            raise ValueError(f"Candidate file absent or changed since review: {name}")
    assets = json.loads(ASSET_HASHES.read_text())
    actual = {str(p.relative_to(ASSET_SOURCE)): digest(p) for p in ASSET_SOURCE.rglob("*") if p.is_file()}
    if actual != assets or digest(ASSET_SOURCE / "training_plan.json") != BASE_PLAN_SHA:
        raise ValueError("Admitted study inputs changed")
    out.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", "--format=tar", BASE_COMMIT, "--", "tools", "isaaclab",
                        "experiments/c_length_study/runtime", "packages"], cwd=ROOT, stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tree:
            tree.extractall(out, filter="data")
    package = out / "robot/hexapod_mkii_length_study"
    shutil.copytree(ASSET_SOURCE, package)
    for name in CANDIDATE_FILES:
        shutil.copyfile(candidate / name, out / "tools" / name)
    shutil.copyfile(HERE / "launch_velocity_probe_spark.py", out / "tools/launch_velocity_probe_spark.py")
    plan = json.loads((package / "training_plan.json").read_text())
    plan.update(training_num_envs=1024, evaluation_num_envs=48, training_iterations=50)
    omni = plan["omni"]
    omni.update(architecture="c_serial_omni_target_velocity_v1", history_frames=5,
        actor_frame_width=99, actor_width=495, critic_width=498,
        velocity_candidate={"profile": "formal_004", "max_acceleration_rad_s2": 8.0},
        diagnostics={"duration_s": 12., "settle_s": 2., "seed": 7057,
                     "trace_envs_per_scenario": 1, "controller": "policy"})
    omni["actor_fields"] += ["executable_target_offset", "executable_target_velocity_normalized"]
    omni["overrides"].update(target_slew_rad_per_20ms=.04, observation_noise_scale=1.,
                             target_filter_time_constant_s=0.)
    (package / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    provenance = dict(base_code_commit=BASE_COMMIT, base_plan_sha256=BASE_PLAN_SHA,
        study_asset_snapshot_sha256=digest(ASSET_HASHES), study_asset_files_verified=len(assets),
        candidate_manifest_sha256=digest(args.manifest), candidate_files={n: frozen[n] for n in CANDIDATE_FILES},
        changed_asset_files=["training_plan.json"], existing_checkpoint_loaded=False,
        dispatch_scope="Fresh standing, exploration calibration and at most two stand-only schema updates; no automatic walking training")
    (out / "source_origin.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "campaign_source_hashes.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(source=str(out), files=len(records),
                         source_manifest_sha256=digest(out / "campaign_source_hashes.json")), indent=2))


if __name__ == "__main__":
    main()
