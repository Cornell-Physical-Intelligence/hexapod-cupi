"""One-off, explicitly authorized checkpoint migration; never a runtime loader.

Frozen learners are imported only for artifact identity/inference proofs. No
environment is created, no fitting occurs, and no simulation state is claimed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
OLD_SCHEMA = "canonical_paper_walk_ppo_amp_v2"
NEW_SCHEMA = "canonical_paper_walk_ppo_amp_v3"
OLD_CHECKPOINT_SHA = "eb2847e9bce7db0b090debdbc2b0b3395679ff3e2b78f8ecf7407aaa4a8b2d1a"
OLD_LEARNER_SHA = "b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a"
ADDITIONS = {
    "rollback_kl_steps": True,
    "kl_backtrack_halvings": 3,
    "kl_backtrack_factor": .5,
    "freeze_actor_obs_normalizer": True,
    "zero_accepted_update_limit": 3,
}
NEW_COUNTER = "consecutive_zero_accepted_updates"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_source(directory, expected_freeze_sha):
    directory = Path(directory).resolve()
    manifest = directory / "FREEZE_SHA256.json"
    require(not manifest.is_symlink(), "Source freeze cannot be a symlink")
    require(sha(manifest) == expected_freeze_sha, "Source freeze digest differs")
    records = json.loads(manifest.read_text())
    require(isinstance(records, dict) and "learner.py" in records, "Malformed source freeze")
    members = list(directory.rglob("*"))
    require(not any(path.is_symlink() for path in members), "Frozen source contains a symlink")
    actual = {str(path.relative_to(directory)) for path in members if path.is_file()}
    require(actual == set(records) | {"FREEZE_SHA256.json"}, "Frozen source has missing or unlisted files")
    for name, digest in records.items():
        rel = Path(name)
        path = directory / rel
        require(not rel.is_absolute() and ".." not in rel.parts and not path.is_symlink(), "Unsafe source member")
        require(path.resolve().is_relative_to(directory), "Source member escaped freeze")
        require(sha(path) == digest, "Frozen source member differs: " + name)
    return directory / "learner.py"


def transform(payload, old, new, old_source_sha, new_source_sha):
    require(old.SCHEMA == OLD_SCHEMA and new.SCHEMA == NEW_SCHEMA, "Unexpected learner schemas")
    require(payload.get("schema") == OLD_SCHEMA, "Input checkpoint schema differs")
    require(payload.get("learner_sha256") == old_source_sha, "Input checkpoint source differs")
    require(payload.get("config", {}).get("target_kl") == .02, "Existing target KL must already be .02")
    require(payload.get("simulation_resume_supported") is False, "Unexpected simulation resume claim")
    require(payload.get("stage2_complete") is False and payload.get("physical_admission") is False, "Unexpected qualification claim")
    require(NEW_COUNTER not in payload["counters"], "Input already has successor counter")
    old_config = old.Config(**payload["config"])
    old_config.validate()
    require(exact(asdict(old_config), payload["config"]), "Old configuration has missing/defaulted fields")
    old_fields = {x.name for x in fields(old.Config)}
    new_fields = {x.name for x in fields(new.Config)}
    require(new_fields - old_fields == set(ADDITIONS) and old_fields <= new_fields, "Unexpected Config schema delta")
    config = dict(payload["config"])
    require(not (set(config) & set(ADDITIONS)), "Input already has successor options")
    config.update(ADDITIONS)
    new_config = new.Config(**config)
    new_config.validate()
    require(exact(asdict(new_config), config), "New configuration has missing/defaulted fields")
    migrated = copy.deepcopy(payload)
    migrated.update(schema=NEW_SCHEMA, learner_sha256=new_source_sha, config=config)
    migrated["counters"][NEW_COUNTER] = 0
    verify_preserved(payload, migrated)
    return migrated


def verify_preserved(before, after):
    require(before.keys() == after.keys(), "Checkpoint top-level fields changed")
    for key in before:
        if key not in {"schema", "learner_sha256", "config", "counters"}:
            require(exact(before[key], after[key]), "Preserved payload differs: " + key)
    require(set(after["config"]) == set(before["config"]) | set(ADDITIONS), "Config coverage differs")
    for key, value in before["config"].items():
        require(exact(value, after["config"][key]), "Existing Config changed: " + key)
    for key, value in ADDITIONS.items():
        require(exact(value, after["config"][key]), "Adopted successor Config differs: " + key)
    require(set(after["counters"]) == set(before["counters"]) | {NEW_COUNTER}, "Counter coverage differs")
    for key, value in before["counters"].items():
        require(exact(value, after["counters"][key]), "Existing counter changed: " + key)
    require(type(after["counters"][NEW_COUNTER]) is int and after["counters"][NEW_COUNTER] == 0, "New streak counter must explicitly start at zero")


def assert_restored(learner, payload):
    for name in ("model", "amp", "optimizer", "discriminator_optimizer"):
        require(exact(getattr(learner, name).state_dict(), payload[name]), "Strict restored state differs: " + name)
    require(all(exact(getattr(learner, k), v) for k, v in payload["counters"].items()), "Restored counters differ")
    require(exact(learner.last_metrics, payload["metrics"]), "Restored metrics differ")


def rejected_without_mutation(learner, path):
    before = {k: copy.deepcopy(getattr(learner, k).state_dict()) for k in ("model", "amp", "optimizer", "discriminator_optimizer")}
    counter_names = ["updates", "transitions", "optimizer_steps", "discriminator_steps", "episodes", "bc_steps"]
    if hasattr(learner, NEW_COUNTER):
        counter_names.append(NEW_COUNTER)
    counters = {k: getattr(learner, k) for k in counter_names}
    metrics = copy.deepcopy(learner.last_metrics)
    rng = learner._rng()
    try:
        learner.load(path)
    except ValueError as exc:
        reason = str(exc)
    else:
        raise ValueError("Cross-version checkpoint was unexpectedly accepted")
    require(all(exact(v, getattr(learner, k).state_dict()) for k, v in before.items()), "Rejected load mutated model/optimizer")
    require(all(exact(v, getattr(learner, k)) for k, v in counters.items()), "Rejected load mutated counters")
    require(exact(metrics, learner.last_metrics) and exact(rng, learner._rng()), "Rejected load mutated metrics/RNG")
    return reason


@torch.no_grad()
def probe_outputs(old, new, trace_path):
    with np.load(trace_path, allow_pickle=False) as data:
        arrays = {name: torch.from_numpy(data[name].copy()).reshape(-1, width) for name, width in (
            ("policy_observation", 231), ("critic_observation", 234), ("amp_state_before", 61), ("amp_state_after", 61))}
    require(all(bool(torch.isfinite(x).all()) for x in arrays.values()), "Nonfinite recorded probe")
    obs, critic, first, second = (arrays[k] for k in ("policy_observation", "critic_observation", "amp_state_before", "amp_state_after"))
    require(len(obs) > 0 and all(len(x) == len(obs) for x in arrays.values()), "Misaligned recorded probes")
    require(torch.equal(obs, critic[:, :231]), "Recorded critic prefix differs")
    outputs = {}
    for name, learner in (("old", old), ("new", new)):
        mean, velocity = learner.model.actor(obs)
        dist, _ = learner.model.distribution(obs)
        outputs[name] = {"actor": mean, "velocity": velocity, "critic": learner.model.value(critic),
                         "action_std": dist.stddev, "discriminator": learner.amp.discriminator(learner.amp.features(first, second)),
                         "style_reward": learner.amp.style_reward(first, second)}
    require(all(bool(torch.isfinite(v).all()) for o in outputs.values() for v in o.values()), "Nonfinite probe output")
    require(exact(outputs["old"], outputs["new"]), "Old/new recorded-input inference differs")
    return {"path": str(trace_path), "sha256": sha(trace_path), "rows": len(obs), "bitwise_equal_outputs": list(outputs["old"])}


def execute(binding_path):
    binding_digest = sha(binding_path)
    request = json.loads(Path(binding_path).read_text())
    require(request.get("root_authorized_execution") is True, "Root has not authorized final migration")
    require(request.get("schema") == "explicit_ppo_v2_to_v3_migration_binding_v1", "Unknown migration binding")
    required_inputs = {str(Path(__file__).resolve()), request["old_checkpoint"], request["prior"], *request["recorded_traces"]}
    require(required_inputs <= set(request["input_sha256"]), "Every executed or probed input must be hash-bound")
    require(request["recorded_traces"], "Real recorded observations are required")
    for path, digest in request["input_sha256"].items():
        require(sha(path) == digest, "Bound input changed: " + path)
    require(request["input_sha256"].get(request["old_checkpoint"]) == OLD_CHECKPOINT_SHA, "Wrong actual input checkpoint")
    old_path = verify_source(request["old_source"], request["old_freeze_sha256"])
    new_path = verify_source(request["new_source"], request["new_freeze_sha256"])
    require(sha(old_path) == OLD_LEARNER_SHA, "Wrong archived learner")
    require(sha(new_path) == request["new_learner_sha256"], "New learner digest differs")
    destination = Path(request["output_directory"]).resolve()
    require(destination.parent == OUT and destination.name == "run_001", "Output must be this artifact's fresh run_001")
    require(not destination.exists(), "Output already exists")
    require(not torch.cuda.is_available(), "This migration proof must execute on a CPU-only host")
    torch.set_num_threads(2)
    old = load_module(old_path, "explicit_migration_old")
    new = load_module(new_path, "explicit_migration_new")
    payload = torch.load(request["old_checkpoint"], map_location="cpu", weights_only=False)
    require(payload["prior_sha256"] == sha(request["prior"]), "Prior identity differs")
    migrated = transform(payload, old, new, sha(old_path), sha(new_path))
    destination.mkdir()
    checkpoint = destination / "checkpoint_update000200_migrated.pt"
    # Deliberately avoid learner.save: CPU save would replace recorded CUDA RNG.
    with checkpoint.open("xb") as stream:
        torch.save(migrated, stream)
    readback = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(exact(migrated, readback), "Serialized migration differs")
    verify_preserved(payload, readback)
    ambient_rng = old.PPOLearner._rng()
    try:
        before = old.PPOLearner(None, request["prior"], destination/"old_strict_loader", old.Config(**payload["config"]), device="cpu")
        before.load(request["old_checkpoint"])
        assert_restored(before, payload)
        old_cpu_rng = before._rng()
        require(all(exact(old_cpu_rng[k], payload["rng"][k]) for k in ("python", "numpy", "torch")), "Old CPU RNG restoration differs")
        after = new.PPOLearner(None, request["prior"], destination/"new_strict_loader", new.Config(**migrated["config"]), device="cpu")
        after.load(checkpoint)
        assert_restored(after, migrated)
        new_cpu_rng = after._rng()
        require(all(exact(new_cpu_rng[k], payload["rng"][k]) for k in ("python", "numpy", "torch")), "New CPU RNG restoration differs")
        cross = {"old_rejects_new": rejected_without_mutation(before, checkpoint), "new_rejects_old": rejected_without_mutation(after, request["old_checkpoint"])}
        probe_rng = after._rng()
        probes = [probe_outputs(before, after, path) for path in request["recorded_traces"]]
        require(exact(probe_rng, after._rng()), "Inference proof changed RNG")
    finally:
        old.PPOLearner._restore_rng(ambient_rng)
    require(exact(ambient_rng, old.PPOLearner._rng()), "Migration verification changed ambient RNG")
    for path, digest in request["input_sha256"].items():
        require(sha(path) == digest, "Input changed during migration: " + path)
    verify_source(request["old_source"], request["old_freeze_sha256"])
    verify_source(request["new_source"], request["new_freeze_sha256"])
    require(sha(binding_path) == binding_digest, "Migration binding changed during execution")
    report = {"schema": "explicit_ppo_checkpoint_migration_verification_v1", "completed": True,
              "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "binding_sha256": binding_digest, "torch_version": torch.__version__, "numpy_version": np.__version__,
              "input_checkpoint_sha256": sha(request["old_checkpoint"]), "output_checkpoint": str(checkpoint),
              "output_checkpoint_sha256": sha(checkpoint), "source_transition": {"old": sha(old_path), "new": sha(new_path)},
              "schema_transition": [OLD_SCHEMA, NEW_SCHEMA], "added_config": ADDITIONS,
              "existing_config_unchanged": True, "preserved_counters": payload["counters"], "new_counter": {NEW_COUNTER: 0},
              "preserved_payload_fields": [k for k in payload if k not in {"schema", "learner_sha256", "config", "counters"}],
              "model_amp_and_both_adam_states_exact": True, "all_serialized_rng_exact_including_cuda": True,
              "cpu_rng_restoration_exact": True, "ambient_rng_preserved": True, "strict_own_version_loads": True,
              "cross_version_rejections": cross, "recorded_input_probes": probes,
              "cuda_generator_runtime_restoration_tested": False,
              "limitation": "CPU artifact migration and exact inference only; preserved CUDA RNG bytes are not a GPU continuation test. PhysX state is absent; continuation requires a fresh native reset and matching admission.",
              "new_training_updates": 0, "simulation_resume_supported": False, "physical_admission": False, "stage2_complete": False}
    with (destination/"VERIFICATION.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.bindings), indent=2, allow_nan=False))
