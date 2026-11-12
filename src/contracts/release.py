"""Identity fields required to load an existing locomotion checkpoint."""

COMPATIBILITY_KEYS = (
    "schema",
    "model_sha256",
    "usd_sha256",
    "physics_source_files",
    "physics_config",
    "stance_sha256",
    "geometry_sha256",
    "geometry_extrema_sha256",
    "adapter_sha256",
    "entry_sha256",
    "seed",
)


def runtime_changes(recorded, current):
    """Compare the full runtime identity, including added files and manifest metadata."""
    result = {name: [] for name in ("added", "removed", "modified", "metadata")}
    if recorded == current:
        return result
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        result["metadata"] = ["manifest"]
        return result
    before, after = recorded.get("files", {}), current.get("files", {})
    result["added"] = sorted(after.keys() - before.keys())
    result["removed"] = sorted(before.keys() - after.keys())
    result["modified"] = sorted(
        name for name in before.keys() & after.keys() if before[name] != after[name]
    )
    result["metadata"] = sorted(
        name
        for name in (recorded.keys() | current.keys()) - {"files"}
        if name not in recorded or name not in current or recorded[name] != current[name]
    )
    if ("files" in recorded) != ("files" in current):
        result["metadata"].append("files")
    return result
