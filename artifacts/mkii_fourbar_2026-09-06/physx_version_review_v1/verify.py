#!/usr/bin/env python3
"""Verify the local evidence records; does not connect to Spark or import SDKs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
installed = json.loads((HERE / "installed_image_metadata.json").read_text())
expected_image = "sha256:8ddc1623d70d5dd622fd728ce4eb3f59dea6ce1ea5cfbe858353dab15e8d0ef8"
expected_plugin = "a66e7338758473c3361e7d81730a2fab3c1f746329e4e99541bf7e7834ab4b35"
assert installed["image"]["Id"] == expected_image == installed["container_image_id"]
assert installed["container_started"] is False
assert installed["gpu_process_started"] is False
assert installed["exact_stopped_container_removed"] is True
for key in ("state_before", "state_after"):
    assert installed[key]["Status"] == "created"
    assert installed[key]["Running"] is False and installed[key]["Pid"] == 0
plugin = next(f for f in installed["files"] if f["path"].endswith("libomni.physx.plugin.so"))
assert plugin["sha256"] == expected_plugin
strings = [s["text"] for s in plugin["selected_printable_strings"]]
assert all(s in strings for s in ("110.1.13", "c38f7d1", "Jun-04-2026", "5.9.0"))
assert any("more than 4 velocity iterations" in s for s in strings)
for entry in installed["files"]:
    if "text" in entry:
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
binding = (HERE / "live_profiler_image_binding.txt").read_text()
assert expected_image in binding and expected_plugin in binding
assert "/proc/15/maps:" in binding and plugin["path"] in binding
manifest = json.loads((HERE / "public_source_manifest.json").read_text())
assert manifest["commit"] == "517a0073715120e114ee055b63b26c95e00d9039"
for entry in manifest["sources"]:
    data = (HERE / entry["local_path"]).read_bytes()
    assert len(data) == entry["bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
header = (HERE / "public_sources/physx__include__foundation__PxPhysicsVersion.h").read_text()
for name, value in (("MAJOR", 5), ("MINOR", 9), ("BUGFIX", 0)):
    assert f"#define PX_PHYSICS_VERSION_{name} {value}" in header
print("PASS: recorded installed-image identity, live plugin binding, static version markers, metadata cleanup and public-source hashes")
