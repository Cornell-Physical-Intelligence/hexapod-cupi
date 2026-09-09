#!/usr/bin/env python3
"""Read installed SDK files and replay only the cloner's Python callback on stubs.

Run with ordinary container Python, without SimulationApp, a GPU, or SDK imports.
Stdout is JSON evidence; no files or scene state are modified.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


SDK = Path("/workspace/isaaclab/source")
REQUESTS = {
    "interactive_scene": (
        SDK / "isaaclab/isaaclab/scene/interactive_scene.py",
        [(149,156), (182,219), (298,332), (403,435), (452,472), (866,875), (996,999)],
    ),
    "physx_replicate": (
        SDK / "isaaclab_physx/isaaclab_physx/cloner/physx_replicate.py",
        [(16,38), (63,113)],
    ),
    "collision_group_authoring": (
        SDK / "isaaclab/isaaclab/cloner/cloner_utils.py", [(374,478)],
    ),
    "direct_env_setup": (
        SDK / "isaaclab/isaaclab/envs/direct_rl_env.py", [(152,160)],
    ),
    "scene_configuration": (
        SDK / "isaaclab/isaaclab/scene/interactive_scene_cfg.py", [(87,112)],
    ),
    "terrain_configuration": (
        SDK / "isaaclab/isaaclab/terrains/terrain_importer_cfg.py", [(23,37)],
    ),
    "terrain_importer": (
        SDK / "isaaclab/isaaclab/terrains/terrain_importer.py", [],
    ),
    "contact_sensor": (
        SDK / "isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py",
        [(52,67), (288,337), (375,409)],
    ),
}


def snapshot(path, ranges):
    data = path.read_bytes()
    source = data.decode()
    lines = source.splitlines()
    return {
        "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
        "line_count": len(lines),
        "excerpts": [{"first_line": a, "last_line": b,
                      "text": "\n".join(lines[a-1:b])} for a, b in ranges],
        "collision_group_occurrences": [i for i, line in enumerate(lines, 1)
                                       if "collision_group" in line],
    }


class Indices:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, mask):
        return Indices([value for value, keep in zip(self.values, mask) if keep])

    def tolist(self):
        return self.values


class Mapping:
    def __init__(self, count):
        self.count = count

    def size(self, axis):
        assert axis == 1
        return self.count

    def __getitem__(self, index):
        assert index == 0
        return [True] * self.count


def replay_callback(source, device):
    """Exercise actual callback control flow, with no actual stage/physics APIs."""
    calls = []

    class Replicator:
        def register_replicator(self, stage_id, attach, attach_end, rename):
            calls.append({"operation": "register", "excluded_paths": attach(stage_id)})
            self.rename = rename
            attach_end(stage_id)

        def replicate(self, stage_id, src, count, **kwargs):
            calls.append({"operation": "replicate", "source": src, "count": count,
                          "destination_paths": [self.rename(src, i) for i in range(count)],
                          "keyword_arguments": kwargs})

        def unregister_replicator(self, stage_id):
            calls.append({"operation": "unregister"})

    rep = Replicator()
    fake_cache = SimpleNamespace(Get=lambda: SimpleNamespace(
        Insert=lambda stage: SimpleNamespace(ToLongInt=lambda: 123)))
    namespace = {"UsdUtils": SimpleNamespace(StageCache=fake_cache),
                 "get_physx_replicator_interface": lambda: rep}
    parsed = ast.parse(source)
    function = next(node for node in parsed.body
                    if isinstance(node, ast.FunctionDef) and node.name == "physx_replicate")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                        names=[ast.alias(name="annotations")]), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "installed_physx_replicate.py", "exec"), namespace)
    namespace["physx_replicate"](
        object(), ["/World/envs/env_0"], ["/World/envs/env_{}"],
        Indices(list(range(32))), Mapping(32), device=device,
    )
    replicate_calls = [call for call in calls if call["operation"] == "replicate"]
    assert len(replicate_calls) == 1
    assert replicate_calls[0]["count"] == 31
    assert replicate_calls[0]["keyword_arguments"]["useEnvIds"] is False
    assert replicate_calls[0]["destination_paths"] == [f"/World/envs/env_{i}" for i in range(1,32)]
    return {"device_argument": device, "num_envs": 32,
            "mapping": "one homogeneous source maps to all environments", "calls": calls}


def main():
    snapshots = {name: snapshot(path, ranges) for name, (path, ranges) in REQUESTS.items()}
    source = REQUESTS["physx_replicate"][0].read_text()
    result = {
        "schema": "hexapod.sdk_collision_isolation.source_review.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Read-only installed SDK source plus actual Python callback executed against stubs; no SDK import, GPU, or simulation.",
        "sources": snapshots,
        "cloner_callback_replays": [replay_callback(source, device) for device in ("cpu", "cuda:0")],
        "proves_live_collision_isolation": False,
        "terrain_importer_reads_collision_group": bool(
            snapshots["terrain_importer"]["collision_group_occurrences"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
