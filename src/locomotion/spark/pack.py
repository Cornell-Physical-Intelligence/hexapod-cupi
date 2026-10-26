"""Refreeze a prepared pack after the root review or a test-only addition."""

import argparse
import json
from pathlib import Path

from locomotion.robots.hexapod import sha
from locomotion.spark.sources import write_runtime


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refreeze(output):
    """Recompute the runtime manifest, the freeze file, the binding's freeze hash and PACK.json."""
    output = Path(output)
    source = output / "source"
    write_runtime(source)
    (source / "FREEZE_SHA256.json").unlink(missing_ok=True)
    save(
        source / "FREEZE_SHA256.json",
        {
            p.relative_to(source).as_posix(): sha(p)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    )
    freeze = sha(source / "FREEZE_SHA256.json")
    binding = json.loads((output / "binding.json").read_text())
    args = binding["command_args"]
    args[args.index("--source-freeze-sha256") + 1] = freeze
    binding["source_freeze_sha256"] = freeze
    save(output / "binding.json", binding)
    packed = json.loads((output / "PACK.json").read_text())
    packed.update(
        source_freeze_sha256=freeze,
        binding_sha256=sha(output / "binding.json"),
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "PACK.json"
        },
    )
    save(output / "PACK.json", packed)
    return packed


def approve(output):
    """Record the completed root review; the launcher refuses an unreviewed binding."""
    output = Path(output)
    binding = json.loads((output / "binding.json").read_text())
    binding["root_review_complete"] = True
    save(output / "binding.json", binding)
    return refreeze(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("approve", "refreeze"))
    parser.add_argument("--output", type=Path, required=True, help="A prepared pack directory")
    args = parser.parse_args()
    packed = approve(args.output) if args.command == "approve" else refreeze(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_freeze_sha256": packed["source_freeze_sha256"],
                "binding_sha256": packed["binding_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
