"""Verify and test immutable synthetic prototype from any working directory."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file() and (p / "tools/terrain_readiness.py").is_file())
PROTOTYPE = HERE / "prototype"
manifest = json.loads((PROTOTYPE / "FREEZE_SHA256.json").read_text())
for rel, expected in manifest.get("files", manifest).items():
    assert hashlib.sha256((PROTOTYPE / rel).read_bytes()).hexdigest() == expected, rel
for rel, expected in json.loads((PROTOTYPE / "SOURCE_SHA256.json").read_text()).items():
    path = ROOT / rel
    if rel == "tmp/stage3_perception_review_001/REVIEW.md":
        path = ROOT / "artifacts/perception_readiness_2026-09-09/stage3_review_001/REVIEW.md"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, rel
sys.path[:0] = [str(PROTOTYPE), str(ROOT / "tools"), str(ROOT / "isaaclab")]
suite = unittest.defaultTestLoader.discover(str(PROTOTYPE), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
