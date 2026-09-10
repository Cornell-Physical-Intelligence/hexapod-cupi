"""Replay the frozen prototype tests in their original repository-relative layout."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
frozen = json.loads((HERE / "candidate/FREEZE_SHA256.json").read_text())
for name, expected in frozen["files"].items():
    path = HERE / "candidate" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen candidate changed: {name}")
with tempfile.TemporaryDirectory(prefix="velocity_cpu_replay_", dir=ROOT / "tmp") as directory:
    target = Path(directory)
    for name in frozen["files"]:
        shutil.copyfile(HERE / "candidate" / name, target / name)
    subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(target),
                    "-p", "test_*.py", "-v"], cwd=ROOT, check=True)
