"""Verify and replay frozen CPU checks in a new workspace; never overwrite evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def compare_review(expected, actual):
    """Reproduce the CPU report with roundoff tolerance, not a dynamics gate."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("Review field coverage changed")
        return max((compare_review(expected[k], actual[k]) for k in expected), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError("Review row coverage changed")
        return max((compare_review(a, b) for a, b in zip(expected, actual)), default=0.)
    if isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"CPU review differs: {expected} versus {actual}")
        return abs(expected - actual)
    if type(expected) is not type(actual) or expected != actual:
        raise ValueError("Review categorical/count result changed")
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(HERE):
        parser.error("Use a fresh output outside the frozen evidence directory")
    records = json.loads((HERE / "FROZEN_SHA256SUMS.json").read_text())
    for name, expected in records.items():
        path = HERE / name
        if not path.resolve().is_relative_to(HERE) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen payload changed: {name}")
    output.mkdir(parents=True)
    prototype = output / "tmp/omni_reference_prototype"
    review = output / "tmp/ppo_repair_003_preparation/reference_review"
    for source, target in ((HERE / "first", prototype),
                           (HERE / "governor", prototype / "continuation_001"),
                           (HERE / "independent_review", review)):
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.py"):
            shutil.copyfile(path, target / path.name)
    # The continuation reports bind its original source freeze as an input.
    shutil.copyfile(HERE / "first/FREEZE_SHA256.json", prototype / "FREEZE_SHA256.json")
    bench = Path("artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300")
    (output / bench).mkdir(parents=True)
    inputs = {}
    for name in ("candidate_c_reference.json", "f050_t060.urdf"):
        source = ROOT / bench / name
        shutil.copyfile(source, output / bench / name)
        inputs[str(bench / name)] = hashlib.sha256(source.read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", str(prototype), "-p", "test_reference.py", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", str(prototype / "continuation_001"), "-p", "test_time_governor.py", "-v"],
        [sys.executable, str(review / "review_cpu.py")],
    ]
    for index, command in enumerate(commands):
        with (output / f"replay_{index}.log").open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    difference = compare_review(json.loads((HERE / "independent_review/cpu_review.json").read_text()),
                                json.loads((review / "cpu_review.json").read_text()))
    (output / "replay.json").write_text(json.dumps(dict(status="completed", input_hashes=inputs,
        frozen_files_verified=len(records), tests=13, source=str(HERE),
        review_max_absolute_difference=difference), indent=2) + "\n")
    print(output / "replay.json")


if __name__ == "__main__":
    main()
