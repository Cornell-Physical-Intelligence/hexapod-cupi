"""Build results.json in this folder from the surrogate evaluation folders and the native v9 96-case run.

The surrogate speed suite is the 13 learning probes plus walks (omni_static, 1000 controls) and walk-then-stop
cases (stop_to_stand, 1050 controls) at 0.05 to 0.175 m/s in 8 directions; the 96 required cases are
locomotion.evaluation.runner.case_manifest(), run as surrogate learning probes with the native 90 s episode
timeout. Pass/fail and failed checks copy each case's report.json. The end gap is the planar distance to the
ghost after the last control (viewer/run_view.py: the ghost starts at the reset pose, follows the command and
restarts at the body when the command changes, as training builds it).

Nothing is written unless every report declares its folder's checkpoint, the evaluation source and the seed, and
every capture a surrogate report lists has the hash it declares. The native run's cases read state.json, and its
reports carry per-case seeds, so it is held to its checkpoint and source, the run's seed in state.json and the
96 cases; its full captures stay on Spark. Each output records what it was bound to.

Usage: uv run python build_results.py <runs/curriculum folder> <native v9 96-case run/standing folder>
"""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
from viewer.run_view import ghost  # noqa: E402

SEED = 20260917
SOURCE_SHA256 = "6a2c16eb69406c710a4c9ba4eafdd8f73928a107b5807499ff9227a0768d5bc1"
V9_SHA256 = "16048b8f7b31c61ba6630742475f796e24c4bc43bd17bc61f9b40bb0d37b6d33"
V29_U600_SHA256 = "ed73eb07c1f9c364e33fd6673c53d2a1c4a54d4086bda1e05ca72b680f36ccf8"
SPEED_SUITES = [
    ("destination_v9", "suite_v9", V9_SHA256),
    (
        "destination_v25_update1350",
        "suite_spark_v25_u1350",
        "b0a2ccf3d82ccaa1c38f66b36340ceec7b6ced969af281aca9f99263c9c3d529",
    ),
    (
        "destination_v29_update300",
        "suite_spark_v29_u300",
        "3983a56956843fa435a157764c8119f6e469954cd462c6cb0f57e9d9d3f97b94",
    ),
    ("destination_v29_update600", "suite_spark_v29_u600", V29_U600_SHA256),
]
REQUIRED = (
    "destination_v29_update600",
    ["full96_spark_v29_u600", "full96_spark_v29_u600_transition"],
    V29_U600_SHA256,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_identity(report, path, declared):
    for key, value in declared.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key} is {report.get(key)!r}, expected {value!r}")


def cases(folder, checkpoint):
    batches = sorted(Path(folder, "evaluation").glob("batch_*"))
    if not batches:
        raise ValueError(f"{folder} has no evaluation batches")
    declared = {"checkpoint_sha256": checkpoint, "source_sha256": SOURCE_SHA256, "seed": SEED}
    for batch in batches:
        report = json.loads((batch / "report.json").read_text())
        check_identity(report, batch, declared)
        files = report.get("files") or {}
        if "control_trace.npz" not in files:
            raise ValueError(f"{batch}: the report declares no control_trace.npz hash")
        for name, digest in files.items():
            if sha256(batch / name) != digest:
                raise ValueError(f"{batch}: {name} differs from its declared hash")
        yield report["assigned_case_ids"][0].split(":", 1)[1], report["results"][0], batch


class Bindings:
    """What a table was reduced from: its identity, and digests of its reports and declared trace hashes."""

    def __init__(self, checkpoint):
        self.checkpoint, self.reports, self.captures = checkpoint, hashlib.sha256(), hashlib.sha256()

    def add(self, batch):
        text = (batch / "report.json").read_bytes()
        self.reports.update(text)
        self.captures.update(json.loads(text)["files"]["control_trace.npz"].encode())

    def record(self):
        return {
            "checkpoint_sha256": self.checkpoint,
            "source_sha256": SOURCE_SHA256,
            "seed": SEED,
            "reports_sha256": self.reports.hexdigest(),
            "captures_sha256": self.captures.hexdigest(),
        }


def speed_suite(folder, checkpoint):
    rows, bound = {}, Bindings(checkpoint)
    for case, result, batch in cases(folder, checkpoint):
        bound.add(batch)
        failed = result.get("failed_bounds") or []
        trace = np.load(batch / "control_trace.npz")
        pose = trace["root_pose_xyzw"][:, 0].astype(float)
        command = trace["command"][:, 0].astype(float)
        end = float(np.hypot(*(pose[-1, :2] - ghost(pose, command)[-1, :2])) * 1000)
        rows[case] = {
            "pass": bool(result["pass"]),
            "fell": "terminated" in failed or bool(trace["terminated"][:, 0].any()),
            "end_gap_mm": round(end, 2),
            "failed_bounds": failed,
        }
    groups = {"probes": [v for k, v in rows.items() if not k.startswith(("walk_", "stop_"))]}
    for k, v in rows.items():
        if k.startswith(("walk_", "stop_")):
            kind, speed, _ = k.split("_")
            groups.setdefault(f"{kind} {speed}", []).append(v)
    table = {
        name: {
            "cases": len(v),
            "passed": sum(x["pass"] for x in v),
            "fell": sum(x["fell"] for x in v),
            "median_end_gap_mm": round(statistics.median(x["end_gap_mm"] for x in v), 2),
            "worst_end_gap_mm": round(max(x["end_gap_mm"] for x in v), 2),
        }
        for name, v in groups.items()
    }
    return {**bound.record(), "groups": table, "cases": rows}


def group(case):
    kind, name = case.split(":", 1)
    if kind != "static":
        return kind
    return "static " + "_".join(p for p in name.split("_") if not p.endswith("deg"))


def required(rows):
    table = {}
    for case, row in rows.items():
        entry = table.setdefault(group(case), {"cases": 0, "passed": 0})
        entry["cases"] += 1
        entry["passed"] += row["pass"]
    return {"passed": sum(r["pass"] for r in rows.values()), "cases": len(rows), "groups": table, "cases_detail": rows}


def native_v9(native):
    """The native v9 cases, once the run's seed and every report's checkpoint and source are as declared."""
    state = json.loads(Path(native, "state.json").read_text())
    check_identity(state["identity"], Path(native, "state.json"), {"seed": SEED})
    rows = {
        r["case_id"]: {"pass": bool(r.get("pass")), "failed_bounds": r.get("failed_bounds") or []}
        for r in state["evaluation"]["results"]
    }
    declared = {"checkpoint_sha256": V9_SHA256, "source_sha256": SOURCE_SHA256}
    reports, covered = hashlib.sha256(), set()
    for batch in sorted(Path(native, "evaluation").glob("batch_*")):
        text = (batch / "report.json").read_bytes()
        report = json.loads(text)
        check_identity(report, batch, declared)
        reports.update(text)
        covered.update(report["assigned_case_ids"])
    if covered != set(rows):
        raise ValueError("The native reports must cover the cases in its state.json")
    return rows, {
        **declared,
        "seed": SEED,
        "reports_sha256": reports.hexdigest(),
        "state_sha256": sha256(Path(native, "state.json")),
    }


def main(runs, native, output=HERE / "results.json"):
    result = {
        "schema": "hexapod_destination_curriculum_results_v1",
        "scope": "Surrogate (CPU MuJoCo) evaluations: design evidence, not acceptance evidence. The native "
        "evaluator does not yet rebuild the ghost_v11 inputs or the speed-scaled gait clock, so the native "
        "trainings of destination_v25 and destination_v29 were evaluated on the surrogate. One trial per case, "
        f"the actor mean without sampling noise, seed {SEED}, without the extra contact-model checks.",
        "speed_suite": {
            name: speed_suite(Path(runs, folder), checkpoint) for name, folder, checkpoint in SPEED_SUITES
        },
    }
    rows, bound = {}, Bindings(REQUIRED[2])
    for folder in REQUIRED[1]:
        for case, res, batch in cases(Path(runs, folder), REQUIRED[2]):
            bound.add(batch)
            rows[case.replace("__", ":", 1)] = {"pass": bool(res["pass"]), "failed_bounds": res.get("failed_bounds") or []}
    v9, native_bound = native_v9(native)
    if set(rows) != set(v9) or len(v9) != 96:
        raise ValueError("The surrogate run and the native v9 run must cover the same 96 required cases")
    result["required_96"] = {
        REQUIRED[0] + "_surrogate": {**bound.record(), **required(rows)},
        "destination_v9_native": {**native_bound, **required(v9)},
    }
    Path(output).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
