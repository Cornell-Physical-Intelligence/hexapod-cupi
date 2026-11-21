"""Tabulate the scorer's verdicts for fitted_eval.py runs beside the normal-simulator judge runs.

Reads only the evaluator's own surrogate_summary.json (and fitted_draws.json for the label); it
computes no metric of its own. Rows are simulator configurations, columns the checkpoints.

Usage: python3 fitted_report.py [--names a,b,...] [LABEL=ROOT ...]   (ROOT holds <name>/surrogate_summary.json;
"{seed}" in ROOT expands over the two evaluation seeds). With no arguments it reports every
configuration scored on 2026-11-10.
"""

import collections
import json
import re
import sys
from pathlib import Path

C = Path("/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum")
NAMES = ("fit1_u2000", "od1_u1200")
SEEDS = (20261110, 20261111)
DEFAULT = [
    "contact only (control)=fitted96_contactonly_s20261110",
    "delay only=fitted96_delay_s{seed}",
    "4 axes, no kd=fitted96_nokd_s{seed}",
    "all 5, drawn-kd check=fitted96_all5_drawnkd_s{seed}",
]
FAMILIES = ("static", "stop", "formal", "quiet", "transition")


def family(case_id):
    return case_id.split(":")[1].split("__")[0]


def kind(case_id):
    name, _, rest = case_id.split(":")[1].partition("__")
    if name != "static":
        return name
    match = re.match(r"([a-z_]+?)_([0-9.]+)_", rest)
    return f"{match.group(1)} {match.group(2)}" if match else rest.split("_")[0]


def load(directory):
    path = directory / "surrogate_summary.json"
    if not path.exists():
        return None
    return {case["case_id"]: case for case in json.loads(path.read_text())["cases"]}


def line(label, name, cases):
    ok = collections.Counter(family(i) for i, c in cases.items() if c["pass"])
    total = collections.Counter(family(i) for i in cases)
    early = sum(bool(c.get("batch_failure")) for c in cases.values())
    split = "".join(f'{str(ok[f]) + "/" + str(total[f]):>8}' for f in FAMILIES)
    print(f"{label:<34}{name:<14}{sum(ok.values()):>3}/{len(cases):<4}{split}{early:>7}")


def main(argv):
    names = NAMES
    if "--names" in argv:
        at = argv.index("--names")
        names = tuple(argv[at + 1].split(","))
        argv = argv[:at] + argv[at + 2 :]
    specs = argv or DEFAULT
    print(f'{"simulator":<34}{"model":<14}{"pass":<8}' + "".join(f"{f[:7]:>8}" for f in FAMILIES) + "  early")
    for name in names:
        cases = load(C / f"full96_{name}")
        if cases:
            line("normal (judge result)", name, cases)
    pooled = {}
    for spec in specs:
        label, _, root = spec.partition("=")
        roots = [(root.format(seed=s), s) for s in SEEDS] if "{seed}" in root else [(root, None)]
        for path, seed in roots:
            tag = f"{label} s{seed}" if seed else label
            for name in names:
                cases = load(C / path / name)
                if cases is None:
                    if name in NAMES:
                        print(f"{tag:<34}{name:<14}MISSING")
                    continue
                line(tag, name, cases)
                table = pooled.setdefault(label, {})
                for case_id, case in cases.items():
                    row = table.setdefault(kind(case_id), collections.Counter())
                    row["n"] += 1
                    row["pass"] += bool(case["pass"])
                    row["early"] += bool(case.get("batch_failure"))
            pairs = [load(C / path / name) for name in NAMES]
            if all(pairs):
                a, b = pairs
                both = sum(a[i]["pass"] and b[i]["pass"] for i in a)
                draws = [C / path / name / "fitted_draws.json" for name in NAMES]
                same = [json.loads(d.read_text())["draws"] for d in draws]
                print(
                    f"   {tag}: both pass {both}, only fit1 "
                    f'{sum(a[i]["pass"] and not b[i]["pass"] for i in a)}, only od1 '
                    f'{sum(b[i]["pass"] and not a[i]["pass"] for i in a)}; '
                    f"draws identical {same[0] == same[1]} ({len(same[0])})"
                )
    kinds = sorted({k for table in pooled.values() for k in table})
    print("\npass (ended early) by command type, pooled over models and seeds")
    print(f'{"kind":<16}' + "".join(f"{label[:22]:>24}" for label in pooled))
    for k in kinds:
        cells = []
        for table in pooled.values():
            r = table.get(k)
            cells.append(f'{r["pass"]}/{r["n"]} ({r["early"]})' if r else "-")
        print(f"{k:<16}" + "".join(f"{cell:>24}" for cell in cells))


if __name__ == "__main__":
    main(sys.argv[1:])
