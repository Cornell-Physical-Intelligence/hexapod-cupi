"""Build results.json in this folder from the raw evaluation records of PR 108 (bm0 and its curriculum).

The raw records are the 12 surrogate suite summaries of ``records/curriculum_20261109/suites``, the 19
sim2real adapter readouts of ``records/curriculum_20261109/adapter/readouts`` and the two fitted-check logs
of ``records/distilled_bm0_u1000_20261112/judge_results``. They stay on Spark, read-only, at RAW_ON_SPARK,
under the same paths they had in the repository, with a SHA256SUMS list. Nothing is written unless every
file named in RAW is present with the hash recorded here.

The reduction keeps, per suite, the identity it was run under, the pass count, each case's verdict and
failed checks, and the pass count of every contact-model variant. Per adapter readout it keeps the R2 per
randomized parameter, or the passes and crossings per draw. Per fitted-check log it keeps each case's
verdict and failed checks and the log's summary line. Per-control metrics, gait statistics, loads and
the progress lines stay in the raw files.

Usage: python3 build_results.py <folder holding the raw records, e.g. RAW_ON_SPARK>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW_ON_SPARK = "/srv/cupi/hexapod/runs/julian/pr108_raw_results_20261119"
SOURCE_COMMIT = "252250d08e3c738a770d984078538d85547db681"
SUITES = "records/curriculum_20261109/suites"
READOUTS = "records/curriculum_20261109/adapter/readouts"
JUDGE = "records/distilled_bm0_u1000_20261112/judge_results"
RAW = {
    f"{READOUTS}/readout_a.json": "3fbdfb5131377c8c53b2abde8eea954a09702582a0c24d254e99d0ed55fd56a4",
    f"{READOUTS}/readout_a_u200.json": "29771737b73d0f61f17066e4e3f4543afff89709ec00a4da342b8e8518b55b84",
    f"{READOUTS}/readout_a_u500.json": "b8c20843599e497adb967409bd6a55c3645a8e9266a1831ca7acfb97388a03ff",
    f"{READOUTS}/readout_b.json": "760cb943488ad99d1c03860654f9e103b9f9af12b627d1b970e90df6a691b1e7",
    f"{READOUTS}/readout_c.json": "a942daef2aa91a66c9613920474f19176e85a9d48a0f3c6e952eecdfa5fef144",
    f"{READOUTS}/readout_d.json": "a452124ee0d75a5c98376828b013ddc705178c326115c40c7ff775bc6ae00c7b",
    f"{READOUTS}/readout_d_wide.json": "982e21c499761dedfbfec7f4a1708eb3a811729b2a435be117be861bb468c1a8",
    f"{READOUTS}/readout_e_wide.json": "b3d7a69f6fed2ded4b1f33dded765c0f3d89da23f7cb1f2a8da2b69a77f9a257",
    f"{READOUTS}/robust_a.json": "22a54b6a20c0d091c1af42c212c4d70a2ee7f28b3a611a062d3f563bbcb009f9",
    f"{READOUTS}/robust_a_u200.json": "439c6d11e5276610bf4198c1f08817efba5b7d91404dd1f1d3f5064a17855fbf",
    f"{READOUTS}/robust_a_u500.json": "8076b5c85505e2b7a9ace5316ae03425b1f7002469a8e2e70645b9e3ad9bb4be",
    f"{READOUTS}/robust_b.json": "85fea322478ce4b378a668575ae06cf46b2e563070e788363fd71d55b2834c27",
    f"{READOUTS}/robust_b_walk.json": "d39bc608098a3ac92780c53f43679289e783ecedcfc96cd5ea3e91636bbfcb63",
    f"{READOUTS}/robust_c.json": "631075d0c268a528203d76f5a7ff9a5a01d1325a2e9add84fe39fea7e2161684",
    f"{READOUTS}/robust_c_walk.json": "9e1fb76090668bd2c307a94d1e844580b3cb2478657f5481018a927875d64286",
    f"{READOUTS}/robust_d.json": "3ad289dafd6aa95676f7b76fb80caed87eaa13ff2b466153fb9199516c3d4d05",
    f"{READOUTS}/robust_d_walk.json": "833731dd4cc35aba9145bec17afc9ae539debc12efab8a89b9b029e28d1c6c84",
    f"{READOUTS}/robust_e_walk.json": "f4e3553ccc12c83754de3a7d7660693ab84c3c3320c326894ed7f4e1aea75f2e",
    f"{READOUTS}/robust_walk2.json": "970911fe9ee11b0e65225df8160b8d93a70ba5578e178809a17ba18c802e8951",
    f"{SUITES}/full96_gb1_u2000/surrogate_summary.json": "90e844512ad55cda67f915237a4402cedd454a75d7dcc0c32ebd1d3d0737739a",
    f"{SUITES}/full96_od1_u1200/surrogate_summary.json": "bda7b9f791bd481c6bca3af645650e2c20771d449ccdaddbe3bc97c5d2bacdfc",
    f"{SUITES}/full96_od1_u1200_gaitcontact/surrogate_summary.json": "fb5ef7c78d5021969a8d774fe16ebbc2a6b65871220101c1acf89569f3daed10",
    f"{SUITES}/full96_od1_u1200_nocontact/surrogate_summary.json": "f8bc1ebadaa88a362048e54fcf5179128e84895edf5fa0ac8fc32dbea788656d",
    f"{SUITES}/full96_od1_u2000/surrogate_summary.json": "e59c3d81f6bf9e3637ed1425b473d3521fa45d5c10a7b5162a4531d17fb40441",
    f"{SUITES}/full96_od1_u600/surrogate_summary.json": "c01e25d0c647d2b35102562dbb658d75fa978f0cb7c1aacabca59104a282eedc",
    f"{SUITES}/full96_tw_s3_u600/surrogate_summary.json": "9609a29d41f2c7e2d53f445d82cc5ebae366c42c54434061d22c0a203582edce",
    f"{SUITES}/omni15_gb1_u2000/surrogate_summary.json": "a8aacefb4e224ad7c697baae5448a9858ccd9dec74698058b0c0b944565a929e",
    f"{SUITES}/omni15_od1_u1200/surrogate_summary.json": "caab66c23006fb9aa52af6586899c89b4936b624bc5cdb6d6b6ee87c6a8d2e9b",
    f"{SUITES}/omni15_od1_u1200_gaitcontact/surrogate_summary.json": "9344af93ad221559df9ed9093a8600838d6a992603e4d712aeb0980e76be1e08",
    f"{SUITES}/omni15_od1_u2000/surrogate_summary.json": "59ba1293a4a23b06929ef335b6b4907ef1bda9e53c36a39e00320e18910aec30",
    f"{SUITES}/omni15_od1_u600/surrogate_summary.json": "1f8240793db21881fd2c31d5960615bb047724277b1a2fa40ff77357c4bf01bf",
    f"{JUDGE}/fitted96_all5_nylon750_bm0_u1000_s20261110.log": "30bc1b094c97a4f1b0e746635ce3031e7a0b0d4472a64f6da822490b1fed963e",
    f"{JUDGE}/fitted96_all5_nylon750_bm0_u1000_s20261111.log": "3a2ccb2cab59d40fc02f20b0f2905381ab810bed1eaed41e1df57ae977dafc30",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(root):
    """Every raw file, checked against its recorded hash before anything is read."""
    for name, digest in RAW.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"{path} is missing")
        if sha256(path) != digest:
            raise ValueError(f"{path} differs from its recorded hash")


def case_rows(cases):
    return {c["case_id"]: [bool(c["pass"]), c.get("failed_bounds") or []] for c in cases}


def suite(path):
    s = json.loads(path.read_text())
    cases = s["cases"]
    failed = Counter(b for c in cases for b in (c.get("failed_bounds") or []))
    variants = s.get("contact_consensus_variants") or {}
    return {
        "checkpoint_sha256": s["checkpoint_sha256"],
        "checkpoint_origin": s.get("checkpoint_origin"),
        "task": s["task"],
        "wrapper_options": s["wrapper_options"],
        "seed": s["seed"],
        "backend": s["backend"],
        "cases": len(cases),
        "passes": sum(bool(c["pass"]) for c in cases),
        "fell": sum("terminated" in (c.get("failed_bounds") or []) for c in cases),
        "failed_bounds": dict(failed.most_common()),
        "contact_variants": {
            name: {
                "fields": v.get("contact_model_fields"),
                "cases": len(v["cases"]),
                "passes": sum(bool(c["pass"]) for c in v["cases"]),
            }
            for name, v in variants.items()
        },
        "case_rows": case_rows(cases),
    }


def readout(path):
    d = json.loads(path.read_text())
    if "readout" in d:
        return {
            "checkpoint_sha256": d["checkpoint_sha256"],
            "seed": d["seed"],
            "episodes": d["episodes"],
            "held_out": d["held_out"],
            "ranges": d["ranges"],
            "r2": {k: round(v["r2"], 4) for k, v in d["readout"].items()},
        }
    out = {k: d[k] for k in ("scored_ranges", "walked_ranges", "speed", "bearings", "base_seed")}
    scored = {}
    for arm, draws in d["results"].get("scored", {}).items():
        per = {draw: [r["passes"], len(r["cases"])] for draw, r in draws.items()}
        scored[arm] = {"per_draw": per, "passes": sum(p for p, _ in per.values()),
                       "cases": sum(n for _, n in per.values())}
    walked = {}
    for arm, worlds in d["results"].get("walked", {}).items():
        if all(k.startswith("draw") for k in worlds):
            worlds = {"unlabelled": worlds}
        walked[arm] = {}
        for world, draws in worlds.items():
            per = {draw: [r["crossed"], len(r["starts"])] for draw, r in draws.items()}
            falls = sum(s["fell_at_s"] is not None for r in draws.values() for s in r["starts"])
            walked[arm][world] = {"per_draw": per, "crossed": sum(c for c, _ in per.values()),
                                  "starts": sum(n for _, n in per.values()), "falls": falls}
    out.update(scored=scored, walked=walked)
    return out


def judge(path):
    rows, summary, body = {}, None, None
    for line in path.read_text().splitlines():
        if line.startswith("{") and '"case"' in line:
            r = json.loads(line)
            rows[r["case"]] = [bool(r["pass"]), r["failed"]]
        elif line.startswith("FITTED "):
            summary = line
        elif line.startswith("BODY MASS: "):
            body = json.loads(line[len("BODY MASS: "):])
    if len(rows) != 96 or summary is None:
        raise ValueError(f"{path}: expected 96 case lines and a summary line")
    return {"summary": summary, "passes": sum(p for p, _ in rows.values()), "cases": len(rows),
            "body_mass": body, "case_rows": rows}


def dump(value, depth=0, inline_from=4):
    """JSON with one line per case row: containers nested deeper than ``inline_from`` stay inline."""
    if depth >= inline_from or not isinstance(value, (dict, list)) or not value:
        return json.dumps(value)
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k)}: {dump(v, depth + 1, inline_from)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    items = [f"{inner}{dump(v, depth + 1, inline_from)}" for v in value]
    return "[\n" + ",\n".join(items) + f"\n{pad}]"


def main(root):
    root = Path(root)
    verified(root)
    results = {
        "schema": "hexapod_pr108_results_v1",
        "scope": (
            "CPU MuJoCo surrogate evaluations of the destination_v70 curriculum (od1, gb1, tw_s3), the "
            "sim2real adapter arms and bm0's fitted checks. Design evidence, not native acceptance."
        ),
        "raw": {"spark_folder": RAW_ON_SPARK, "source_commit": SOURCE_COMMIT, "files_sha256": RAW},
        "suites": {
            p: suite(root / SUITES / p / "surrogate_summary.json")
            for p in sorted({name.split("/")[3] for name in RAW if name.startswith(SUITES)})
        },
        "adapter_readouts": {
            Path(name).stem: readout(root / name) for name in RAW if name.startswith(READOUTS)
        },
        "bm0_fitted_checks": {
            Path(name).stem: judge(root / name) for name in RAW if name.startswith(JUDGE)
        },
    }
    (HERE / "results.json").write_text(dump(results) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
