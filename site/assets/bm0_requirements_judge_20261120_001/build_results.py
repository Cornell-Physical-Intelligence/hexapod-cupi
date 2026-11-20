"""Build results.json in this folder from bm0's six walking-requirements judge runs (PR 108).

bm0 u1000 (records/distilled_bm0_u1000_20261112) was judged on 2026-11-20 by main's walking-requirements
judge (src/locomotion/evaluation/requirements.py, set stage2_flat, 26 cases) on the CPU MuJoCo surrogate,
with the merged code at commit SOURCE_COMMIT:

- standard_body, own_body: src/locomotion/surrogate/evaluate.py --cases all --video none, seed 20261110,
  default contact model plus the pyramidal-cone and friction-0.9 consensus runs; no fitted axes.
- fitted_{standard,own}_{20261110,20261111}: the same judge through
  records/distilled_bm0_u1000_20261112/scripts/requirements_fitted_judge.py, which draws the five
  leg-stand fitted axes per case (action delay 1-2 controls, kd scale 0.13-0.37, toe friction scale
  0.45-0.6, joint friction, rotor inertia), turns the touchdown model off, checks the motor law against
  each case's drawn damping and skips the contact consensus (it varies friction, which the draws set).
- "own" runs wrap the command in records/distilled_bm0_u1000_20261112/scripts/body_mass.py run: bm0's
  training body, nylon centre piece + 0.75 kg (robot 7.280 kg). own_body has no body_mass.json because
  evaluate.py's closing printout raised KeyError('metrics') after it wrote its summary (a bug on main
  too), which skipped the wrapper's record; the fitted runs catch that error and do have it.

The raw files stay on Spark, read-only, at RAW_ON_SPARK with a SHA256SUMS list; the 400 Hz captures
(about 5 GB) were not kept. Nothing is written unless every file named in RAW is present with the hash
recorded here. Per run the reduction keeps the identity, the pass count, each case's verdict and checked
values, the consensus runs' pass counts, the drawn axes' ranges and the body mass.

Usage: python3 build_results.py <folder holding the raw records, e.g. RAW_ON_SPARK>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW_ON_SPARK = "/srv/cupi/hexapod/runs/julian/pr108_bm0_judge_20261120"
SOURCE_COMMIT = "d5b3c8976b295ba9c607538ab7bc818f29278ffb"
RUNS = {
    "standard_body": ("standard", False),
    "own_body": ("nylon + 0.75 kg", False),
    "fitted_standard_20261110": ("standard", True),
    "fitted_standard_20261111": ("standard", True),
    "fitted_own_20261110": ("nylon + 0.75 kg", True),
    "fitted_own_20261111": ("nylon + 0.75 kg", True),
}
RAW = {
    "fitted_judge.py": "ab7d1176fa2909d41e757ed0189836ad1c916d974e07c406ab61dc3db01473b6",
    "fitted_own_20261110/body_mass.json": "10fab58a6590642de18ddc433cae6d5a5e29fb18fd0814e5930e4d2c71d9d18c",
    "fitted_own_20261110/evaluation/judgement.json": "f90fc0a77cef89132802c7cb451e77c23c4292c8c2409f66d5a3448b661673e4",
    "fitted_own_20261110/fitted_draws.json": "ad2d33af524d46b3124d774e83a47ced3f8b34d06c0b3020a36a6e9c322f5e6b",
    "fitted_own_20261110/surrogate_summary.json": "5478a05d2f82ca27ddaeb4b2d363be2f8c2c7e6e20fdb066b83bccb2a63fe7cf",
    "fitted_own_20261111/body_mass.json": "10fab58a6590642de18ddc433cae6d5a5e29fb18fd0814e5930e4d2c71d9d18c",
    "fitted_own_20261111/evaluation/judgement.json": "1bbeee7d97806db86bfb890a682f4708da9cc6074824dce49465cf17b8ee6e25",
    "fitted_own_20261111/fitted_draws.json": "2bf2d47ed8af70333f557e7df48dddbea9e6bc8c91e0f12fed9cbbacb061ca27",
    "fitted_own_20261111/surrogate_summary.json": "a071e3a7df9e12392c0317336b0cfb300335290e188007afbbd0028c0385ea05",
    "fitted_standard_20261110/evaluation/judgement.json": "31c8ca7723829372986730f7397282b4d2c6bb0e38809fffa71241cf5f977866",
    "fitted_standard_20261110/fitted_draws.json": "ad2d33af524d46b3124d774e83a47ced3f8b34d06c0b3020a36a6e9c322f5e6b",
    "fitted_standard_20261110/surrogate_summary.json": "10af22bdd42e2d335735e8249cab8dfb06d12af395ff2f8d665015d25cff9f34",
    "fitted_standard_20261111/evaluation/judgement.json": "acd5a7b89c30ac0e7b381a01e663c7696fe2a1ea4ec9e5392f2900790fe3a57a",
    "fitted_standard_20261111/fitted_draws.json": "2bf2d47ed8af70333f557e7df48dddbea9e6bc8c91e0f12fed9cbbacb061ca27",
    "fitted_standard_20261111/surrogate_summary.json": "08b8617bcc21980c8345bb033ca48a5be148db534068d41419d766f249dac80f",
    "own_body/contact_consensus/friction_0.9/evaluation/judgement.json": "2ad2c33c962aea10e0c044d4f365da1fa9b18f1ab626bd412d980daa9da96770",
    "own_body/contact_consensus/pyramidal_cone/evaluation/judgement.json": "c55c24801a75a0c8b25a3aee865382eba6a88f563e69de70fde0c10df4c6aa64",
    "own_body/evaluation/judgement.json": "624a82880dc76519b26334d44758b6fae2276eacdd07bc80c9210aec33737f65",
    "own_body/surrogate_summary.json": "3a119962256f7aac8662b284ac44e409f993f12e35a66b96a76f562c417d5487",
    "standard_body/contact_consensus/friction_0.9/evaluation/judgement.json": "dd5664b2d7c7783a3db86f5ebef0135e911d80fd4eb3f001474cb82432f5210c",
    "standard_body/contact_consensus/pyramidal_cone/evaluation/judgement.json": "c0522f67b9c610bbf89e278bbdfa1cec803155ffdbb11464ca4bab8f062b75fc",
    "standard_body/evaluation/judgement.json": "9afffc0fc95845c1e84006fbf600445bc8172033636b43d1e8bc4203f957fd12",
    "standard_body/surrogate_summary.json": "ad94b300559217007fff5309e8ce176ebc9a60ff1aba744ea9a7cf0c8df8c357",
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


def load(path):
    return json.loads(path.read_text())


def draws(path):
    """Per drawn axis, the range over the run's episodes; the delay as a count per value."""
    rows = load(path)["draws_per_episode"]
    keys = [k for k in rows[0] if k != "action_delay_controls"]
    return {
        "episodes": len(rows),
        "action_delay_controls": dict(
            sorted(Counter(str(int(r["action_delay_controls"])) for r in rows).items())
        ),
        **{k: [round(min(r[k] for r in rows), 6), round(max(r[k] for r in rows), 6)] for k in keys},
    }


def run(root, name):
    body, fitted = RUNS[name]
    folder = root / name
    summary = load(folder / "surrogate_summary.json")
    cases = summary["cases"]
    row = {
        "body": body,
        "fitted_axes": fitted,
        "seed": summary["seed"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "repository": summary["repository"]["revision"],
        "requirements_set": summary["requirements_set"],
        "requirements_sha256": summary["requirements_sha256"],
        "backend": summary["backend"],
        "cases": len(cases),
        "passes": sum(bool(c["pass"]) for c in cases),
        "worst": {
            check: max(
                ([round(c["checks"][check]["value"], 4), c["case_id"]] for c in cases
                 if check in c["checks"]),  # fmt: skip
            )
            for check in dict.fromkeys(k for c in cases for k in c["checks"])
        },
        "case_rows": {
            c["case_id"]: [
                bool(c["pass"]),
                c["failed_checks"],
                {k: round(v["value"], 4) for k, v in c["checks"].items()},
            ]
            for c in cases
        },
    }
    consensus = folder / "contact_consensus"
    if consensus.is_dir():
        row["contact_variants"] = {
            variant.name: sum(
                bool(c["pass"]) for c in load(variant / "evaluation/judgement.json")["cases"]
            )
            for variant in sorted(consensus.iterdir())
        }
    if (folder / "fitted_draws.json").is_file():
        row["drawn"] = draws(folder / "fitted_draws.json")
    if (folder / "body_mass.json").is_file():
        mass = load(folder / "body_mass.json")
        row["body_mass"] = {
            "body_link_kg": [round(mass["old_mass_kg"], 4), round(mass["mass_kg"], 4)],
            "robot_kg": round(mass["robot_mass_kg"], 4),
        }
    return row


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
    first = load(root / "standard_body/surrogate_summary.json")["cases"]
    results = {
        "schema": "hexapod_pr108_bm0_requirements_judge_v1",
        "scope": (
            "bm0 u1000 judged by main's walking-requirements judge (stage2_flat) on the CPU MuJoCo "
            "surrogate, with and without the five leg-stand fitted axes, on the standard body and on "
            "bm0's nylon + 0.75 kg training body. Design evidence, not native acceptance. The applied "
            "torque equals the 1.6 N m bound in every case: on the surrogate the motors' rated cap is "
            "the bound, so that check cannot fail here."
        ),
        "raw": {"spark_folder": RAW_ON_SPARK, "source_commit": SOURCE_COMMIT, "files_sha256": RAW},
        "bounds": {
            k: v["bound"] for c in first for k, v in c["checks"].items()
        },
        "case_row": "[pass, failed checks, {check: value}]",
        "runs": {name: run(root, name) for name in RUNS},
    }
    (HERE / "results.json").write_text(dump(results) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
