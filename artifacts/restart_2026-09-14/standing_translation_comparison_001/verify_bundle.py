"""Check local evidence hashes and compilation; never contact Spark."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / "BUNDLE.json").read_text())
    files = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
    require(files == set(manifest["files"]) | {"BUNDLE.json"}, "Unexpected bundle files")
    for relative, expected in manifest["files"].items():
        require(sha(ROOT / relative) == expected, "Bundle hash differs: " + relative)
    summary = json.loads((ROOT / "SUMMARY.json").read_text())
    require(sha(ROOT / "ANALYSIS.json") == summary["analysis_sha256"], "Analysis hash differs")
    receipts = dict(summary["receipt_inputs"])
    batch = summary["batch_report_input"]
    receipts[batch["path"]] = batch["sha256"]
    for relative, expected in receipts.items():
        require(sha(REPO / relative) == expected, "Receipt hash differs: " + relative)
    execution = json.loads((ROOT / "EXECUTION.json").read_text())
    require(execution["returncode"] == 0, "Read-only analysis did not complete")
    require(execution["remote_files_written"] is False, "Unexpected remote mutation")
    require(execution["native_or_gpu_calls"] is False, "Unexpected native/GPU execution")
    for relative, expected in execution["local_sources"].items():
        require(sha(ROOT / relative) == expected, "Executed source hash differs: " + relative)
    for key in ("standing_admission", "batch_admission", "training_allowed", "stage2_complete"):
        require(summary[key] is False, "Unexpected admission: " + key)
    sources = sorted(ROOT.glob("*.py"))
    for path in sources:
        compile(path.read_text(), str(path), "exec")
    print(json.dumps({"status": "passed", "bundle_payload_files": len(manifest["files"]),
                      "external_receipts_rehashed": len(receipts), "sources_compiled": len(sources),
                      "remote_access": False, "native_or_gpu_calls": False}, indent=2))


if __name__ == "__main__":
    main()
