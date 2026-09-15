"""Artifact-only exact-case check for the current registry on macOS."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("site_case_audit", ROOT / "tools/project_site.py")
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)
original = site.reference
checked = set()


def exact_reference(value):
    result = original(value)
    if result is not None:
        current = ROOT
        for part in Path(value).parts:
            names = {path.name for path in current.iterdir()}
            if part not in names:
                raise ValueError("Evidence path case mismatch: " + value)
            current = current / part
        checked.add(value)
    return result


site.reference = exact_reference
project = site.registry()
site.validate_progress(project)
altered = copy.deepcopy(project)
target = next(f for f in altered["progress"]["facts"] if f["id"] == "canonical-bc-start-quiet-visual-20260914")
assert target["source"].endswith("/visual_review_004/REVIEW.json")
target["source"] = target["source"].replace("/REVIEW.json", "/review.json")
try:
    site.validate_progress(altered)
except ValueError as error:
    assert "case mismatch" in str(error)
else:
    raise AssertionError("Lowercase regression was not rejected")
print(json.dumps({"current_registry_exact_case_valid": True,
                  "incorrect_case_negative_fixture_rejected": True,
                  "unique_references_checked": len(checked)}))
