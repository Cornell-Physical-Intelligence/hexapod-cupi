"""Verify the published review and verbatim specialist manifests."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((root / "BUNDLE_SHA256.json").read_text())
for name, digest in manifest.items():
    assert sha(root / name) == digest, name
for directory in (root / "reviews").iterdir():
    frozen = json.loads((directory / "FREEZE_SHA256.json").read_text())
    for name, digest in frozen.get("payloads", frozen).items():
        assert sha(directory / name) == digest, str(directory / name)
print(f"Verified {len(manifest)} payloads and all three specialist freezes")
