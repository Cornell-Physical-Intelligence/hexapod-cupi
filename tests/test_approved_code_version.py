"""The approved code version names a receipt that exists, binds the accepted controller, and is on the branch."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "configs/approved_code_version.json"
ACCEPTED = ROOT / "configs/accepted_controller.json"


class ApprovedCodeVersionTests(unittest.TestCase):
    def setUp(self):
        self.approved = json.loads(APPROVED.read_text())

    def test_receipt_bytes_match_the_recorded_hash(self):
        receipt = self.approved["receipt"]
        digest = hashlib.sha256((ROOT / receipt["path"]).read_bytes()).hexdigest()
        self.assertEqual(receipt["sha256"], digest)
        bound = json.loads((ROOT / receipt["path"]).read_text())["candidate_runtime_sources"][
            "files"
        ]
        self.assertEqual(receipt["runtime_files_bound"], len(bound))

    def test_receipt_binds_the_accepted_controller(self):
        accepted = json.loads(ACCEPTED.read_text())
        receipt = json.loads((ROOT / self.approved["receipt"]["path"]).read_text())
        self.assertEqual(receipt["checkpoint_sha256"], accepted["controller"]["checkpoint_sha256"])
        self.assertEqual(
            self.approved["controller_checkpoint_sha256"],
            accepted["controller"]["checkpoint_sha256"],
        )
        self.assertEqual(
            accepted["compatibility_receipt"]["sha256"], self.approved["receipt"]["sha256"]
        )

    def test_certified_commit_is_on_this_branch(self):
        commit = self.approved["certified_commit"]
        known = subprocess.run(["git", "cat-file", "-e", commit], cwd=ROOT, capture_output=True)
        if known.returncode != 0:
            self.skipTest("shallow clone without the certified commit")
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT)
        self.assertEqual(ancestor.returncode, 0, commit + " is not an ancestor of HEAD")
