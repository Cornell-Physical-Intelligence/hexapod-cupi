"""Preserve admission checks for the retained historical demonstration bytes."""

import copy
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

import locomotion.amp.dataset as dataset


class DatasetTests(unittest.TestCase):
    def test_retained_bank_hashes_coverage_and_continuity(self):
        arrays, manifest = dataset.load(dataset.DATASET)
        self.assertEqual(len(arrays["states"]), 18000)
        self.assertEqual(manifest["file_sha256"], dataset.sha(dataset.DATASET / "transitions.npz"))
        self.assertEqual(manifest["coverage"], dataset.motion_cases())
        self.assertEqual(
            dataset.sha(dataset.DATASET / "manifest.json"),
            "7501de7a6be975159ff5c7ede32bbe75aed90095826ff2dfd6106347976ed6a1",
        )

    def test_processing_layouts_keep_history_and_reject_missing_dependencies(self):
        fixture = Path(__file__).with_name("fixtures") / "packages_02f551fd.tar.gz"
        with tarfile.open(fixture) as archive:
            names = ["locomotion/" + entry.name for entry in archive if entry.isfile()]
        names += ["locomotion/priors/commands.py", "locomotion/priors/dataset.py"]
        historical = {
            "schema": "hexapod_processing_sources_v2",
            "numpy_version": "recorded",
            "source_files": {name: "a" * 64 for name in names},
        }
        dataset.validate_processing_identity(historical)
        del historical["source_files"]["locomotion/core/motor.py"]
        with self.assertRaisesRegex(ValueError, "processing source"):
            dataset.validate_processing_identity(historical)
        current = dataset.processing_identity()
        dataset.validate_processing_identity(current)
        changed = copy.deepcopy(current)
        del changed["source_files"]["locomotion/omniwalk/mdp/observations.py"]
        with self.assertRaisesRegex(ValueError, "processing source"):
            dataset.validate_processing_identity(changed)
        changed = copy.deepcopy(current)
        changed["source_files"]["locomotion/omniwalk/mdp/observations.py"] = "invalid"
        with self.assertRaisesRegex(ValueError, "processing source"):
            dataset.validate_processing_identity(changed)
        for schema in ("hexapod_processing_sources_v3", "unreviewed"):
            with self.assertRaisesRegex(ValueError, "processing source"):
                dataset.validate_processing_identity({"schema": schema})

    def test_bank_rejects_changed_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank"
            shutil.copytree(dataset.DATASET, path)
            with (path / "transitions.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes"):
                dataset.load(path)

    def test_hash_bound_bank_rejects_broken_continuity(self):
        arrays, manifest = dataset.load(dataset.DATASET)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank"
            shutil.copytree(dataset.DATASET, path)
            arrays["states"][1, 0] += 1
            np.savez_compressed(path / "transitions.npz", **arrays)
            manifest["file_sha256"] = dataset.sha(path / "transitions.npz")
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "continuity"):
                dataset.load(path)

    def test_native_phases_preserve_gait_trajectory_and_distinct_states(self):
        _, manifest = dataset.load(dataset.DATASET)
        for field, expected in (
            ("period_s", "gait"),
            ("trajectory_sha256", "different optimized trajectories"),
            ("startup_ramp_controls", "startup ramps"),
        ):
            changed = copy.deepcopy(manifest["clips"])
            if field == "period_s":
                changed[0]["identity"]["optimization_config"][field] = 1.4
            else:
                changed[0]["identity"][field] = (
                    50 if field == "startup_ramp_controls" else "changed"
                )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                dataset.require_bank_identity(changed)
        changed = copy.deepcopy(manifest["clips"])
        changed[1]["native_state_sha256"] = changed[0]["native_state_sha256"]
        with self.assertRaisesRegex(ValueError, "second start phase"):
            dataset.require_bank_identity(changed)

    def test_review_retains_direction_tripod_and_video_bindings(self):
        _, manifest = dataset.load(dataset.DATASET)
        accepted = dataset.reviewed_clips(manifest["review"])
        record = manifest["clips"][0]
        for field, value in (
            ("direction_accepted", False),
            ("tripod_accepted", False),
            ("video_sha256", "changed"),
        ):
            changed = copy.deepcopy(accepted)
            changed[record["report_sha256"]][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "acceptance"):
                dataset.require_review(record, changed)


if __name__ == "__main__":
    unittest.main()
