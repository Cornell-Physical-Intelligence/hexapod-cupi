"""Keep complete physical measurements equal to the frozen pre-batching capture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/mkii_fourbar_2026-09-06/runtime_efficiency_integration_v3"


class CompletePhysicalMetricsEquivalenceTests(unittest.TestCase):
    def test_current_capture_preserves_rows_sensor_order_and_before_reset_guard(self):
        # A separate interpreter keeps the fixture's explicit SDK shim out of
        # the other tests. Default CPU execution never initializes CUDA.
        with tempfile.TemporaryDirectory(prefix="hexapod-metric-equivalence-") as directory:
            report = Path(directory) / "comparison.json"
            result = subprocess.run([
                sys.executable, str(BUNDLE / "compare_complete_metrics.py"),
                "--source-dir", str(ROOT), "--candidate", str(ROOT / "isaaclab/validate_mkii_fourbar.py"),
                "--device", "cpu", "--report", str(report),
            ], cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                text=True, capture_output=True, timeout=180)
            self.assertTrue(report.is_file(), result.stdout + result.stderr)
            evidence = json.loads(report.read_text())
            detail = json.dumps({key: evidence.get(key) for key in
                ("errors", "failures", "test_errors", "traceback")})
            self.assertEqual(result.returncode, 0, detail)
            self.assertIs(evidence["pass"], True, detail)
            self.assertEqual(evidence["tests_run"], 8)
            self.assertIs(evidence["cuda_initialized"], False)
            self.assertIs(evidence["source_unchanged"], True)
            self.assertGreater(evidence["complete_row_comparison"]["row_comparisons"], 0)
            self.assertEqual(evidence["complete_row_comparison"]["max_absolute_difference_by_field"], [0.] * 18)


if __name__ == "__main__":
    unittest.main()
