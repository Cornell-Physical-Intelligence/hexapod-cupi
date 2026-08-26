from __future__ import annotations

import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ISAACLAB_DIR / "deploy"
PYTHON_ENTRYPOINT = "_isaac_sim/python.sh"
SAFE_KIT_ARGUMENT = (
    '--kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true '
    '--/app/extensions/excluded/4=omni.kit.telemetry"'
)


class TelemetryStartupLauncherContractTests(unittest.TestCase):
    def test_every_containerized_isaac_launcher_uses_combined_mitigation(self):
        launchers: list[Path] = []
        for path in sorted(DEPLOY_DIR.iterdir()):
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            invocation_count = source.count(PYTHON_ENTRYPOINT)
            if invocation_count == 0:
                continue
            launchers.append(path)
            self.assertEqual(
                source.count(SAFE_KIT_ARGUMENT),
                invocation_count,
                f"{path.name} must pass the combined telemetry mitigation once "
                "per Isaac Python invocation",
            )
        self.assertTrue(launchers, "expected at least one Isaac deploy launcher")

    def test_phase3_documented_launchers_use_combined_mitigation(self):
        for relative_path in (
            "phase3_sensor_smoke.py",
            "hexapod_phase3/README.md",
        ):
            source = (ISAACLAB_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn(SAFE_KIT_ARGUMENT, source, relative_path)

    def test_skip_deferred_startup_is_never_passed_as_a_standalone_value(self):
        unsafe = "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true"
        for path in sorted(DEPLOY_DIR.iterdir()):
            if path.is_file():
                self.assertNotIn(unsafe, path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
