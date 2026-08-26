"""Static contract for collision-safe Stage2 fixed-command evaluation spacing."""

from __future__ import annotations

from pathlib import Path
import unittest


ISAACLAB_DIR = Path(__file__).parents[1]
EVALUATOR_PATH = ISAACLAB_DIR / "evaluate_checkpoint.py"
LAUNCHER_PATHS = (
    ISAACLAB_DIR
    / "deploy"
    / "screen-phase2-recovery-stage2c-stable-forward-batch",
    ISAACLAB_DIR / "deploy" / "screen-phase2-recovery-stage2c-robust-best",
    ISAACLAB_DIR / "deploy" / "screen-phase2-recovery-stage2d-homotopy-stage",
    ISAACLAB_DIR / "deploy" / "screen-phase2-recovery-stage2e-joystick-static",
)


class Stage2FixedCommandEvaluatorSpacingTest(unittest.TestCase):
    def test_evaluator_exposes_and_applies_spacing_override(self):
        source = EVALUATOR_PATH.read_text(encoding="utf-8")
        spacing_argument = source[source.index('"--env-spacing"') :][:200]
        self.assertIn("type=float", spacing_argument)
        self.assertIn("env_cfg.scene.env_spacing = args.env_spacing", source)

    def test_multi_environment_launchers_pin_collision_safe_spacing(self):
        for path in LAUNCHER_PATHS:
            with self.subTest(launcher=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertEqual(source.count("evaluate_checkpoint.py"), 1)
                self.assertEqual(source.count("--env-spacing 8.0"), 1)
                evaluation = source[source.index("evaluate_checkpoint.py") :]
                self.assertLess(
                    evaluation.index("--env-spacing 8.0"),
                    evaluation.index("--json"),
                )


if __name__ == "__main__":
    unittest.main()
