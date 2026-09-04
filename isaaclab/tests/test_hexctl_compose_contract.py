"""Contract tests for hexctl run composition.

Composition turns a named experiment config plus intervention deltas into the
exact argv of the existing hardened launcher. Three properties matter and are
pinned here:

* the argv is exactly the launcher, the label, the seed, and the intervention
  overrides — baseline keys are never re-emitted, because the launcher pins the
  whole baseline itself and merges interventions into it;
* the override order is stable, so the same intervention set always composes to
  the same command line regardless of how the config files are supplied;
* composition refuses outright when the named config and the launcher's pinned
  bash arrays have diverged, so a YAML file can never silently stop describing
  what the launcher will actually run.

Standard library only: nothing is launched, nothing is imported from Isaac Lab,
and the real launcher on disk is read but never written.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_train"
if str(PACKAGE_DIR) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
    sys.path.insert(0, str(PACKAGE_DIR))

from hexapod_train import compose as compose_module  # noqa: E402
from hexapod_train import configio  # noqa: E402


LAUNCHER = ISAACLAB_DIR / "deploy" / "probe-stage2c-single-current-best"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "stage2c_accel.yaml"
INTERVENTION_CONFIG = REPO_ROOT / "configs" / "intervention" / "probe21_bilateral.yaml"

EXPECTED_PROBE21_ARGV = [
    "./isaaclab/deploy/probe-stage2c-single-current-best",
    "X",
    "99",
    "env.inactive_bilateral_longitudinal_contact_moment_reward_scale=-2.0",
    "env.inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8",
]


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class ProbeCompositionTests(unittest.TestCase):
    def test_probe21_composes_the_exact_documented_argv(self) -> None:
        invocation = compose_module.compose_probe(
            EXPERIMENT_CONFIG,
            [INTERVENTION_CONFIG],
            seed=99,
            label="X",
        )
        self.assertEqual(invocation.argv, EXPECTED_PROBE21_ARGV)

    def test_intervention_values_keep_their_source_spelling(self) -> None:
        # 1.8 is the calibrated reference; it must reach the launcher as the
        # token the config file actually contains, never re-rendered.
        invocation = compose_module.compose_probe(
            EXPERIMENT_CONFIG, [INTERVENTION_CONFIG], seed=99, label="X"
        )
        self.assertIn(
            "env.inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8",
            invocation.argv,
        )

    def test_no_baseline_key_is_re_emitted(self) -> None:
        baseline = compose_module.parse_launcher(LAUNCHER.read_text(encoding="utf-8"))
        invocation = compose_module.compose_probe(
            EXPERIMENT_CONFIG, [INTERVENTION_CONFIG], seed=99, label="X"
        )
        emitted = {token.split("=", 1)[0] for token in invocation.overrides}
        self.assertTrue(emitted.issubset(set(baseline.allowed_intervention_keys)))
        untouched = {f"env.{key}" for key in baseline.baseline_env} - emitted
        for key in sorted(untouched):
            self.assertNotIn(key, {token.split("=", 1)[0] for token in invocation.overrides})
        for key in baseline.fixed_agent:
            self.assertNotIn(
                f"agent.{key}", {token.split("=", 1)[0] for token in invocation.overrides}
            )

    def test_zero_intervention_control_composes_to_label_and_seed_only(self) -> None:
        invocation = compose_module.compose_probe(
            EXPERIMENT_CONFIG, [], seed=99, label="probe20_control"
        )
        self.assertEqual(
            invocation.argv,
            ["./isaaclab/deploy/probe-stage2c-single-current-best", "probe20_control", "99"],
        )

    def test_override_order_is_stable_across_supply_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            scale = write(
                directory,
                "scale.yaml",
                "env:\n"
                "  inactive_bilateral_longitudinal_contact_moment_reward_scale: -2.0\n",
            )
            reference = write(
                directory,
                "reference.yaml",
                "env:\n"
                "  inactive_bilateral_longitudinal_contact_moment_reference_nm: 1.8\n",
            )
            forward = compose_module.compose_probe(
                EXPERIMENT_CONFIG, [scale, reference], seed=99, label="X"
            )
            backward = compose_module.compose_probe(
                EXPERIMENT_CONFIG, [reference, scale], seed=99, label="X"
            )
        self.assertEqual(forward.argv, EXPECTED_PROBE21_ARGV)
        self.assertEqual(backward.argv, EXPECTED_PROBE21_ARGV)


class CompositionRefusalTests(unittest.TestCase):
    def _compose_with_intervention(self, body: str):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), "intervention.yaml", body)
            return compose_module.compose_probe(
                EXPERIMENT_CONFIG, [path], seed=99, label="X"
            )

    def test_key_outside_the_launcher_allowlist_is_refused(self) -> None:
        # A real baseline key, deliberately not part of the intervention
        # surface: the launcher would exit 64, so composition refuses first.
        with self.assertRaises(compose_module.CompositionError) as raised:
            self._compose_with_intervention(
                "env:\n  inactive_yaw_rate_reward_scale: -1.0\n"
            )
        self.assertIn("allowed_intervention_keys", str(raised.exception))

    def test_key_outside_the_named_baseline_is_refused(self) -> None:
        with self.assertRaises(compose_module.CompositionError) as raised:
            self._compose_with_intervention("env:\n  not_a_real_reward_scale: 1.0\n")
        self.assertIn("named experiment baseline", str(raised.exception))

    def test_non_finite_and_non_positive_reference_values_are_refused(self) -> None:
        for body, expected in (
            ("env:\n  inactive_bilateral_longitudinal_contact_moment_reward_scale: nan\n",
             "finite"),
            ("env:\n  inactive_bilateral_longitudinal_contact_moment_reference_nm: 0.0\n",
             "strictly positive"),
        ):
            with self.subTest(body=body):
                with self.assertRaises(compose_module.CompositionError) as raised:
                    self._compose_with_intervention(body)
                self.assertIn(expected, str(raised.exception))

    def test_duplicate_intervention_key_across_files_is_refused(self) -> None:
        body = "env:\n  inactive_bilateral_longitudinal_contact_moment_reward_scale: -2.0\n"
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first = write(directory, "a.yaml", body)
            second = write(directory, "b.yaml", body)
            with self.assertRaises(compose_module.CompositionError) as raised:
                compose_module.compose_probe(
                    EXPERIMENT_CONFIG, [first, second], seed=99, label="X"
                )
        self.assertIn("duplicate intervention key", str(raised.exception))

    def test_unsafe_label_and_seed_are_refused(self) -> None:
        for label, seed in (("../escape", 99), ("ok_label", -1), ("ok_label", 2**31)):
            with self.subTest(label=label, seed=seed):
                with self.assertRaises(compose_module.CompositionError):
                    compose_module.compose_probe(
                        EXPERIMENT_CONFIG, [INTERVENTION_CONFIG], seed=seed, label=label
                    )

    def test_divergence_between_config_and_launcher_refuses_composition(self) -> None:
        # Simulated on a copy. The real launcher is never modified.
        original = LAUNCHER.read_text(encoding="utf-8")
        drifted = original.replace(
            "env.foot_slip_reward_scale=-0.50", "env.foot_slip_reward_scale=-0.55", 1
        )
        self.assertNotEqual(original, drifted, "the drift fixture did not apply")
        with tempfile.TemporaryDirectory() as raw:
            copy = write(Path(raw), "probe-launcher-copy", drifted)
            with self.assertRaises(compose_module.CompositionError) as raised:
                compose_module.compose_probe(
                    EXPERIMENT_CONFIG,
                    [INTERVENTION_CONFIG],
                    seed=99,
                    label="X",
                    launcher=copy,
                )
        message = str(raised.exception)
        self.assertIn("diverged", message)
        self.assertIn("env.foot_slip_reward_scale", message)
        self.assertEqual(LAUNCHER.read_text(encoding="utf-8"), original)

    def test_divergent_run_identity_refuses_composition(self) -> None:
        original = LAUNCHER.read_text(encoding="utf-8")
        drifted = original.replace("num_envs=12288", "num_envs=4096", 1)
        with tempfile.TemporaryDirectory() as raw:
            copy = write(Path(raw), "probe-launcher-copy", drifted)
            with self.assertRaises(compose_module.CompositionError) as raised:
                compose_module.compose_probe(
                    EXPERIMENT_CONFIG, [], seed=99, label="X", launcher=copy
                )
        self.assertIn("run.num_envs", str(raised.exception))


class RestrictedConfigSubsetTests(unittest.TestCase):
    """The reader accepts exactly the subset these config files may contain."""

    def test_scalars_are_read_with_the_repository_rules(self) -> None:
        document = configio.parse_config(
            "run:\n"
            "  parent_checkpoint: model_2.pt\n"
            "  num_envs: 12288\n"
            "env:\n"
            "  gate_longitudinal_reward_by_command: false\n"
            "  foot_slip_reward_scale: -0.50\n",
            "fixture.yaml",
        )
        self.assertEqual(document.section("run")["parent_checkpoint"], "model_2.pt")
        self.assertEqual(document.section("run")["num_envs"], 12288)
        self.assertIs(document.section("env")["gate_longitudinal_reward_by_command"], False)
        self.assertEqual(document.section("env")["foot_slip_reward_scale"], -0.5)
        self.assertEqual(document.token("env", "foot_slip_reward_scale"), "-0.50")

    def test_anything_outside_the_subset_is_rejected_with_a_clear_error(self) -> None:
        cases = (
            ("env:\n  nested:\n    key: 1\n", "nested mappings"),
            ("env:\n  - item\n", "sequences"),
            ("secrets:\n  token: abc\n", "unknown section"),
            ("env:\n  key: 1\n  key: 2\n", "duplicate key"),
            ("env:\n  key: 1\nenv:\n  other: 2\n", "duplicate section"),
            ("top_level: 1\n", "top-level scalars"),
            ("env:\n\tkey: 1\n", "tabs"),
            ("env:\n      key: 1\n", "unexpected indentation"),
            ("# only a comment\n", "declares no sections"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                with self.assertRaises(configio.ConfigError) as raised:
                    configio.parse_config(text, "fixture.yaml")
                self.assertIn(expected, str(raised.exception))

    def test_the_real_config_files_parse(self) -> None:
        experiment = configio.load_config(EXPERIMENT_CONFIG)
        intervention = configio.load_config(INTERVENTION_CONFIG)
        self.assertEqual(set(experiment.sections), {"run", "agent", "env"})
        self.assertEqual(set(intervention.sections), {"env"})


if __name__ == "__main__":
    unittest.main()
