"""Golden contract tests for the named Stage2C experiment config.

Three representations must agree on the Stage2C probe baseline:

* ``configs/experiment/stage2c_accel.yaml`` -- the named, reviewable source of
  the baseline;
* ``isaaclab/deploy/probe-stage2c-single-current-best`` -- the bash arrays the
  launcher actually turns into Hydra ``env.*=``/``agent.*=`` argv;
* the Probe20 zero-intervention resolved config recorded under ``artifacts/``
  -- what the trainer actually resolved when launched from this exact
  baseline.

The launcher-equality, schema-existence, and intervention tests use only the
standard library, so they run under any interpreter.  The resolved-config
golden needs PyYAML to read the trainer's own dump and skips without it.

The YAML files adapt to the launcher and the configclasses, never the other
way around: a mismatch here is reported, not papered over.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

try:  # PyYAML is optional; only the recorded-evidence golden needs it.
    import yaml
except ImportError:  # pragma: no cover - depends on the interpreter in use
    yaml = None


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

LAUNCHER = ISAACLAB_DIR / "deploy" / "probe-stage2c-single-current-best"
PACKAGE_ROOT = REPO_ROOT / "packages" / "hexapod_env" / "hexapod_env"
ENV_CFG_PATH = PACKAGE_ROOT / "env_cfg.py"
PHASE2_CFG_PATH = PACKAGE_ROOT / "phase2_cfg.py"

EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "stage2c_accel.yaml"
INTERVENTION_CONFIG = REPO_ROOT / "configs" / "intervention" / "probe21_bilateral.yaml"

PROBE_ROOT = (
    REPO_ROOT / "artifacts" / "phase2_recovery_stage2c_stable_forward" / "probes"
)
PROBE20_DIRECTORY_MARKER = "probe20_matched_zero"

# ``run:`` key -> the pinned shell variable in the launcher that owns it.
RUN_KEY_TO_LAUNCHER_VARIABLE = {
    "task_id": "task_id",
    "experiment_name": "experiment",
    "parent_run": "parent_run",
    "parent_checkpoint": "parent_checkpoint",
    "parent_sha256": "parent_sha256",
    "num_envs": "num_envs",
    "max_iterations": "max_iterations",
}

# The launcher's own finite-number grammar, used for its bash tokens.
_SHELL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_SHELL_NUMBER = re.compile(
    r"^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$"
)
# PyYAML resolves a plain scalar as a float only when it carries a decimal
# point, so the stdlib parser below uses exactly that rule and cannot disagree
# with PyYAML about any scalar in these files.
_YAML_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_YAML_FLOAT = re.compile(r"^[+-]?([0-9]+[.][0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?$")

_MISSING = object()
_OPAQUE = object()


def _shell_scalar(token: str):
    """Interpret one bash override value the way the launcher/Hydra would."""
    if token in ("true", "false"):
        return token == "true"
    if _SHELL_INTEGER.match(token):
        return int(token)
    if _SHELL_NUMBER.match(token):
        return float(token)
    return token


def _yaml_scalar(token: str):
    """Interpret one plain YAML scalar from this directory's restricted subset."""
    if token in ("true", "false"):
        return token == "true"
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    if _YAML_INTEGER.match(token):
        return int(token)
    if _YAML_FLOAT.match(token):
        return float(token)
    return token


def parse_config_yaml(path: Path) -> dict[str, dict]:
    """Parse the two-level ``section:``/``  key: value`` subset these files use.

    Deliberately minimal and stdlib-only: it accepts nothing beyond what the
    config files themselves are allowed to contain, and raises on anything
    else rather than guessing.
    """
    document: dict[str, dict] = {}
    section: dict | None = None
    section_name: str | None = None
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split(" #", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"{path.name}:{number}: expected 'key: value'")
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value:
                raise ValueError(
                    f"{path.name}:{number}: top-level scalars are not part of this subset"
                )
            if key in document:
                raise ValueError(f"{path.name}:{number}: duplicate section {key}")
            section_name = key
            section = {}
            document[key] = section
        elif indent == 2:
            if section is None:
                raise ValueError(f"{path.name}:{number}: key outside any section")
            if not value:
                raise ValueError(f"{path.name}:{number}: nested mappings are not supported")
            if key in section:
                raise ValueError(
                    f"{path.name}:{number}: duplicate key {section_name}.{key}"
                )
            section[key] = _yaml_scalar(value)
        else:
            raise ValueError(f"{path.name}:{number}: unexpected indentation {indent}")
    return document


def launcher_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"(?m)^{re.escape(name)}=\(\n(?P<body>.*?)^\)$",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing shell array {name}")
    return [
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def launcher_variable(source: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}=(?P<value>\S+)$", source)
    if match is None:
        raise AssertionError(f"missing pinned shell variable {name}")
    return match.group("value")


def launcher_overrides(tokens: list[str], prefix: str) -> dict:
    overrides: dict = {}
    for token in tokens:
        if "=" not in token:
            raise AssertionError(f"malformed override token: {token}")
        key, _, value = token.partition("=")
        if not key.startswith(f"{prefix}."):
            raise AssertionError(f"override {key} is not a {prefix}.* key")
        bare = key[len(prefix) + 1 :]
        if bare in overrides:
            raise AssertionError(f"duplicate override key: {key}")
        overrides[bare] = _shell_scalar(value)
    return overrides


def value_category(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def configclass_attribute_names(path: Path) -> set[str]:
    """Every name assigned inside a class body in one configclass module."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                names.add(statement.target.id)
    return names


class ScalarComparisonMixin:
    def assertSameScalar(self, actual, expected, message: str) -> None:
        self.assertEqual(
            value_category(actual),
            value_category(expected),
            f"{message}: type category differs ({actual!r} vs {expected!r})",
        )
        if value_category(expected) == "number":
            self.assertEqual(
                float(actual),
                float(expected),
                f"{message}: {actual!r} != {expected!r}",
            )
        else:
            self.assertEqual(actual, expected, message)


class Stage2CExperimentLauncherEqualityTests(ScalarComparisonMixin, unittest.TestCase):
    """The named config and the launcher's bash arrays must be one thing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.config = parse_config_yaml(EXPERIMENT_CONFIG)
        cls.launcher_env = launcher_overrides(
            launcher_array(cls.source, "baseline_env_overrides"), "env"
        )
        cls.launcher_agent = launcher_overrides(
            launcher_array(cls.source, "fixed_agent_overrides"), "agent"
        )

    def test_config_has_exactly_the_expected_sections(self) -> None:
        self.assertEqual(set(self.config), {"run", "agent", "env"})

    def test_env_overrides_match_the_launcher_bidirectionally(self) -> None:
        config_env = self.config["env"]
        self.assertEqual(
            set(config_env),
            set(self.launcher_env),
            "env keys differ between stage2c_accel.yaml and baseline_env_overrides",
        )
        for key in sorted(self.launcher_env):
            with self.subTest(key=f"env.{key}"):
                self.assertSameScalar(
                    config_env[key], self.launcher_env[key], f"env.{key}"
                )

    def test_agent_overrides_match_the_launcher_bidirectionally(self) -> None:
        config_agent = self.config["agent"]
        self.assertEqual(
            set(config_agent),
            set(self.launcher_agent),
            "agent keys differ between stage2c_accel.yaml and fixed_agent_overrides",
        )
        for key in sorted(self.launcher_agent):
            with self.subTest(key=f"agent.{key}"):
                self.assertSameScalar(
                    config_agent[key], self.launcher_agent[key], f"agent.{key}"
                )

    def test_run_identity_matches_the_pinned_launcher_variables(self) -> None:
        config_run = self.config["run"]
        self.assertEqual(set(config_run), set(RUN_KEY_TO_LAUNCHER_VARIABLE))
        for key, variable in sorted(RUN_KEY_TO_LAUNCHER_VARIABLE.items()):
            with self.subTest(key=f"run.{key}"):
                self.assertSameScalar(
                    config_run[key],
                    _shell_scalar(launcher_variable(self.source, variable)),
                    f"run.{key} vs launcher {variable}",
                )

    def test_numeric_equality_is_value_based_not_textual(self) -> None:
        # 2.0e-5 and 0.00002 are the same override; the comparison above must
        # not depend on how either representation spells a number.
        self.assertEqual(self.config["agent"]["algorithm.learning_rate"], 0.00002)
        self.assertSameScalar(_shell_scalar("2.0e-5"), 0.00002, "learning rate")
        self.assertSameScalar(_shell_scalar("0.040"), 0.04, "slew limit")
        self.assertSameScalar(_yaml_scalar("-160.0"), -160.0, "inactive yaw scale")
        self.assertEqual(value_category(_yaml_scalar("false")), "boolean")
        self.assertEqual(value_category(_yaml_scalar("0.0")), "number")

    def test_intervention_surface_is_pinned_by_the_named_baseline(self) -> None:
        # Every allowlisted intervention key must have a control value in the
        # named baseline, or a probe would not be a one-key delta from it.
        allowed = launcher_array(self.source, "allowed_intervention_keys")
        config_env = self.config["env"]
        for key in allowed:
            with self.subTest(key=key):
                self.assertTrue(key.startswith("env."))
                self.assertIn(key[len("env.") :], config_env)


class Stage2CConfigSchemaExistenceTests(unittest.TestCase):
    """Poor man's Hydra struct mode: no config key may be a typo."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.attributes = configclass_attribute_names(
            ENV_CFG_PATH
        ) | configclass_attribute_names(PHASE2_CFG_PATH)

    def test_experiment_env_keys_exist_on_the_configclasses(self) -> None:
        for key in sorted(parse_config_yaml(EXPERIMENT_CONFIG)["env"]):
            with self.subTest(key=key):
                self.assertIn(
                    key,
                    self.attributes,
                    f"env.{key} is not a configclass attribute in env_cfg.py or phase2_cfg.py",
                )

    def test_intervention_env_keys_exist_on_the_configclasses(self) -> None:
        for key in sorted(parse_config_yaml(INTERVENTION_CONFIG)["env"]):
            with self.subTest(key=key):
                self.assertIn(
                    key,
                    self.attributes,
                    f"env.{key} is not a configclass attribute in env_cfg.py or phase2_cfg.py",
                )

    def test_experiment_name_matches_the_stage2c_runner_configclass(self) -> None:
        tree = ast.parse(PHASE2_CFG_PATH.read_text(encoding="utf-8"))
        wanted = "HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg"
        literals = [
            ast.literal_eval(statement.value)
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == wanted
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "experiment_name"
                for target in statement.targets
            )
        ]
        self.assertEqual(len(literals), 1, f"missing {wanted}.experiment_name")
        self.assertEqual(
            literals[0], parse_config_yaml(EXPERIMENT_CONFIG)["run"]["experiment_name"]
        )


class Probe21InterventionValueTests(unittest.TestCase):
    """The Probe21 causal arm is exactly two keys with two calibrated values."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = parse_config_yaml(INTERVENTION_CONFIG)

    def test_intervention_is_only_an_env_delta(self) -> None:
        self.assertEqual(set(self.config), {"env"})

    def test_intervention_holds_exactly_the_calibrated_pair(self) -> None:
        self.assertEqual(
            self.config["env"],
            {
                "inactive_bilateral_longitudinal_contact_moment_reward_scale": -2.0,
                "inactive_bilateral_longitudinal_contact_moment_reference_nm": 1.8,
            },
        )

    def test_intervention_keys_are_launchable(self) -> None:
        allowed = set(launcher_array(LAUNCHER.read_text(encoding="utf-8"), "allowed_intervention_keys"))
        for key in sorted(self.config["env"]):
            with self.subTest(key=key):
                self.assertIn(f"env.{key}", allowed)


if yaml is not None:

    class _EvidenceLoader(yaml.SafeLoader):
        """SafeLoader that tolerates the trainer dump's ``!!python/`` tags.

        Nothing is imported, constructed, or called: python tuples become
        plain lists and every other python-tagged node becomes an opaque
        marker that no assertion reads.
        """

    def _construct_python_tuple(loader, node):
        return loader.construct_sequence(node, deep=True)

    def _construct_opaque(loader, tag_suffix, node):
        return _OPAQUE

    _EvidenceLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple", _construct_python_tuple
    )
    _EvidenceLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_opaque)


def _resolve_dotted(mapping, dotted_key: str):
    current = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _probe20_resolved_config_directory() -> Path:
    """The single Probe20 artifact that actually recorded a resolved config.

    Probe20's first launch timed out before AppReady and produced no resolved
    config; only the successful retry has one.
    """
    candidates = sorted(
        path
        for path in PROBE_ROOT.iterdir()
        if path.is_dir()
        and PROBE20_DIRECTORY_MARKER in path.name
        and (path / "resolved_config" / "env.yaml").is_file()
        and (path / "resolved_config" / "agent.yaml").is_file()
    )
    if len(candidates) != 1:
        raise AssertionError(
            "expected exactly one Probe20 artifact with a resolved config; found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0] / "resolved_config"


@unittest.skipUnless(yaml is not None, "PyYAML is required to read the recorded config")
class Probe20ResolvedConfigGoldenTests(ScalarComparisonMixin, unittest.TestCase):
    """Probe20 was the zero-intervention control launched from this baseline.

    Its recorded resolved config is what the trainer actually saw, so the
    named baseline must agree with it on every key it pins.
    """

    @classmethod
    def setUpClass(cls) -> None:
        resolved = _probe20_resolved_config_directory()
        with (resolved / "env.yaml").open(encoding="utf-8") as handle:
            cls.resolved_env = yaml.load(handle, Loader=_EvidenceLoader)
        with (resolved / "agent.yaml").open(encoding="utf-8") as handle:
            cls.resolved_agent = yaml.load(handle, Loader=_EvidenceLoader)
        cls.config = parse_config_yaml(EXPERIMENT_CONFIG)

    def test_every_baseline_env_key_matches_the_resolved_env(self) -> None:
        for key, expected in sorted(self.config["env"].items()):
            with self.subTest(key=key):
                actual = _resolve_dotted(self.resolved_env, key)
                self.assertIsNot(
                    actual, _MISSING, f"resolved env.yaml does not record {key}"
                )
                self.assertSameScalar(actual, expected, f"resolved env.{key}")

    def test_every_baseline_agent_override_matches_the_resolved_agent(self) -> None:
        for key, expected in sorted(self.config["agent"].items()):
            with self.subTest(key=key):
                actual = _resolve_dotted(self.resolved_agent, key)
                self.assertIsNot(
                    actual, _MISSING, f"resolved agent.yaml does not record {key}"
                )
                self.assertSameScalar(actual, expected, f"resolved agent.{key}")

    def test_run_identity_matches_the_resolved_dumps(self) -> None:
        run = self.config["run"]
        self.assertSameScalar(
            _resolve_dotted(self.resolved_agent, "max_iterations"),
            run["max_iterations"],
            "resolved agent.max_iterations",
        )
        self.assertSameScalar(
            _resolve_dotted(self.resolved_agent, "experiment_name"),
            run["experiment_name"],
            "resolved agent.experiment_name",
        )
        self.assertSameScalar(
            _resolve_dotted(self.resolved_env, "scene.num_envs"),
            run["num_envs"],
            "resolved env.scene.num_envs",
        )
        # The launcher builds its resume regexes from these two pinned values.
        self.assertEqual(
            _resolve_dotted(self.resolved_agent, "load_run"),
            "^" + run["parent_run"].replace(".", "[.]") + "$",
        )
        self.assertEqual(
            _resolve_dotted(self.resolved_agent, "load_checkpoint"),
            "^" + run["parent_checkpoint"].replace(".", "[.]") + "$",
        )


if __name__ == "__main__":
    unittest.main()
