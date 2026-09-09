"""Bind `hexapod_eval.gates` to the sources that actually enforce each number.

`packages/hexapod_eval/hexapod_eval/gates.py` is a mirror, not an authority.
These tests re-extract every mirrored value from the grader, launcher, or prose
ledger it claims to come from, so a threshold that drifts on either side fails
here instead of diverging silently.

Everything except `gates.py` itself is read with stdlib `ast` or `re`: no
grader is imported, so these contracts hold without Isaac Lab, torch, or a GPU.
`gates.py` is dependency-free and is loaded straight from its file path, so no
`PYTHONPATH` is required either.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import sys
import unittest


ISAACLAB = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "packages" / "hexapod_eval"
PROBES_README = (
    REPO_ROOT
    / "artifacts"
    / "phase2_recovery_stage2c_stable_forward"
    / "probes"
    / "README.md"
)
SHARDED_SCREEN = ISAACLAB / "deploy" / "screen-stage2c-probe-sharded"

DISPATCHED_SCRIPT_PREFIXES = ("grade_", "analyze_", "merge_", "summarize_")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = _load(
    "hexapod_eval_gates_under_test", PACKAGE_ROOT / "hexapod_eval" / "gates.py"
)
dispatch = _load(
    "hexapod_eval_dispatch_under_test",
    PACKAGE_ROOT / "hexapod_eval" / "dispatch.py",
)


# ---------------------------------------------------------------------------
# Minimal literal evaluator over a module's own top-level constants
# ---------------------------------------------------------------------------


class _Unresolved(Exception):
    """Raised when a node is not a literal built from known constants."""


def _value(node: ast.AST, names: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise _Unresolved(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _value(node.operand, names)
        if not isinstance(operand, (int, float)):
            raise _Unresolved("unary minus on non-number")
        return -operand
    if isinstance(node, ast.BinOp):
        left = _value(node.left, names)
        right = _value(node.right, names)
        if not isinstance(left, (int, float)) or not isinstance(
            right, (int, float)
        ):
            raise _Unresolved("binary op on non-number")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise _Unresolved("unsupported binary operator")
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_value(element, names) for element in node.elts)
    if isinstance(node, ast.Dict):
        resolved: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise _Unresolved("dict unpacking")
            resolved[_value(key_node, names)] = _value(value_node, names)
        return resolved
    raise _Unresolved(type(node).__name__)


def _constants(path: Path, seed: dict[str, object] | None = None) -> dict[str, object]:
    """Return the module-level constants of `path` that are plain literals."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: dict[str, object] = dict(seed or {})
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        try:
            names[targets[0].id] = _value(value, names)
        except _Unresolved:
            continue
    return names


def _dict_entries(path: Path, name: str) -> dict[object, object]:
    """Return the entries of a module-level dict literal that are resolvable.

    Some pinned snapshots mix plain literals with imported names and computed
    expressions.  Only the literal entries are returned; the rest are skipped
    rather than guessed at.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = _constants(path)
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        if targets[0].id != name:
            continue
        assert isinstance(value, ast.Dict), f"{name} is not a dict literal"
        entries: dict[object, object] = {}
        for key_node, value_node in zip(value.keys, value.values):
            if key_node is None:
                continue
            try:
                entries[_value(key_node, names)] = _value(value_node, names)
            except _Unresolved:
                continue
        return entries
    raise AssertionError(f"{name} not found in {path}")


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    """Collapse the prose ledger's hard line wrapping to single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _component_targets(components: object) -> dict[str, float]:
    """Turn a `(key, display, unit, target)` tuple-of-tuples into a target map."""

    assert isinstance(components, tuple), components
    targets: dict[str, float] = {}
    for row in components:
        assert isinstance(row, tuple) and len(row) == 4, row
        targets[str(row[0])] = float(row[3])
    return targets


def _paragraph(text: str, needle: str) -> str:
    for block in text.split("\n\n"):
        normalized = _normalize(block)
        if needle in normalized:
            return normalized
    raise AssertionError(f"paragraph containing {needle!r} not found")


STAGE2C_GRADER = ISAACLAB / "grade_stage2c_stable_forward.py"
STAGE2C_CONSTANTS = _constants(STAGE2C_GRADER)


# ---------------------------------------------------------------------------
# (a) Stage 2C gate values against the grader that enforces them
# ---------------------------------------------------------------------------


class Stage2CAdmissionGateBindingTest(unittest.TestCase):
    """Every Stage2C admission number equals the grader literal behind it."""

    def test_yaw_rate_rmse_gate_matches_the_tail_component_target(self) -> None:
        targets = _component_targets(
            STAGE2C_CONSTANTS["TAIL_STABILITY_COMPONENTS"]
        )
        self.assertEqual(
            gates.STAGE2C_ADMISSION.maximum_yaw_rate_rmse_radps,
            targets["yaw_rate_rmse_radps"],
        )

    def test_moving_deck_composite_gate_matches_the_grader_comparison(self) -> None:
        # `stability_composite <= 1.0 + GATE_ABS_TOLERANCE` inside
        # `_grade_operational_row` is where the composite limit is written down.
        row = _function(_module_tree(STAGE2C_GRADER), "_grade_operational_row")
        limits: list[float] = []
        for node in ast.walk(row):
            if not isinstance(node, ast.Compare):
                continue
            if not (
                isinstance(node.left, ast.Name)
                and node.left.id == "stability_composite"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.LtE)
            ):
                continue
            comparator = node.comparators[0]
            self.assertIsInstance(comparator, ast.BinOp)
            self.assertIsInstance(comparator.left, ast.Constant)
            limits.append(float(comparator.left.value))
        self.assertEqual(len(limits), 1, "expected exactly one composite limit")
        self.assertEqual(
            gates.STAGE2C_ADMISSION.maximum_moving_normalized_deck_composite,
            limits[0],
        )

    def test_moving_composite_normalizers_are_the_absolute_rms_targets(self) -> None:
        self.assertEqual(
            dict(gates.STAGE2C_STABLE_FORWARD.moving_rms_targets),
            _component_targets(STAGE2C_CONSTANTS["STABILITY_COMPONENTS"]),
        )
        self.assertEqual(
            dict(gates.STAGE2C_STABLE_FORWARD.stand_rms_targets),
            _component_targets(STAGE2C_CONSTANTS["STAND_STABILITY_COMPONENTS"]),
        )
        self.assertEqual(
            dict(gates.STAGE2C_STABLE_FORWARD.tail_targets),
            _component_targets(STAGE2C_CONSTANTS["TAIL_STABILITY_COMPONENTS"]),
        )

    def test_zero_falls_and_zero_timeouts_are_still_enforced(self) -> None:
        row = _function(_module_tree(STAGE2C_GRADER), "_grade_operational_row")
        guarded = {
            node.test.id
            for node in ast.walk(row)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        }
        self.assertIn("falls", guarded)
        self.assertIn("timeouts", guarded)
        source = STAGE2C_GRADER.read_text(encoding="utf-8")
        self.assertIn('f"falls={falls}, expected 0"', source)
        self.assertIn('f"timeouts={timeouts}, expected 0"', source)
        self.assertEqual(gates.STAGE2C_ADMISSION.maximum_falls, 0)
        self.assertEqual(gates.STAGE2C_ADMISSION.maximum_timeouts, 0)

    def test_forward_command_fractions_match_the_grader_thresholds(self) -> None:
        thresholds = STAGE2C_CONSTANTS["THRESHOLDS"]
        self.assertEqual(
            gates.STAGE2C_ADMISSION.minimum_forward_command_fraction,
            thresholds["minimum_forward_command_fraction"],
        )
        self.assertEqual(
            gates.STAGE2C_ADMISSION.maximum_forward_command_fraction,
            thresholds["maximum_forward_command_fraction"],
        )

    def test_headline_forward_floor_is_the_grader_product(self) -> None:
        contract = dict(
            (str(label), tuple(float(value) for value in command))
            for label, command in STAGE2C_CONSTANTS["COMMAND_CONTRACT"]
        )
        commanded_vx = contract["forward_0p30"][0]
        thresholds = STAGE2C_CONSTANTS["THRESHOLDS"]
        expected = commanded_vx * thresholds["minimum_forward_command_fraction"]
        self.assertEqual(
            gates.STAGE2C_ADMISSION.headline_forward_command_mps, commanded_vx
        )
        self.assertAlmostEqual(
            gates.STAGE2C_ADMISSION.minimum_achieved_forward_mps_at_headline_command,
            expected,
            places=12,
        )
        self.assertAlmostEqual(expected, 0.240, places=12)

    def test_rs05_safety_set_matches_the_pinned_config_snapshot(self) -> None:
        snapshot = _dict_entries(
            ISAACLAB / "grade_stage2_command_transitions.py",
            "EXPECTED_CONFIG_SNAPSHOT",
        )
        rs05 = gates.RS05_SAFETY
        self.assertEqual(rs05.continuous_rated_torque_nm, snapshot["rated_torque_nm"])
        self.assertEqual(
            rs05.continuous_rated_torque_nm, snapshot["actuator_effort_limit_nm"]
        )
        self.assertEqual(
            rs05.raw_demand_termination_nm,
            snapshot["terminate_on_computed_torque_demand_nm"],
        )
        self.assertEqual(
            rs05.raw_demand_termination_duration_s,
            snapshot["terminate_on_computed_torque_demand_duration_s"],
        )
        self.assertEqual(
            rs05.raw_demand_termination_grace_s,
            snapshot["torque_demand_termination_grace_s"],
        )
        self.assertIs(gates.STAGE2C_ADMISSION.rs05, rs05)


# ---------------------------------------------------------------------------
# (b) The prose ledger under artifacts/ (append-only evidence, read-only here)
# ---------------------------------------------------------------------------


class ProbeLedgerProseTest(unittest.TestCase):
    """The probe README's acceptance paragraph must state the same numbers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = PROBES_README.read_text(encoding="utf-8")

    def test_acceptance_paragraph_numbers_match_gates(self) -> None:
        paragraph = _paragraph(self.text, "The acceptance goals remain")
        yaw = re.search(
            r"yaw-rate RMSE at or below ([0-9]*\.?[0-9]+) rad/s", paragraph
        )
        deck = re.search(
            r"moving normalized deck composite at or below ([0-9]*\.?[0-9]+)",
            paragraph,
        )
        forward = re.search(
            r"achieved forward velocity at least ([0-9]*\.?[0-9]+) m/s", paragraph
        )
        self.assertIsNotNone(yaw, paragraph)
        self.assertIsNotNone(deck, paragraph)
        self.assertIsNotNone(forward, paragraph)
        self.assertEqual(
            float(yaw.group(1)),
            gates.STAGE2C_ADMISSION.maximum_yaw_rate_rmse_radps,
        )
        self.assertEqual(
            float(deck.group(1)),
            gates.STAGE2C_ADMISSION.maximum_moving_normalized_deck_composite,
        )
        self.assertAlmostEqual(
            float(forward.group(1)),
            gates.STAGE2C_ADMISSION.minimum_achieved_forward_mps_at_headline_command,
            places=12,
        )

    def test_acceptance_paragraph_still_demands_zero_falls_and_timeouts(self) -> None:
        paragraph = _paragraph(self.text, "The acceptance goals remain")
        self.assertIn("zero falls/timeouts", paragraph)
        self.assertEqual(gates.STAGE2C_ADMISSION.maximum_falls, 0)
        self.assertEqual(gates.STAGE2C_ADMISSION.maximum_timeouts, 0)

    def test_formal_screen_paragraph_matches_the_canonical_screen(self) -> None:
        paragraph = _paragraph(self.text, "All formal probe screens use")
        screen = gates.CANONICAL_FORMAL_SCREEN
        commands = tuple(
            tuple(float(part) for part in match)
            for match in re.findall(
                r"\(\s*(-?[0-9]*\.?[0-9]+),\s*(-?[0-9]*\.?[0-9]+),"
                r"\s*(-?[0-9]*\.?[0-9]+)\s*\)",
                paragraph,
            )
        )
        self.assertEqual(commands, screen.commands)
        seed = re.search(r"seed (\d+)", paragraph)
        warmup = re.search(r"(\d+) warmup steps", paragraph)
        requested = re.search(r"(\d+) requested steps", paragraph)
        limiter = re.search(
            r"([0-9]*\.?[0-9]+) rad/20 ms final target limiter", paragraph
        )
        for match, label in (
            (seed, "seed"),
            (warmup, "warmup steps"),
            (requested, "requested steps"),
            (limiter, "limiter"),
        ):
            self.assertIsNotNone(match, f"{label} not found in: {paragraph}")
        self.assertEqual(int(seed.group(1)), screen.seed)
        self.assertEqual(int(warmup.group(1)), screen.warmup_steps)
        self.assertEqual(int(requested.group(1)), screen.requested_steps)
        self.assertEqual(
            float(limiter.group(1)),
            screen.processed_joint_target_slew_limit_rad_per_20ms,
        )


# ---------------------------------------------------------------------------
# (c) Screen parameters against the evaluator/analyzer/launcher sources
# ---------------------------------------------------------------------------


class ScreenParameterBindingTest(unittest.TestCase):
    """Formal and diagnostic screen parameters come from real literals."""

    def test_canonical_screen_matches_the_grader_playback_contract(self) -> None:
        screen = gates.CANONICAL_FORMAL_SCREEN
        self.assertEqual(screen.seed, STAGE2C_CONSTANTS["EXPECTED_SEED"])
        self.assertEqual(
            screen.warmup_steps, STAGE2C_CONSTANTS["EXPECTED_WARMUP_STEPS"]
        )
        self.assertEqual(
            screen.requested_steps, STAGE2C_CONSTANTS["EXPECTED_REQUESTED_STEPS"]
        )
        self.assertEqual(
            screen.policy_step_seconds,
            STAGE2C_CONSTANTS["EXPECTED_POLICY_STEP_SECONDS"],
        )
        self.assertEqual(
            screen.post_warmup_samples, STAGE2C_CONSTANTS["EXPECTED_SAMPLES"]
        )
        self.assertAlmostEqual(
            screen.measured_seconds,
            STAGE2C_CONSTANTS["EXPECTED_MEASURED_SECONDS"],
            places=12,
        )
        self.assertEqual(screen.post_warmup_samples, 475)
        self.assertAlmostEqual(
            screen.requested_steps * screen.policy_step_seconds, 10.0, places=12
        )
        self.assertTrue(screen.formal_admission_eligible)

    def test_canonical_screen_commands_match_the_grader_contract(self) -> None:
        contract = tuple(
            tuple(float(value) for value in command)
            for _, command in STAGE2C_CONSTANTS["COMMAND_CONTRACT"]
        )
        self.assertEqual(gates.CANONICAL_FORMAL_SCREEN.commands, contract)

    def test_playback_limiter_matches_the_merger_and_the_launcher(self) -> None:
        merger = _constants(ISAACLAB / "merge_stage2c_probe_shards.py")[
            "EXPECTED_ACTION_PROCESSING"
        ]
        limiter = gates.CANONICAL_FORMAL_SCREEN.processed_joint_target_slew_limit_rad_per_20ms
        self.assertEqual(
            limiter, merger["processed_joint_target_slew_limit_rad_per_20ms"]
        )
        launcher = SHARDED_SCREEN.read_text(encoding="utf-8")
        match = re.search(
            r"--processed-joint-target-slew-limit-rad-per-20ms\s+"
            r"([0-9]*\.?[0-9]+)",
            launcher,
        )
        self.assertIsNotNone(match, "sharded screen launcher sets no limiter")
        self.assertEqual(float(match.group(1)), limiter)
        self.assertEqual(
            gates.DIAGNOSTIC_SHORT_SCREEN.processed_joint_target_slew_limit_rad_per_20ms,
            limiter,
        )

    def test_diagnostic_screen_is_never_formal_admission_eligible(self) -> None:
        analyzer = ISAACLAB / "analyze_stage2c_probe_sweep.py"
        contract_fn = _function(_module_tree(analyzer), "_sample_contract")
        eligibility: dict[str, bool] = {}
        for node in ast.walk(contract_fn):
            if not isinstance(node, ast.Dict):
                continue
            entries = {
                key.value: value
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant)
            }
            mode = entries.get("mode")
            flag = entries.get("formal_admission_eligible")
            if isinstance(mode, ast.Constant) and isinstance(flag, ast.Constant):
                eligibility[str(mode.value)] = bool(flag.value)
        self.assertEqual(
            eligibility,
            {"formal_admission": True, "diagnostic_short_duration": False},
        )
        self.assertFalse(gates.DIAGNOSTIC_SHORT_SCREEN.formal_admission_eligible)

    def test_diagnostic_screen_shape_matches_the_analyzer_help_text(self) -> None:
        analyzer = (ISAACLAB / "analyze_stage2c_probe_sweep.py").read_text(
            encoding="utf-8"
        )
        match = re.search(r"(\d+) samples from a (\d+) s run", analyzer)
        self.assertIsNotNone(match, "analyzer help text names no short screen")
        samples = int(match.group(1))
        seconds = float(match.group(2))
        screen = gates.DIAGNOSTIC_SHORT_SCREEN
        self.assertEqual(screen.post_warmup_samples, samples)
        self.assertAlmostEqual(
            screen.requested_steps * screen.policy_step_seconds, seconds, places=12
        )
        # The analyzer builds the diagnostic contract from the formal warmup.
        self.assertEqual(
            screen.requested_steps,
            screen.post_warmup_samples + gates.CANONICAL_FORMAL_SCREEN.warmup_steps,
        )
        self.assertAlmostEqual(
            screen.measured_seconds,
            screen.post_warmup_samples * screen.policy_step_seconds,
            places=12,
        )
        self.assertLess(
            screen.post_warmup_samples,
            gates.CANONICAL_FORMAL_SCREEN.post_warmup_samples,
        )


# ---------------------------------------------------------------------------
# Per-stage threshold mirrors
# ---------------------------------------------------------------------------


class StageThresholdMirrorTest(unittest.TestCase):
    """Each mirrored stage table equals its grader's own THRESHOLDS dict."""

    # stage key -> (grader file, seed constants imported from elsewhere)
    GRADERS = {
        "stage1-recovery": ("grade_stage1_recovery.py", {}),
        "stage2-axis-acquisition": ("grade_stage2_axis_acquisition.py", {}),
        "stage2b-lateral-acquisition": (
            "grade_stage2b_lateral_acquisition.py",
            {},
        ),
        "stage2c-stable-forward": ("grade_stage2c_stable_forward.py", {}),
        "stage2c-robustness": ("grade_stage2c_robustness.py", {}),
        "stage2d-homotopy": ("grade_stage2d_homotopy.py", {}),
        "stage2e-static": ("grade_stage2e_static.py", {}),
        "stage2-command-transitions": (
            "grade_stage2_command_transitions.py",
            {"MAXIMUM_SETTLING_TIME_SECONDS": None},
        ),
    }

    def _seed(self, seed: dict[str, object]) -> dict[str, object]:
        if not seed:
            return {}
        imported = _constants(ISAACLAB / "stage2_command_transition_contract.py")
        return {name: imported[name] for name in seed}

    def test_every_mirrored_stage_matches_its_grader_thresholds(self) -> None:
        for stage, (filename, seed) in self.GRADERS.items():
            with self.subTest(stage=stage):
                constants = _constants(ISAACLAB / filename, self._seed(seed))
                self.assertEqual(
                    dict(gates.STAGES[stage].thresholds),
                    constants["THRESHOLDS"],
                )

    def test_mirrored_command_contracts_match_their_graders(self) -> None:
        for stage, (filename, seed) in self.GRADERS.items():
            mirrored = gates.STAGES[stage].command_contract
            if mirrored is None:
                continue
            with self.subTest(stage=stage):
                constants = _constants(ISAACLAB / filename, self._seed(seed))
                grader_contract = tuple(
                    (str(label), tuple(float(value) for value in command))
                    for label, command in constants["COMMAND_CONTRACT"]
                )
                self.assertEqual(mirrored, grader_contract)

    def test_mirrored_component_targets_match_their_graders(self) -> None:
        cases = (
            (
                "stage2c-robustness",
                "grade_stage2c_robustness.py",
                {
                    "moving_rms_targets": "MOVING_RMS_COMPONENTS",
                    "stand_rms_targets": "STAND_RMS_COMPONENTS",
                    "tail_targets": "TAIL_COMPONENTS",
                },
            ),
            (
                "stage2d-homotopy",
                "grade_stage2d_homotopy.py",
                {
                    "moving_rms_targets": "RMS_STABILITY_COMPONENTS",
                    "tail_targets": "TAIL_STABILITY_COMPONENTS",
                },
            ),
            (
                "stage2e-static",
                "grade_stage2e_static.py",
                {"stand_rms_targets": "STAND_RMS_STABILITY_COMPONENTS"},
            ),
            (
                "stage2-command-transitions",
                "grade_stage2_command_transitions.py",
                {
                    "moving_rms_targets": "RMS_MOVING_COMPONENTS",
                    "stand_rms_targets": "RMS_STAND_COMPONENTS",
                    "tail_targets": "TAIL_COMPONENTS",
                },
            ),
        )
        for stage, filename, fields in cases:
            constants = _constants(ISAACLAB / filename)
            for field_name, constant_name in fields.items():
                with self.subTest(stage=stage, field=field_name):
                    self.assertEqual(
                        dict(getattr(gates.STAGES[stage], field_name)),
                        _component_targets(constants[constant_name]),
                    )

    def test_stance_validation_mirrors_its_named_constants(self) -> None:
        constants = _constants(ISAACLAB / "summarize_stance_validation.py")
        thresholds = gates.STANCE_VALIDATION.thresholds
        self.assertEqual(
            thresholds["applied_torque_limit_nm"],
            constants["APPLIED_TORQUE_LIMIT_NM"],
        )
        self.assertEqual(
            thresholds["computed_torque_peak_limit_nm"],
            constants["COMPUTED_TORQUE_PEAK_LIMIT_NM"],
        )
        self.assertEqual(
            thresholds["saturation_fraction_limit"],
            constants["SATURATION_FRACTION_LIMIT"],
        )
        self.assertEqual(
            thresholds["minimum_base_height_m"], constants["MINIMUM_BASE_HEIGHT_M"]
        )

    def test_stage_tables_are_read_only_and_self_consistent(self) -> None:
        for stage, table in gates.STAGES.items():
            with self.subTest(stage=stage):
                self.assertEqual(table.stage, stage)
                self.assertTrue((REPO_ROOT / table.source).is_file(), table.source)
                with self.assertRaises(TypeError):
                    table.thresholds["injected"] = 1.0


# ---------------------------------------------------------------------------
# (d) Dispatcher mapping
# ---------------------------------------------------------------------------


class DispatcherMappingTest(unittest.TestCase):
    """The dispatcher points only at real scripts, and misses none of them."""

    def test_every_mapped_script_exists_and_is_python(self) -> None:
        for (action, stage), relative in dispatch.MAPPING.items():
            with self.subTest(action=action, stage=stage):
                path = ISAACLAB / relative
                self.assertTrue(path.is_file(), path)
                self.assertFalse(path.is_symlink(), path)
                self.assertEqual(path.suffix, ".py", path)
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertTrue(
                    first_line.startswith("#!") or path.suffix == ".py",
                    f"{path} is neither executable nor a python module",
                )

    def test_every_dispatchable_script_appears_in_the_mapping(self) -> None:
        on_disk = {
            path.name
            for path in ISAACLAB.glob("*.py")
            if path.name.startswith(DISPATCHED_SCRIPT_PREFIXES)
        }
        mapped = set(dispatch.MAPPING.values())
        self.assertTrue(on_disk, "no dispatchable scripts found")
        self.assertEqual(
            on_disk - mapped,
            set(),
            "evaluation scripts missing from the dispatcher mapping",
        )

    def test_script_root_resolves_to_the_repository_isaaclab_directory(self) -> None:
        self.assertEqual(dispatch.script_root().resolve(), ISAACLAB)

    def test_mapping_keys_are_unique_pairs_and_resolve(self) -> None:
        self.assertEqual(len(dispatch.MAPPING), len(set(dispatch.MAPPING)))
        for (action, stage), relative in dispatch.MAPPING.items():
            with self.subTest(action=action, stage=stage):
                self.assertEqual(
                    dispatch.script_path(action, stage), ISAACLAB / relative
                )

    def test_unknown_pair_fails_with_the_table(self) -> None:
        with self.assertRaises(dispatch.DispatchError) as caught:
            dispatch.script_path("grade", "no-such-stage")
        message = str(caught.exception)
        self.assertIn("no-such-stage", message)
        self.assertIn("grade_stage2c_stable_forward.py", message)

    def test_argv_after_the_stage_is_forwarded_untouched(self) -> None:
        script, forwarded = dispatch.resolve(
            [
                "grade",
                "--stage",
                "stage2d-homotopy",
                "batch.json",
                "--stage",
                "C0",
                "--run-name",
                "run",
            ]
        )
        self.assertEqual(script, ISAACLAB / "grade_stage2d_homotopy.py")
        self.assertEqual(
            forwarded, ["batch.json", "--stage", "C0", "--run-name", "run"]
        )

    def test_equals_form_of_the_stage_flag_is_accepted(self) -> None:
        script, forwarded = dispatch.resolve(
            ["analyze", "--stage=stage2c-probe-sweep", "sweep.json"]
        )
        self.assertEqual(script, ISAACLAB / "analyze_stage2c_probe_sweep.py")
        self.assertEqual(forwarded, ["sweep.json"])

    def test_missing_or_misplaced_stage_flag_is_refused(self) -> None:
        for argv in ([], ["grade"], ["grade", "input.json"], ["--stage", "x"]):
            with self.subTest(argv=argv):
                with self.assertRaises(dispatch.DispatchError):
                    dispatch.parse(argv)

    def test_table_lists_every_mapping_row(self) -> None:
        table = dispatch.format_table()
        for (action, stage), relative in dispatch.MAPPING.items():
            self.assertIn(stage, table)
            self.assertIn(f"isaaclab/{relative}", table)
        self.assertEqual(
            len(table.splitlines()), len(dispatch.MAPPING) + 1, table
        )

    def test_every_dispatcher_stage_that_names_a_grader_has_gates(self) -> None:
        for (action, stage), relative in dispatch.MAPPING.items():
            if action != "grade":
                continue
            with self.subTest(stage=stage):
                self.assertIn(stage, gates.STAGES)
                self.assertEqual(
                    gates.STAGES[stage].source, f"isaaclab/{relative}"
                )


if __name__ == "__main__":
    unittest.main()
