"""Static contracts for the hexapod_env package and its hexapod_rl shims.

The training package physically lives in ``packages/hexapod_env/hexapod_env``.
``isaaclab/hexapod_rl`` stays behind as pure re-export shims so the Spark
deployment, the frozen gym entry-point strings, and every historical
``import hexapod_rl`` keep resolving. Everything here is stdlib ``ast`` only:
nothing is imported, so these contracts hold without Isaac Lab, gymnasium, or a
GPU.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_DIR = ROOT / "hexapod_rl"
PACKAGE_ROOT = REPO_ROOT / "packages" / "hexapod_env" / "hexapod_env"
REWARDS_ROOT = PACKAGE_ROOT / "rewards"

# The module names that existed under ``hexapod_rl`` before the split. Every one
# of them must keep a shim; nothing else may.
HISTORICAL_MODULES = frozenset(
    {
        "asset_cfg",
        "command_sampling",
        "env",
        "env_cfg",
        "phase1_v2_cfg",
        "phase1_v3_cfg",
        "phase1_v4_cfg",
        "phase1_v5_cfg",
        "phase2_cfg",
        "phase2d_cfg",
        "phase2e_cfg",
        "phase2g_cfg",
        "ppo_cfg",
        "register",
        "showcase_sequence",
    }
)

# The pure shaping helpers moved out of ``env.py``, keyed by themed module.
REWARDS_LAYOUT = {
    "gait": (
        "assign_tripod_pairs_from_foot_offsets",
        "tripod_expected_stance",
        "gait_phase_contact_reward",
        "swing_clearance_reward",
        "advance_gait_phase",
    ),
    "contact_wrench": (
        "bounded_inactive_ground_contact_yaw_moment_cost",
        "bounded_inactive_bilateral_longitudinal_contact_moment_cost",
        "reset_batch_time_mean_and_p50",
    ),
    "torque": (
        "applied_torque_slew_l2",
        "max_joint_rated_torque_excess_l2",
        "max_joint_rated_torque_excess_l1",
    ),
    "stability": (
        "normalized_deck_stability_reward",
        "select_command_conditioned_nominal_height",
        "select_support_contact_target",
    ),
    "axes": (
        "reset_safe_exponential_moving_average",
        "capped_normalized_axis_error",
        "bounded_inactive_yaw_rate_slew_cost",
    ),
    "actions": (
        "apply_command_conditioned_stand_action_scale",
        "limit_processed_joint_target_slew",
    ),
}

ALLOWED_REWARDS_IMPORTS = frozenset({"__future__", "math", "torch", "typing"})

ENTRY_POINT_RE = re.compile(r"^hexapod_rl\.(\w+):(\w+)$")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _reward_module_paths() -> list[Path]:
    return sorted(
        path for path in REWARDS_ROOT.glob("*.py") if path.name != "__init__.py"
    )


def _top_level_functions(tree: ast.Module) -> list[str]:
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _dunder_all(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    return None


def _string_tables(tree: ast.Module) -> dict[str, list[list[str | None]]]:
    """Map ``name -> rows`` for assignments of a tuple/list of tuples.

    Elements that are not plain string literals become ``None``; the Stage2D and
    Stage2E registration tables mix a task-id ``Name`` with two class-name
    strings.
    """

    tables: dict[str, list[list[str | None]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, (ast.Tuple, ast.List)
        ):
            continue
        rows: list[list[str | None]] = []
        for row in node.value.elts:
            if not isinstance(row, (ast.Tuple, ast.List)):
                rows = []
                break
            rows.append(
                [
                    element.value
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                    else None
                    for element in row.elts
                ]
            )
        if not rows:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                tables[target.id] = rows
    return tables


def _loop_variable_values(
    tree: ast.Module, loop: ast.For
) -> dict[str, set[str]]:
    """Map this ``for`` loop's tuple-target names to the strings they take."""

    if not isinstance(loop.target, ast.Tuple) or not isinstance(loop.iter, ast.Name):
        return {}
    rows = _string_tables(tree).get(loop.iter.id)
    if rows is None:
        return {}
    values: dict[str, set[str]] = {}
    for index, element in enumerate(loop.target.elts):
        if not isinstance(element, ast.Name):
            continue
        column = {
            row[index]
            for row in rows
            if index < len(row) and isinstance(row[index], str)
        }
        if column:
            values[element.id] = column
    return values


def _formatted_pairs(
    node: ast.JoinedStr, values: dict[str, set[str]]
) -> set[tuple[str, str]]:
    prefix = "".join(
        part.value
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )
    if not prefix.startswith("hexapod_rl.") or ":" not in prefix:
        return set()
    module = prefix[len("hexapod_rl.") : prefix.index(":")]
    pairs: set[tuple[str, str]] = set()
    for part in node.values:
        if isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
            for class_name in values.get(part.value.id, set()):
                pairs.add((module, class_name))
    return pairs


def _entry_point_pairs(tree: ast.Module) -> set[tuple[str, str]]:
    """Return every ``(module, class)`` an entry-point string can resolve to.

    Plain string literals resolve directly. The Stage2D and Stage2E task loops
    build their entry points with f-strings, so those are resolved against the
    registration table their own ``for`` loop iterates over.
    """

    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = ENTRY_POINT_RE.match(node.value)
            if match:
                pairs.add((match.group(1), match.group(2)))
        elif isinstance(node, ast.For):
            values = _loop_variable_values(tree, node)
            for inner in ast.walk(node):
                if isinstance(inner, ast.JoinedStr):
                    pairs |= _formatted_pairs(inner, values)
    return pairs


class ShimCoverageTest(unittest.TestCase):
    def test_shims_exist_exactly_for_the_historical_modules(self):
        shims = {
            path.stem
            for path in SHIM_DIR.glob("*.py")
            if path.name != "__init__.py"
        }
        self.assertEqual(shims, set(HISTORICAL_MODULES))
        self.assertTrue((SHIM_DIR / "__init__.py").is_file())

    def test_every_historical_module_still_has_an_implementation(self):
        for module in sorted(HISTORICAL_MODULES):
            self.assertTrue(
                (PACKAGE_ROOT / f"{module}.py").is_file(),
                f"missing implementation for {module}",
            )

    def test_new_modules_need_no_shim(self):
        implementations = {
            path.stem
            for path in PACKAGE_ROOT.glob("*.py")
            if path.name != "__init__.py"
        }
        self.assertEqual(implementations - set(HISTORICAL_MODULES), set())
        for path in _reward_module_paths():
            self.assertFalse((SHIM_DIR / path.name).exists())

    def test_shim_all_matches_the_source_module(self):
        for module in sorted(HISTORICAL_MODULES):
            shim_all = _dunder_all(_tree(SHIM_DIR / f"{module}.py"))
            self.assertIsNotNone(shim_all, f"{module} shim declares no __all__")
            source_tree = _tree(PACKAGE_ROOT / f"{module}.py")
            source_all = _dunder_all(source_tree)
            if source_all is not None:
                self.assertEqual(shim_all, source_all, module)
            imported = _imported_names(_tree(SHIM_DIR / f"{module}.py"))
            for name in shim_all or []:
                self.assertIn(name, imported, f"{module}.{name} not re-exported")


class ShimsCarryNoLogicTest(unittest.TestCase):
    ALLOWED_TOP_LEVEL = (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
    FORBIDDEN = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
    )

    def shim_paths(self):
        return sorted(SHIM_DIR.glob("*.py"))

    def test_shims_contain_only_a_docstring_imports_and_assignments(self):
        for path in self.shim_paths():
            tree = _tree(path)
            for index, node in enumerate(tree.body):
                self.assertIsInstance(node, self.ALLOWED_TOP_LEVEL, f"{path.name}")
                if isinstance(node, ast.Expr):
                    self.assertEqual(index, 0, f"{path.name}: stray expression")
                    self.assertIsInstance(node.value, ast.Constant)
                    self.assertIsInstance(node.value.value, str)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.assertIsInstance(target, (ast.Name, ast.Subscript))

    def test_shims_define_no_functions_or_classes(self):
        for path in self.shim_paths():
            for node in ast.walk(_tree(path)):
                self.assertNotIsInstance(node, self.FORBIDDEN, f"{path.name}")

    # ``sys`` and ``pathlib`` are the package __init__'s sys.path bootstrap.
    ALLOWED_SHIM_IMPORTS = frozenset({"hexapod_env", "sys", "pathlib"})

    def test_shims_import_only_from_hexapod_env_or_the_bootstrap(self):
        for path in self.shim_paths():
            for node in _tree(path).body:
                if isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                elif isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                    self.assertTrue(
                        roots <= self.ALLOWED_SHIM_IMPORTS,
                        f"{path.name} imports {sorted(roots)}",
                    )
                    continue
                else:
                    continue
                self.assertIn(root, self.ALLOWED_SHIM_IMPORTS, path.name)

    def test_submodule_shims_import_only_from_hexapod_env(self):
        for path in self.shim_paths():
            if path.name == "__init__.py":
                continue
            for node in _tree(path).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.assertIsInstance(node, ast.ImportFrom, path.name)
                    assert isinstance(node, ast.ImportFrom)
                    self.assertEqual(
                        node.module, f"hexapod_env.{path.stem}", path.name
                    )


class EntryPointResolutionTest(unittest.TestCase):
    def setUp(self):
        self.pairs = _entry_point_pairs(_tree(PACKAGE_ROOT / "register.py"))

    def test_entry_points_are_discovered(self):
        # Direct strings plus the Stage2D/Stage2E f-string registrations. A
        # lower bound, so adding a task does not fail this on its own.
        self.assertGreaterEqual(len(self.pairs), 45)
        self.assertIn(("env", "HexapodEnv"), self.pairs)
        self.assertIn(
            ("phase2d_cfg", "HexapodPhase2RecoveryStage2DC5EnvCfg"), self.pairs
        )
        self.assertIn(
            ("phase2e_cfg", "HexapodPhase2RecoveryStage2EE2PPORunnerCfg"), self.pairs
        )

    def test_every_entry_point_module_has_a_shim(self):
        for module, _class_name in sorted(self.pairs):
            self.assertIn(module, HISTORICAL_MODULES, module)
            self.assertTrue((SHIM_DIR / f"{module}.py").is_file(), module)

    def test_every_entry_point_class_is_defined_in_the_package(self):
        defined: dict[str, set[str]] = {}
        for module in {module for module, _ in self.pairs}:
            tree = _tree(PACKAGE_ROOT / f"{module}.py")
            defined[module] = {
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            }
        for module, class_name in sorted(self.pairs):
            self.assertIn(class_name, defined[module], f"{module}:{class_name}")

    def test_entry_point_strings_still_name_hexapod_rl(self):
        source = (PACKAGE_ROOT / "register.py").read_text(encoding="utf-8")
        self.assertNotIn("hexapod_env.", source)
        self.assertIn("hexapod_rl.env:HexapodEnv", source)


class RewardsPackageTest(unittest.TestCase):
    def test_rewards_modules_match_the_declared_split(self):
        self.assertEqual(
            {path.stem for path in _reward_module_paths()},
            set(REWARDS_LAYOUT),
        )
        self.assertTrue((REWARDS_ROOT / "__init__.py").is_file())

    def test_each_function_lives_in_exactly_one_rewards_module(self):
        located: dict[str, list[str]] = {}
        for path in _reward_module_paths():
            for name in _top_level_functions(_tree(path)):
                located.setdefault(name, []).append(path.stem)
        for module, names in REWARDS_LAYOUT.items():
            for name in names:
                self.assertEqual(located.get(name), [module], name)
        expected = {name for names in REWARDS_LAYOUT.values() for name in names}
        self.assertEqual(set(located), expected)

    def test_rewards_modules_import_only_math_torch_and_typing(self):
        for path in _reward_module_paths() + [REWARDS_ROOT / "__init__.py"]:
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(
                            alias.name.split(".")[0],
                            ALLOWED_REWARDS_IMPORTS,
                            f"{path.name} imports {alias.name}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(
                        (node.module or "").split(".")[0],
                        ALLOWED_REWARDS_IMPORTS,
                        f"{path.name} imports from {node.module}",
                    )
                    self.assertEqual(node.level, 0, f"{path.name} imports relatively")

    def test_env_reexports_every_moved_function(self):
        tree = _tree(PACKAGE_ROOT / "env.py")
        imported = _imported_names(tree)
        exported = _dunder_all(tree)
        self.assertIsNotNone(exported)
        for names in REWARDS_LAYOUT.values():
            for name in names:
                self.assertIn(name, imported, f"env.py does not import {name}")
                self.assertIn(name, exported or [], f"env.py does not export {name}")
        self.assertIn("HexapodEnv", exported or [])

    def test_env_keeps_the_environment_class_and_no_reward_functions(self):
        tree = _tree(PACKAGE_ROOT / "env.py")
        self.assertEqual(_top_level_functions(tree), [])
        self.assertEqual(
            [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
            ["HexapodEnv"],
        )


if __name__ == "__main__":
    unittest.main()
