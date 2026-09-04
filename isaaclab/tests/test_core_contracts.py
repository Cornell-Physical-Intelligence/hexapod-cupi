"""Bind every ``hexapod_core`` number to the training source literal it claims.

``hexapod_core`` is a set of frozen constants copied out of the Isaac Lab task.
Copied constants rot: someone retunes ``phase2_cfg.py`` and the runtime keeps
scaling actions by a number the policy no longer uses. This module is the
tripwire. It parses the real sources under
``packages/hexapod_env/hexapod_env/`` with stdlib ``ast`` -- nothing is
imported, so no Isaac Lab, gymnasium, torch, or GPU is needed -- and asserts
every contract value against the literal it was taken from, resolving the
config inheritance chain rather than trusting a docstring or a comment.

Two things are checked structurally rather than by value:

* The 66-dimensional observation order is read off the ``observation_terms``
  list in ``env.py::HexapodEnv._get_observations``, so a reordered
  concatenation fails here even if every width stays the same.
* The observation slices must tile ``[0, 66)`` exactly, in declared order, with
  no gap and no overlap.

The one contract value that has no source literal is the runtime joint order:
Isaac Lab builds it from the USD at load time, and the training Python never
declares it. That one is bound to the documented table in ``docs/TRAINING.md``
instead, and :mod:`hexapod_core.joints` says so in its module docstring.
"""

from __future__ import annotations

import ast
import math
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PACKAGE = REPO_ROOT / "packages" / "hexapod_env" / "hexapod_env"
CORE_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_core"
TRAINING_DOC = REPO_ROOT / "docs" / "TRAINING.md"

if str(CORE_PACKAGE_DIR) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
    sys.path.insert(0, str(CORE_PACKAGE_DIR))

from hexapod_core import action as action_contract  # noqa: E402
from hexapod_core import actuator as actuator_contract  # noqa: E402
from hexapod_core import command as command_contract  # noqa: E402
from hexapod_core import frames as frames_contract  # noqa: E402
from hexapod_core import joints as joint_contract  # noqa: E402
from hexapod_core import observation as observation_contract  # noqa: E402


def _tree(name: str) -> ast.Module:
    return ast.parse((ENV_PACKAGE / name).read_text(encoding="utf-8"))


TREES = {
    name: _tree(name)
    for name in (
        "env.py",
        "env_cfg.py",
        "asset_cfg.py",
        "command_sampling.py",
        "phase1_v2_cfg.py",
        "phase1_v3_cfg.py",
        "phase1_v4_cfg.py",
        "phase1_v5_cfg.py",
        "phase2_cfg.py",
    )
}
TREES["rewards/actions.py"] = ast.parse(
    (ENV_PACKAGE / "rewards" / "actions.py").read_text(encoding="utf-8")
)

# Every configclass in the modules above, by name. The Stage2C environment
# config inherits most of its interface, so resolution has to walk this index.
CLASSES: dict[str, tuple[str, ast.ClassDef]] = {}
for _module_name, _tree_node in TREES.items():
    for _node in _tree_node.body:
        if isinstance(_node, ast.ClassDef):
            CLASSES.setdefault(_node.name, (_module_name, _node))

STAGE2C_ENV_CLASS = "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"
STAGE2C_COMMAND_CLASS = "HexapodStage2CStabilizedForwardCommandCfg"
BASE_ENV_CLASS = "HexapodFlatEnvCfg"


def _number(node: ast.expr):
    """Evaluate a literal or a constant arithmetic expression such as ``1.0 / 200.0``."""

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        expression = ast.Expression(body=node)
        ast.fix_missing_locations(expression)
        return eval(  # noqa: S307 - a config literal from this repo, not input
            compile(expression, "<config>", "eval"), {"__builtins__": {}, "math": math}
        )


def _assignment(class_node: ast.ClassDef, name: str) -> ast.expr | None:
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node.value
    return None


def _declares(class_name: str, attribute: str) -> bool:
    _, class_node = CLASSES[class_name]
    return _assignment(class_node, attribute) is not None


def _resolve(class_name: str, attribute: str) -> tuple[str, ast.expr]:
    """Return ``(declaring class, value node)`` for ``attribute``, walking the bases.

    This is the configclass equivalent of attribute lookup on the MRO, and it
    is the whole reason this helper exists: several deployed values are not
    declared on the class that runs them.
    """

    if class_name not in CLASSES:
        raise AssertionError(f"Unknown config class {class_name}")
    _, class_node = CLASSES[class_name]
    value = _assignment(class_node, attribute)
    if value is not None:
        return class_name, value
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id in CLASSES:
            try:
                return _resolve(base.id, attribute)
            except AssertionError:
                continue
    raise AssertionError(f"{class_name} does not resolve attribute {attribute!r}")


def _resolved_value(class_name: str, attribute: str):
    _, node = _resolve(class_name, attribute)
    return _number(node)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    _, class_node = CLASSES[class_name]
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"Missing {class_name}.{method_name}")


def _call_keywords(node: ast.expr) -> dict[str, ast.expr]:
    if not isinstance(node, ast.Call):
        raise AssertionError(f"Expected a call, got {ast.dump(node)[:60]}")
    return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}


def _module_assignment(module_name: str, name: str) -> ast.expr:
    for node in TREES[module_name].body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"Missing module-level {name} in {module_name}")


class ObservationLayoutTests(unittest.TestCase):
    """The 66-dimensional layout, bound to the concatenation that produces it."""

    def test_observation_terms_match_the_declared_field_order(self) -> None:
        method = _method("HexapodEnv", "_get_observations")
        terms = None
        for node in method.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "observation_terms"
                for target in node.targets
            ):
                terms = node.value
        self.assertIsInstance(terms, ast.List, "observation_terms must be a list literal")
        rendered = [ast.unparse(element) for element in terms.elts]
        self.assertEqual(
            rendered,
            [
                "root_lin_vel",
                "root_ang_vel",
                "projected_gravity",
                "self._commands",
                "self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch",
                "self._robot.data.joint_vel.torch",
                "self._actions",
            ],
        )
        # The names above map one-to-one, in order, onto the contract's fields.
        self.assertEqual(
            observation_contract.FIELD_NAMES,
            (
                "root_linear_velocity",
                "root_angular_velocity",
                "projected_gravity",
                "command",
                "joint_position_error",
                "joint_velocity",
                "action",
            ),
        )

    def test_leading_blocks_are_expressed_in_the_command_frame(self) -> None:
        method = _method("HexapodEnv", "_get_observations")
        bindings = {
            target.id: ast.unparse(node.value)
            for node in method.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            bindings["root_lin_vel"],
            "self._vector_in_command_frame(self._robot.data.root_lin_vel_b.torch)",
        )
        self.assertEqual(
            bindings["root_ang_vel"],
            "self._vector_in_command_frame(self._robot.data.root_ang_vel_b.torch)",
        )
        self.assertEqual(
            bindings["projected_gravity"],
            "self._vector_in_command_frame(self._robot.data.projected_gravity_b.torch)",
        )

    def test_action_block_is_the_stand_scaled_clipped_action(self) -> None:
        # ``self._actions`` is assigned in _pre_physics_step from the clipped
        # action after the stand scale, so the observed action block is zero
        # under the Stage2C stand_action_scale while the command is inactive.
        method = _method("HexapodEnv", "_pre_physics_step")
        rendered = ast.unparse(method)
        self.assertIn("clipped_actions = actions.clone().clamp(-1.0, 1.0)", rendered)
        self.assertIn(
            "self._actions = apply_command_conditioned_stand_action_scale(", rendered
        )

    def test_dimension_and_slices_tile_the_vector(self) -> None:
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "observation_space")),
            observation_contract.OBSERVATION_DIM,
        )
        self.assertEqual(observation_contract.OBSERVATION_DIM, 66)

        cursor = 0
        seen: set[str] = set()
        for name, field in observation_contract.FIELDS:
            self.assertNotIn(name, seen, f"duplicate observation field {name!r}")
            seen.add(name)
            self.assertEqual(field.start, cursor, f"gap or overlap before {name!r}")
            self.assertGreater(field.stop, field.start, f"empty field {name!r}")
            self.assertIsNone(field.step, f"{name!r} must be a contiguous slice")
            cursor = field.stop
        self.assertEqual(cursor, observation_contract.OBSERVATION_DIM)

    def test_field_widths_and_layout_helper(self) -> None:
        self.assertEqual(
            [observation_contract.width_of(name) for name in observation_contract.FIELD_NAMES],
            [3, 3, 3, 3, 18, 18, 18],
        )
        layout = observation_contract.field_layout()
        self.assertEqual(list(layout), list(observation_contract.FIELD_NAMES))
        for name, (start, stop) in layout.items():
            field = observation_contract.slice_of(name)
            self.assertEqual((field.start, field.stop), (start, stop))
        self.assertEqual(
            sum(stop - start for start, stop in layout.values()),
            observation_contract.OBSERVATION_DIM,
        )
        with self.assertRaises(KeyError):
            observation_contract.slice_of("gait_phase")


class ActionInterfaceTests(unittest.TestCase):
    """Action dimension, clip bounds, scale, slew budget, stand scale, timing."""

    def test_action_dimension_and_clip_bounds(self) -> None:
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "action_space")), action_contract.ACTION_DIM
        )
        self.assertEqual(action_contract.ACTION_DIM, 18)

        method = _method("HexapodEnv", "_pre_physics_step")
        clamps = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clamp"
        ]
        self.assertTrue(clamps, "expected an action clamp in _pre_physics_step")
        self.assertEqual(
            [_number(argument) for argument in clamps[0].args],
            [action_contract.ACTION_CLIP_MIN, action_contract.ACTION_CLIP_MAX],
        )

    def test_action_scale_is_inherited_not_declared_by_stage2c(self) -> None:
        # The Stage2C class comments that it uses "the inherited 0.20 action
        # scale" but never declares it. Resolve the chain instead of trusting
        # either the comment or the docs.
        self.assertFalse(_declares(STAGE2C_ENV_CLASS, "action_scale"))
        declaring, node = _resolve(STAGE2C_ENV_CLASS, "action_scale")
        self.assertEqual(declaring, "HexapodPhase1V4EnvCfg")
        self.assertEqual(_number(node), action_contract.ACTION_SCALE_RAD)
        self.assertEqual(action_contract.ACTION_SCALE_RAD, 0.20)
        # ...and the base default is a different number that must not leak into
        # the runtime.
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "action_scale")),
            action_contract.BASE_ACTION_SCALE_RAD,
        )
        self.assertEqual(action_contract.BASE_ACTION_SCALE_RAD, 0.30)
        self.assertNotEqual(
            action_contract.ACTION_SCALE_RAD, action_contract.BASE_ACTION_SCALE_RAD
        )

    def test_stand_action_scale_override(self) -> None:
        self.assertTrue(_declares(STAGE2C_ENV_CLASS, "stand_action_scale"))
        self.assertEqual(
            _resolved_value(STAGE2C_ENV_CLASS, "stand_action_scale"),
            action_contract.STAND_ACTION_SCALE,
        )
        self.assertEqual(action_contract.STAND_ACTION_SCALE, 0.0)
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "stand_action_scale")),
            action_contract.BASE_STAND_ACTION_SCALE,
        )
        self.assertEqual(action_contract.BASE_STAND_ACTION_SCALE, 1.0)

    def test_slew_limit_override_and_reference_step(self) -> None:
        self.assertTrue(
            _declares(STAGE2C_ENV_CLASS, "processed_joint_target_slew_limit_rad_per_20ms")
        )
        self.assertEqual(
            _resolved_value(
                STAGE2C_ENV_CLASS, "processed_joint_target_slew_limit_rad_per_20ms"
            ),
            action_contract.PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS,
        )
        self.assertEqual(
            action_contract.PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS, 0.040
        )
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertIsNone(
            _number(
                _assignment(base, "processed_joint_target_slew_limit_rad_per_20ms")
            ),
            "the base default must stay a None pass-through",
        )
        self.assertIsNone(
            action_contract.BASE_PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS
        )

        # The 20 ms the budget is quoted against is a literal in the limiter.
        limiter = _function(TREES["rewards/actions.py"], "limit_processed_joint_target_slew")
        allowed = None
        for node in ast.walk(limiter):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "allowed_delta"
                for target in node.targets
            ):
                allowed = node.value
        self.assertIsNotNone(allowed, "expected an allowed_delta assignment")
        self.assertIsInstance(allowed.op, ast.Div)
        self.assertEqual(
            _number(allowed.right), action_contract.SLEW_REFERENCE_STEP_S
        )
        self.assertEqual(action_contract.SLEW_REFERENCE_STEP_S, 0.020)

    def test_command_active_threshold(self) -> None:
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "axis_command_active_threshold")),
            action_contract.COMMAND_ACTIVE_THRESHOLD,
        )
        self.assertEqual(
            _resolved_value(STAGE2C_ENV_CLASS, "axis_command_active_threshold"),
            action_contract.COMMAND_ACTIVE_THRESHOLD,
        )

    def test_timing_contract(self) -> None:
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "decimation")), action_contract.DECIMATION
        )
        self.assertEqual(
            _number(_assignment(base, "episode_length_s")),
            action_contract.EPISODE_LENGTH_S,
        )
        simulation = _call_keywords(_assignment(base, "sim"))
        self.assertAlmostEqual(
            _number(simulation["dt"]), action_contract.PHYSICS_DT_S, places=12
        )
        self.assertAlmostEqual(action_contract.PHYSICS_RATE_HZ, 200.0, places=12)
        self.assertAlmostEqual(action_contract.POLICY_STEP_DT_S, 0.020, places=12)
        self.assertAlmostEqual(action_contract.POLICY_RATE_HZ, 50.0, places=12)
        # The slew budget is quoted per 20 ms and the policy step is 20 ms, so
        # the deployed rescaling factor is exactly one. Anything else means the
        # timing contract and the limiter have drifted apart.
        self.assertAlmostEqual(
            action_contract.POLICY_STEP_DT_S / action_contract.SLEW_REFERENCE_STEP_S,
            1.0,
            places=12,
        )

    def test_provenance_covers_every_recorded_value(self) -> None:
        for name in (
            "ACTION_DIM",
            "ACTION_SCALE_RAD",
            "BASE_ACTION_SCALE_RAD",
            "PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS",
            "STAND_ACTION_SCALE",
            "BASE_STAND_ACTION_SCALE",
            "SLEW_REFERENCE_STEP_S",
        ):
            self.assertIn(name, action_contract.PROVENANCE)
            self.assertTrue(action_contract.PROVENANCE[name].strip())


class CommandContractTests(unittest.TestCase):
    """Command envelopes bound to phase2_cfg, plus the dataclass's own behavior."""

    def test_stage2c_envelope_matches_the_trained_command_cfg(self) -> None:
        _, command_cfg = CLASSES[STAGE2C_COMMAND_CLASS]
        self.assertEqual(
            _number(_assignment(command_cfg, "lin_vel_x_range")),
            command_contract.STAGE2C_FORWARD_RANGE_MPS,
        )
        self.assertEqual(
            _number(_assignment(command_cfg, "lin_vel_y_range")),
            command_contract.STAGE2C_LATERAL_RANGE_MPS,
        )
        self.assertEqual(
            _number(_assignment(command_cfg, "ang_vel_z_range")),
            command_contract.STAGE2C_YAW_RATE_RANGE_RAD_S,
        )
        self.assertEqual(
            _number(_assignment(command_cfg, "standing_probability")),
            command_contract.STAGE2C_STANDING_PROBABILITY,
        )
        self.assertEqual(command_contract.STAGE2C_FORWARD_RANGE_MPS, (0.16, 0.32))

    def test_phase2_target_envelope_matches_the_final_cfg(self) -> None:
        _, final_cfg = CLASSES["HexapodPhase2FinalEnvCfg"]
        keywords = _call_keywords(_assignment(final_cfg, "velocity_command"))
        self.assertEqual(
            _number(keywords["lin_vel_x_range"]),
            command_contract.PHASE2_FORWARD_RANGE_MPS,
        )
        self.assertEqual(
            _number(keywords["lin_vel_y_range"]),
            command_contract.PHASE2_LATERAL_RANGE_MPS,
        )
        self.assertEqual(
            _number(keywords["ang_vel_z_range"]),
            command_contract.PHASE2_YAW_RATE_RANGE_RAD_S,
        )
        self.assertEqual(command_contract.PHASE2_FORWARD_RANGE_MPS, (-0.40, 0.60))
        self.assertEqual(command_contract.PHASE2_LATERAL_RANGE_MPS, (-0.35, 0.35))
        self.assertEqual(command_contract.PHASE2_YAW_RATE_RANGE_RAD_S, (-0.75, 0.75))

    def test_command_occupies_three_observation_slots(self) -> None:
        self.assertEqual(observation_contract.width_of("command"), 3)
        self.assertEqual(
            len(command_contract.VelocityCommand(0.2, 0.0, 0.0).as_tuple()), 3
        )

    def test_validate_accepts_and_rejects(self) -> None:
        inside = command_contract.VelocityCommand(0.25, 0.10, -0.20)
        inside.validate(command_contract.PHASE2_TARGET_ENVELOPE)
        self.assertTrue(inside.is_within(command_contract.PHASE2_TARGET_ENVELOPE))

        outside = command_contract.VelocityCommand(0.90, 0.0, 0.0)
        with self.assertRaises(ValueError):
            outside.validate(command_contract.PHASE2_TARGET_ENVELOPE)
        self.assertFalse(outside.is_within(command_contract.PHASE2_TARGET_ENVELOPE))

        with self.assertRaises(ValueError):
            command_contract.VelocityCommand(float("nan"), 0.0, 0.0).validate()

    def test_stand_command_is_admissible_inside_the_forward_only_envelope(self) -> None:
        # Stage2C samples a standing command 20% of the time even though 0.0 is
        # below its 0.16 m/s forward floor, so a validator that rejected the
        # stand command would reject the training distribution itself.
        stand = command_contract.VelocityCommand(0.0, 0.0, 0.0)
        self.assertTrue(stand.is_stand())
        stand.validate(command_contract.STAGE2C_ENVELOPE)
        command_contract.VelocityCommand(0.20, 0.0, 0.0).validate(
            command_contract.STAGE2C_ENVELOPE
        )
        with self.assertRaises(ValueError):
            command_contract.VelocityCommand(0.20, 0.05, 0.0).validate(
                command_contract.STAGE2C_ENVELOPE
            )

    def test_frozen_and_clamped(self) -> None:
        command = command_contract.VelocityCommand(0.25, 0.0, 0.0)
        with self.assertRaises(Exception):
            command.vx_mps = 0.30  # type: ignore[misc]
        clamped = command_contract.PHASE2_TARGET_ENVELOPE.clamp(
            command_contract.VelocityCommand(9.0, -9.0, 9.0)
        )
        self.assertEqual(clamped.as_tuple(), (0.60, -0.35, 0.75))

    def test_navigation_is_the_deployed_command_frame(self) -> None:
        self.assertEqual(
            _resolved_value(STAGE2C_ENV_CLASS, "command_frame"),
            command_contract.Frame.NAVIGATION.value,
        )


class JointOrderTests(unittest.TestCase):
    """Runtime joint order, its documented source, and the default stance."""

    def _asset_group(self, name: str) -> tuple[str, ...]:
        return tuple(_number(_module_assignment("asset_cfg.py", name)))

    def test_asset_cfg_name_lists_match(self) -> None:
        self.assertEqual(
            self._asset_group("COXA_JOINTS"), joint_contract.ASSET_CFG_COXA_JOINTS
        )
        self.assertEqual(
            self._asset_group("FEMUR_JOINTS"), joint_contract.ASSET_CFG_FEMUR_JOINTS
        )
        self.assertEqual(
            self._asset_group("TIBIA_JOINTS"), joint_contract.ASSET_CFG_TIBIA_JOINTS
        )

    def test_runtime_groups_are_permutations_of_the_asset_lists(self) -> None:
        names = joint_contract.RUNTIME_JOINT_NAMES
        self.assertEqual(len(names), joint_contract.JOINT_COUNT)
        self.assertEqual(len(set(names)), joint_contract.JOINT_COUNT)
        for group_slice, asset_names in (
            (joint_contract.COXA_SLICE, joint_contract.ASSET_CFG_COXA_JOINTS),
            (joint_contract.FEMUR_SLICE, joint_contract.ASSET_CFG_FEMUR_JOINTS),
            (joint_contract.TIBIA_SLICE, joint_contract.ASSET_CFG_TIBIA_JOINTS),
        ):
            self.assertEqual(set(names[group_slice]), set(asset_names))

    def test_declaration_order_is_not_the_runtime_order(self) -> None:
        # The trap this contract exists to prevent: coxa happens to coincide,
        # femur and tibia do not. Indexing an action vector with the asset_cfg
        # declaration order silently permutes two thirds of the robot.
        names = joint_contract.RUNTIME_JOINT_NAMES
        self.assertEqual(
            names[joint_contract.COXA_SLICE], joint_contract.ASSET_CFG_COXA_JOINTS
        )
        self.assertNotEqual(
            names[joint_contract.FEMUR_SLICE], joint_contract.ASSET_CFG_FEMUR_JOINTS
        )
        self.assertNotEqual(
            names[joint_contract.TIBIA_SLICE], joint_contract.ASSET_CFG_TIBIA_JOINTS
        )

    def test_runtime_order_matches_the_documented_table(self) -> None:
        text = TRAINING_DOC.read_text(encoding="utf-8")
        marker = text.index("Runtime joint/action order")
        block = text[marker : text.index("```", text.index("```", marker) + 3)]
        documented = {
            int(index): name
            for index, name in re.findall(r"(\d+)\s+(revolute_[0-9_]+)", block)
        }
        self.assertEqual(len(documented), joint_contract.JOINT_COUNT)
        self.assertEqual(
            tuple(documented[index] for index in range(joint_contract.JOINT_COUNT)),
            joint_contract.RUNTIME_JOINT_NAMES,
        )

    def test_group_helpers(self) -> None:
        self.assertEqual(joint_contract.group_of(0), "coxa")
        self.assertEqual(joint_contract.group_of(5), "coxa")
        self.assertEqual(joint_contract.group_of(6), "femur")
        self.assertEqual(joint_contract.group_of(11), "femur")
        self.assertEqual(joint_contract.group_of(12), "tibia")
        self.assertEqual(joint_contract.group_of(17), "tibia")
        with self.assertRaises(IndexError):
            joint_contract.group_of(18)
        self.assertEqual(joint_contract.index_of("revolute_1_1"), 0)
        self.assertEqual(joint_contract.index_of("revolute_2_1"), 17)
        with self.assertRaises(KeyError):
            joint_contract.index_of("revolute_99")

    def _dict_comp_values(self, node: ast.expr) -> dict[str, float]:
        """Map ``{name: value for name in GROUP}`` comprehensions to group -> value."""

        values: dict[str, float] = {}
        for comprehension in ast.walk(node):
            if not isinstance(comprehension, ast.DictComp):
                continue
            generator = comprehension.generators[0]
            self.assertIsInstance(generator.iter, ast.Name)
            values[generator.iter.id] = _number(comprehension.value)
        return values

    def test_default_joint_positions(self) -> None:
        init_state = _call_keywords(_module_assignment("asset_cfg.py", "HEXAPOD_CFG"))[
            "init_state"
        ]
        base_joint_pos = _call_keywords(init_state)["joint_pos"]
        base = self._dict_comp_values(base_joint_pos)
        self.assertEqual(base["COXA_JOINTS"], 0.0)
        self.assertEqual(base["FEMUR_JOINTS"], 0.40)
        self.assertEqual(base["TIBIA_JOINTS"], 2.10)
        self.assertEqual(
            joint_contract.BASE_DEFAULT_JOINT_POSITIONS_RAD,
            (0.0,) * 6 + (0.40,) * 6 + (2.10,) * 6,
        )

        _, stage2c = CLASSES[STAGE2C_ENV_CLASS]
        stage2c_joint_pos = _call_keywords(
            _call_keywords(_assignment(stage2c, "robot"))["init_state"]
        )["joint_pos"]
        override = self._dict_comp_values(stage2c_joint_pos)
        self.assertEqual(override["FEMUR_JOINTS"], 0.60)
        self.assertEqual(override["TIBIA_JOINTS"], 2.2335)
        self.assertNotIn("COXA_JOINTS", override, "coxa stays at the inherited 0.0")
        self.assertEqual(
            joint_contract.STAGE2C_DEFAULT_JOINT_POSITIONS_RAD,
            (0.0,) * 6 + (0.60,) * 6 + (2.2335,) * 6,
        )


class ActuatorContractTests(unittest.TestCase):
    """RS05 numbers, bound to ROBSTRIDE_RS05_CFG and the Stage2C termination gate."""

    def setUp(self) -> None:
        self.actuator = _call_keywords(
            _module_assignment("asset_cfg.py", "ROBSTRIDE_RS05_CFG")
        )
        self.articulation = _call_keywords(
            _module_assignment("asset_cfg.py", "HEXAPOD_CFG")
        )

    def test_torque_limits(self) -> None:
        self.assertEqual(
            _number(self.actuator["effort_limit"]), actuator_contract.CONTINUOUS_TORQUE_NM
        )
        self.assertEqual(
            _number(self.actuator["saturation_effort"]), actuator_contract.PEAK_TORQUE_NM
        )
        self.assertEqual(
            _number(self.actuator["effort_limit_sim"]), actuator_contract.PEAK_TORQUE_NM
        )
        _, base = CLASSES[BASE_ENV_CLASS]
        self.assertEqual(
            _number(_assignment(base, "rated_torque_nm")),
            actuator_contract.RATED_TORQUE_NM,
        )
        self.assertEqual(
            actuator_contract.RATED_TORQUE_NM, actuator_contract.CONTINUOUS_TORQUE_NM
        )
        self.assertEqual(actuator_contract.CONTINUOUS_TORQUE_NM, 1.6)
        self.assertEqual(actuator_contract.PEAK_TORQUE_NM, 5.5)

    def test_velocity_limits_keep_their_rpm_form(self) -> None:
        self.assertAlmostEqual(
            _number(self.actuator["velocity_limit"]),
            actuator_contract.NOMINAL_VELOCITY_RAD_S,
            places=12,
        )
        self.assertAlmostEqual(
            _number(self.actuator["velocity_limit_sim"]),
            actuator_contract.SIM_VELOCITY_RAD_S,
            places=12,
        )
        # The documented rounded values, kept honest to two decimals.
        self.assertAlmostEqual(actuator_contract.NOMINAL_VELOCITY_RAD_S, 50.27, places=2)
        self.assertAlmostEqual(actuator_contract.SIM_VELOCITY_RAD_S, 55.29, places=2)

    def test_mechanical_model(self) -> None:
        self.assertEqual(
            _number(self.actuator["stiffness"]), actuator_contract.STIFFNESS_NM_PER_RAD
        )
        self.assertEqual(
            _number(self.actuator["damping"]), actuator_contract.DAMPING_NM_S_PER_RAD
        )
        self.assertEqual(
            _number(self.actuator["armature"]), actuator_contract.ARMATURE_KG_M2
        )
        self.assertEqual(actuator_contract.ARMATURE_KG_M2, 0.0007)
        self.assertEqual(
            _number(self.actuator["friction"]), actuator_contract.STATIC_FRICTION
        )
        self.assertEqual(
            _number(self.actuator["dynamic_friction"]), actuator_contract.DYNAMIC_FRICTION
        )
        self.assertEqual(
            _number(self.actuator["viscous_friction"]), actuator_contract.VISCOUS_FRICTION
        )
        self.assertEqual(
            _number(self.articulation["soft_joint_pos_limit_factor"]),
            actuator_contract.SOFT_JOINT_LIMIT_FACTOR,
        )
        self.assertEqual(actuator_contract.SOFT_JOINT_LIMIT_FACTOR, 0.95)
        articulation_props = _call_keywords(
            _call_keywords(self.articulation["spawn"])["articulation_props"]
        )
        self.assertEqual(
            _number(articulation_props["enabled_self_collisions"]),
            actuator_contract.SELF_COLLISIONS_ENABLED,
        )

    def test_stage2c_torque_demand_termination(self) -> None:
        self.assertEqual(
            _resolved_value(STAGE2C_ENV_CLASS, "terminate_on_computed_torque_demand_nm"),
            actuator_contract.TERMINATION_RAW_DEMAND_NM,
        )
        self.assertEqual(
            _resolved_value(
                STAGE2C_ENV_CLASS, "terminate_on_computed_torque_demand_duration_s"
            ),
            actuator_contract.TERMINATION_RAW_DEMAND_DURATION_S,
        )
        self.assertEqual(
            _resolved_value(STAGE2C_ENV_CLASS, "torque_demand_termination_grace_s"),
            actuator_contract.TERMINATION_RAW_DEMAND_GRACE_S,
        )
        self.assertEqual(
            actuator_contract.TERMINATION_RAW_DEMAND_NM, actuator_contract.PEAK_TORQUE_NM
        )


class FrameContractTests(unittest.TestCase):
    """The anatomical coordinate contract, bound to body_to_navigation_frame."""

    def test_rotation_matches_the_training_implementation(self) -> None:
        function = _function(TREES["command_sampling.py"], "body_to_navigation_frame")
        returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertEqual(
            ast.unparse(returns[0].value),
            "torch.stack((-vectors[..., 1], vectors[..., 0], vectors[..., 2]), dim=-1)",
        )

    def test_pure_python_rotation_agrees(self) -> None:
        self.assertEqual(frames_contract.body_to_navigation((1.0, 2.0, 3.0)), (-2.0, 1.0, 3.0))
        self.assertEqual(
            frames_contract.navigation_to_body(
                frames_contract.body_to_navigation((0.3, -0.7, 0.1))
            ),
            (0.3, -0.7, 0.1),
        )
        # Body -Y is navigation forward, body +X is navigation lateral.
        self.assertEqual(frames_contract.body_to_navigation((0.0, -1.0, 0.0)), (1.0, 0.0, 0.0))
        self.assertEqual(frames_contract.body_to_navigation((1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    def test_axis_constants_and_matrix(self) -> None:
        self.assertEqual(frames_contract.NAVIGATION_FORWARD_IN_BODY, (0.0, -1.0, 0.0))
        self.assertEqual(frames_contract.NAVIGATION_LATERAL_IN_BODY, (1.0, 0.0, 0.0))
        self.assertEqual(frames_contract.NAVIGATION_UP_IN_BODY, (0.0, 0.0, 1.0))
        matrix = frames_contract.BODY_TO_NAVIGATION_MATRIX
        body = (0.4, -0.2, 0.9)
        rotated = tuple(sum(r * b for r, b in zip(row, body)) for row in matrix)
        self.assertEqual(rotated, frames_contract.body_to_navigation(body))
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        self.assertAlmostEqual(determinant, 1.0, places=12)


class SchemaVersionTests(unittest.TestCase):
    def test_every_module_is_schema_v1(self) -> None:
        for module in (
            observation_contract,
            action_contract,
            command_contract,
            joint_contract,
            actuator_contract,
            frames_contract,
        ):
            self.assertEqual(module.SCHEMA_VERSION, 1, module.__name__)


if __name__ == "__main__":
    unittest.main()
