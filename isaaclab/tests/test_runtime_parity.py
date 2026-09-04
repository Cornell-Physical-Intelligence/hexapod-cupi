"""Prove the deployable action path and the training env's action path agree.

``hexapod_runtime.action_pipeline`` is a pure-Python re-implementation of what
``HexapodEnv._pre_physics_step`` does with torch tensors. A re-implementation
is a liability unless something keeps it honest, so this module runs the real
training functions -- imported from ``packages/hexapod_env/hexapod_env`` -- and
the pure-Python ones on identical inputs and asserts elementwise agreement to
1e-6.

The parity class is skipped when ``torch`` is unavailable, which is the normal
state on an operator laptop. The torch-free classes below are not skipped:
clip/scale/offset, the stateful pipeline, and the observation builder's
validation all run everywhere, which is the point of a runtime that does not
need the simulator.

Edge cases covered against the torch implementation, chosen because each one is
a place where a plausible re-implementation silently differs:

* a disabled limiter (``None``), which must be an exact pass-through that never
  touches the history;
* the first step with no history, which must pass through unlimited;
* a saturated sign flip -- 0.40 rad, exactly what a full-scale action reversal
  produces at the 0.20 rad action scale -- which must move by exactly one
  budget;
* a step landing exactly on the budget, which must pass through *and* not be
  counted as limited;
* step-dt scaling at half and double the 20 ms reference;
* mixed per-joint directions, so a per-joint clamp cannot be confused with a
  vector-norm clamp;
* the stand-scale threshold, which is strict: a command exactly at the
  threshold is inactive.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_env"
CORE_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_core"
RUNTIME_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_runtime"

for _package_dir in (CORE_PACKAGE_DIR, RUNTIME_PACKAGE_DIR):
    if str(_package_dir) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
        sys.path.insert(0, str(_package_dir))

from hexapod_core import action as action_contract  # noqa: E402
from hexapod_core import joints as joint_contract  # noqa: E402
from hexapod_core import observation as observation_contract  # noqa: E402
from hexapod_runtime import action_pipeline  # noqa: E402
from hexapod_runtime import observation_builder  # noqa: E402


HAS_TORCH = importlib.util.find_spec("torch") is not None
TOLERANCE = 1.0e-6

ACTION_DIM = action_contract.ACTION_DIM
DEFAULTS = list(joint_contract.STAGE2C_DEFAULT_JOINT_POSITIONS_RAD)
SLEW = action_contract.PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS
STEP_DT = action_contract.POLICY_STEP_DT_S


def _rows(count: int, generator: random.Random, low: float, high: float) -> list[list[float]]:
    return [
        [generator.uniform(low, high) for _ in range(ACTION_DIM)] for _ in range(count)
    ]


@unittest.skipUnless(HAS_TORCH, "torch is not importable in this interpreter")
class ActionPipelineParityTests(unittest.TestCase):
    """The pure-Python pipeline against the training env's own torch functions."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch  # noqa: PLC0415 - guarded by skipUnless

        if str(ENV_PACKAGE_DIR) not in sys.path:
            sys.path.insert(0, str(ENV_PACKAGE_DIR))
        # ``hexapod_env/__init__`` defers its gymnasium-dependent imports, so
        # the reward primitives import with torch alone.
        from hexapod_env.rewards.actions import (  # noqa: PLC0415
            apply_command_conditioned_stand_action_scale,
            limit_processed_joint_target_slew,
        )

        cls.torch = torch
        cls.limit_slew = staticmethod(limit_processed_joint_target_slew)
        cls.stand_scale = staticmethod(apply_command_conditioned_stand_action_scale)

    def _compare_slew(
        self,
        targets: list[list[float]],
        previous: list[list[float]],
        history: list[bool],
        *,
        max_delta: float | None,
        step_dt: float,
        dtype=None,
        tolerance: float = TOLERANCE,
    ) -> None:
        torch = self.torch
        dtype = dtype or torch.float64
        reference_target, reference_fraction = self.limit_slew(
            torch.tensor(targets, dtype=dtype),
            torch.tensor(previous, dtype=dtype),
            torch.tensor(history, dtype=torch.bool),
            max_delta_rad_per_20ms=max_delta,
            step_dt=step_dt,
        )
        for row, (target, prior, has_history) in enumerate(
            zip(targets, previous, history)
        ):
            result = action_pipeline.limit_joint_target_slew(
                target,
                prior,
                has_history,
                max_delta_rad_per_20ms=max_delta,
                step_dt=step_dt,
            )
            for joint, value in enumerate(result.target):
                self.assertAlmostEqual(
                    value,
                    float(reference_target[row][joint]),
                    delta=tolerance,
                    msg=f"row {row} joint {joint}",
                )
            self.assertAlmostEqual(
                result.limited_fraction,
                float(reference_fraction[row]),
                delta=tolerance,
                msg=f"limited fraction, row {row}",
            )

    def test_randomized_agreement(self) -> None:
        generator = random.Random(20260827)
        previous = _rows(24, generator, -1.0, 2.5)
        # Deltas span well under, around, and well over the 0.040 rad budget.
        targets = [
            [value + generator.uniform(-0.25, 0.25) for value in row] for row in previous
        ]
        history = [generator.random() < 0.75 for _ in previous]
        self._compare_slew(
            targets, previous, history, max_delta=SLEW, step_dt=STEP_DT
        )

    def test_randomized_agreement_in_float32(self) -> None:
        # The environment runs float32 on device; the runtime is float64. The
        # contract is agreement to 1e-6, not bit-identity.
        generator = random.Random(7)
        previous = _rows(8, generator, -0.5, 2.4)
        targets = [
            [value + generator.uniform(-0.5, 0.5) for value in row] for row in previous
        ]
        history = [True] * len(previous)
        self._compare_slew(
            targets,
            previous,
            history,
            max_delta=SLEW,
            step_dt=STEP_DT,
            dtype=self.torch.float32,
        )

    def test_first_step_without_history_passes_through(self) -> None:
        generator = random.Random(1)
        previous = _rows(4, generator, -1.0, 1.0)
        targets = [[value + 5.0 for value in row] for row in previous]
        history = [False] * len(previous)
        self._compare_slew(targets, previous, history, max_delta=SLEW, step_dt=STEP_DT)
        # ...and the pass-through really is the raw target, not a clamp to it.
        result = action_pipeline.limit_joint_target_slew(
            targets[0], previous[0], False, max_delta_rad_per_20ms=SLEW, step_dt=STEP_DT
        )
        self.assertEqual(result.target, [float(value) for value in targets[0]])
        self.assertEqual(result.limited_fraction, 0.0)

    def test_disabled_limiter_is_an_exact_passthrough(self) -> None:
        generator = random.Random(2)
        previous = _rows(4, generator, -1.0, 1.0)
        targets = _rows(4, generator, -1.0, 1.0)
        self._compare_slew(
            targets, previous, [True] * 4, max_delta=None, step_dt=STEP_DT
        )

    def test_saturated_sign_flip_moves_exactly_one_budget(self) -> None:
        # A full-scale action reversal at the 0.20 rad action scale asks for a
        # 0.40 rad jump: ten times the 0.040 rad budget, alternating sign per
        # joint so a per-joint clamp is distinguishable from a norm clamp.
        flip = 2.0 * action_contract.ACTION_SCALE_RAD
        previous = [list(DEFAULTS)]
        targets = [
            [
                value + (flip if joint % 2 == 0 else -flip)
                for joint, value in enumerate(DEFAULTS)
            ]
        ]
        self._compare_slew(targets, previous, [True], max_delta=SLEW, step_dt=STEP_DT)
        result = action_pipeline.limit_joint_target_slew(
            targets[0], previous[0], True, max_delta_rad_per_20ms=SLEW, step_dt=STEP_DT
        )
        for joint, value in enumerate(result.target):
            expected = DEFAULTS[joint] + (SLEW if joint % 2 == 0 else -SLEW)
            self.assertAlmostEqual(value, expected, delta=1.0e-12)
        self.assertEqual(result.limited_fraction, 1.0)

    def test_exact_budget_step_is_not_counted_as_limited(self) -> None:
        previous = [list(DEFAULTS)]
        targets = [[value + SLEW for value in DEFAULTS]]
        self._compare_slew(targets, previous, [True], max_delta=SLEW, step_dt=STEP_DT)
        result = action_pipeline.limit_joint_target_slew(
            targets[0], previous[0], True, max_delta_rad_per_20ms=SLEW, step_dt=STEP_DT
        )
        self.assertEqual(result.limited_fraction, 0.0)

    def test_step_dt_scaling(self) -> None:
        generator = random.Random(3)
        previous = _rows(6, generator, -0.5, 2.4)
        targets = [[value + 0.30 for value in row] for row in previous]
        for step_dt in (STEP_DT / 2.0, STEP_DT, STEP_DT * 2.0, 0.005, 0.033):
            with self.subTest(step_dt=step_dt):
                self._compare_slew(
                    targets, previous, [True] * len(previous), max_delta=SLEW, step_dt=step_dt
                )
        # Half the policy period must allow exactly half the budget.
        halved = action_pipeline.limit_joint_target_slew(
            targets[0], previous[0], True, max_delta_rad_per_20ms=SLEW, step_dt=STEP_DT / 2.0
        )
        for joint, value in enumerate(halved.target):
            self.assertAlmostEqual(value - previous[0][joint], SLEW / 2.0, delta=1.0e-12)

    def test_mixed_history_within_one_batch(self) -> None:
        # Half the fleet has history and half does not; a re-implementation that
        # applied the mask to the wrong axis passes every uniform case and fails
        # this one.
        generator = random.Random(4)
        previous = _rows(10, generator, -0.5, 2.4)
        targets = [[value + 0.5 for value in row] for row in previous]
        history = [index % 2 == 0 for index in range(len(previous))]
        self._compare_slew(targets, previous, history, max_delta=SLEW, step_dt=STEP_DT)

    def test_stand_action_scale_parity(self) -> None:
        torch = self.torch
        generator = random.Random(5)
        actions = _rows(6, generator, -1.0, 1.0)
        threshold = action_contract.COMMAND_ACTIVE_THRESHOLD
        commands = [
            [0.0, 0.0, 0.0],  # standing
            [0.24, 0.0, 0.0],  # forward
            [0.0, 0.0, -0.30],  # yaw only
            [threshold, 0.0, 0.0],  # exactly at the threshold: inactive
            [threshold * 1.001, 0.0, 0.0],  # just above: active
            [-0.20, 0.05, 0.0],  # reverse with lateral
        ]
        for stand_action_scale in (0.0, 0.5, 1.0):
            with self.subTest(stand_action_scale=stand_action_scale):
                reference = self.stand_scale(
                    torch.tensor(actions, dtype=torch.float64),
                    torch.tensor(commands, dtype=torch.float64),
                    stand_action_scale=stand_action_scale,
                    command_active_threshold=threshold,
                )
                for row, (action, command) in enumerate(zip(actions, commands)):
                    scaled = action_pipeline.apply_stand_action_scale(
                        action,
                        command,
                        stand_action_scale=stand_action_scale,
                        command_active_threshold=threshold,
                    )
                    for joint, value in enumerate(scaled):
                        self.assertAlmostEqual(
                            value,
                            float(reference[row][joint]),
                            delta=TOLERANCE,
                            msg=f"row {row} joint {joint}",
                        )

    def test_full_pipeline_reproduces_the_pre_physics_step(self) -> None:
        """Clip, stand-scale, offset and slew, chained exactly as the env chains them."""

        torch = self.torch
        generator = random.Random(6)
        raw_actions = _rows(5, generator, -1.8, 1.8)
        commands = [[0.24, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]] * 2
        previous_target = torch.tensor([DEFAULTS] * len(raw_actions), dtype=torch.float64)
        has_previous = torch.ones(len(raw_actions), dtype=torch.bool)

        clipped = torch.tensor(raw_actions, dtype=torch.float64).clamp(-1.0, 1.0)
        scaled = self.stand_scale(
            clipped,
            torch.tensor(commands, dtype=torch.float64),
            stand_action_scale=action_contract.STAND_ACTION_SCALE,
            command_active_threshold=action_contract.COMMAND_ACTIVE_THRESHOLD,
        )
        targets = action_contract.ACTION_SCALE_RAD * scaled + torch.tensor(
            [DEFAULTS] * len(raw_actions), dtype=torch.float64
        )
        reference, _ = self.limit_slew(
            targets,
            previous_target,
            has_previous,
            max_delta_rad_per_20ms=SLEW,
            step_dt=STEP_DT,
        )

        for row, (action, command) in enumerate(zip(raw_actions, commands)):
            pipeline = action_pipeline.ActionPipeline(default_joint_positions=DEFAULTS)
            pipeline.reset(DEFAULTS)
            produced = pipeline.step(action, command)
            for joint, value in enumerate(produced):
                self.assertAlmostEqual(
                    value,
                    float(reference[row][joint]),
                    delta=TOLERANCE,
                    msg=f"row {row} joint {joint}",
                )


class ActionPipelineUnitTests(unittest.TestCase):
    """Torch-free checks of the stages the parity test chains together."""

    def test_clip(self) -> None:
        action = [2.0, -2.0] + [0.5] * (ACTION_DIM - 2)
        clipped = action_pipeline.clip_action(action)
        self.assertEqual(clipped[0], 1.0)
        self.assertEqual(clipped[1], -1.0)
        self.assertEqual(clipped[2], 0.5)
        with self.assertRaises(ValueError):
            action_pipeline.clip_action([0.0] * (ACTION_DIM - 1))
        with self.assertRaises(ValueError):
            action_pipeline.clip_action([float("nan")] * ACTION_DIM)

    def test_scale_and_offset(self) -> None:
        action = [1.0] * ACTION_DIM
        target = action_pipeline.scale_and_offset(action, DEFAULTS)
        for joint, value in enumerate(target):
            self.assertAlmostEqual(
                value, DEFAULTS[joint] + action_contract.ACTION_SCALE_RAD, places=12
            )
        zero = action_pipeline.scale_and_offset([0.0] * ACTION_DIM, DEFAULTS)
        self.assertEqual(zero, DEFAULTS)

    def test_stand_scale_zeroes_the_offset_while_standing(self) -> None:
        # The deployed combination: an inactive command plus stand scale 0.0
        # requests exactly the default pose, whatever the policy asked for.
        action = [1.0, -1.0] * (ACTION_DIM // 2)
        scaled = action_pipeline.apply_stand_action_scale(
            action, [0.0, 0.0, 0.0], stand_action_scale=0.0
        )
        self.assertEqual(scaled, [0.0] * ACTION_DIM)
        self.assertEqual(action_pipeline.scale_and_offset(scaled, DEFAULTS), DEFAULTS)
        passed_through = action_pipeline.apply_stand_action_scale(
            action, [0.24, 0.0, 0.0], stand_action_scale=0.0
        )
        self.assertEqual(passed_through, [float(value) for value in action])

    def test_soft_limit_clamp(self) -> None:
        target = [3.0] * ACTION_DIM
        limits = [(-0.5, 0.5)] * ACTION_DIM
        self.assertEqual(
            action_pipeline.clamp_to_soft_limits(target, limits), [0.5] * ACTION_DIM
        )
        self.assertEqual(action_pipeline.clamp_to_soft_limits(target, None), target)
        with self.assertRaises(ValueError):
            action_pipeline.clamp_to_soft_limits(target, [(0.5, -0.5)] * ACTION_DIM)

    def test_pipeline_converges_over_successive_steps(self) -> None:
        # A saturated command cannot be reached in one step, and the limiter is
        # a rate limit, not a deadband: the target must keep advancing.
        pipeline = action_pipeline.ActionPipeline(default_joint_positions=DEFAULTS)
        pipeline.reset(DEFAULTS)
        action = [1.0] * ACTION_DIM
        goal = DEFAULTS[0] + action_contract.ACTION_SCALE_RAD
        previous = DEFAULTS[0]
        for _ in range(4):
            target = pipeline.step(action, [0.24, 0.0, 0.0])
            self.assertAlmostEqual(target[0] - previous, SLEW, places=12)
            previous = target[0]
        self.assertLess(previous, goal)
        for _ in range(20):
            target = pipeline.step(action, [0.24, 0.0, 0.0])
        self.assertAlmostEqual(target[0], goal, places=12)
        self.assertEqual(pipeline.last_limited_fraction, 0.0)

    def test_reset_clears_history(self) -> None:
        pipeline = action_pipeline.ActionPipeline(default_joint_positions=DEFAULTS)
        pipeline.reset()  # no measured pose: the first step is unlimited
        first = pipeline.step([1.0] * ACTION_DIM, [0.24, 0.0, 0.0])
        self.assertAlmostEqual(
            first[0], DEFAULTS[0] + action_contract.ACTION_SCALE_RAD, places=12
        )
        # With the history anchored to a measured pose, the same step is limited.
        pipeline.reset(DEFAULTS)
        limited = pipeline.step([1.0] * ACTION_DIM, [0.24, 0.0, 0.0])
        self.assertAlmostEqual(limited[0], DEFAULTS[0] + SLEW, places=12)

    def test_disabled_limiter_never_builds_history(self) -> None:
        pipeline = action_pipeline.ActionPipeline(
            default_joint_positions=DEFAULTS, slew_limit_rad_per_20ms=None
        )
        pipeline.reset(DEFAULTS)
        for _ in range(3):
            target = pipeline.step([1.0] * ACTION_DIM, [0.24, 0.0, 0.0])
        self.assertAlmostEqual(
            target[0], DEFAULTS[0] + action_contract.ACTION_SCALE_RAD, places=12
        )
        self.assertEqual(pipeline.last_limited_fraction, 0.0)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            action_pipeline.limit_joint_target_slew(
                [0.0] * ACTION_DIM,
                [0.0] * ACTION_DIM,
                True,
                max_delta_rad_per_20ms=-0.1,
                step_dt=STEP_DT,
            )
        with self.assertRaises(ValueError):
            action_pipeline.limit_joint_target_slew(
                [0.0] * ACTION_DIM,
                [0.0] * ACTION_DIM,
                True,
                max_delta_rad_per_20ms=SLEW,
                step_dt=0.0,
            )
        with self.assertRaises(ValueError):
            action_pipeline.limit_joint_target_slew(
                [0.0] * ACTION_DIM,
                [0.0] * (ACTION_DIM - 1),
                True,
                max_delta_rad_per_20ms=SLEW,
                step_dt=STEP_DT,
            )
        with self.assertRaises(ValueError):
            action_pipeline.apply_stand_action_scale(
                [0.0] * ACTION_DIM, [0.0, 0.0, 0.0], stand_action_scale=1.5
            )


class ObservationBuilderTests(unittest.TestCase):
    """The 66-vector assembly, its length checks, and its NaN refusal."""

    def _inputs(self) -> dict[str, list[float]]:
        return {
            "root_linear_velocity": [0.21, 0.01, -0.02],
            "root_angular_velocity": [0.0, 0.0, 0.03],
            "projected_gravity": [0.0, 0.0, -1.0],
            "command": [0.24, 0.0, 0.0],
            "joint_position_error": [0.01 * index for index in range(ACTION_DIM)],
            "joint_velocity": [0.5] * ACTION_DIM,
            "action": [0.1] * ACTION_DIM,
        }

    def test_builds_in_contract_order(self) -> None:
        vector = observation_builder.build_observation(**self._inputs())
        self.assertEqual(len(vector), observation_contract.OBSERVATION_DIM)
        inputs = self._inputs()
        for name, (start, stop) in observation_contract.field_layout().items():
            self.assertEqual(vector[start:stop], inputs[name], name)
        self.assertEqual(
            vector[observation_contract.COMMAND], [0.24, 0.0, 0.0]
        )
        self.assertEqual(vector[observation_contract.ACTION], [0.1] * ACTION_DIM)

    def test_round_trip_through_split(self) -> None:
        inputs = self._inputs()
        split = observation_builder.split_observation(
            observation_builder.build_observation(**inputs)
        )
        self.assertEqual(split, inputs)

    def test_rejects_wrong_widths(self) -> None:
        for name in ("command", "joint_velocity", "projected_gravity"):
            inputs = self._inputs()
            inputs[name] = inputs[name][:-1]
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    observation_builder.build_observation(**inputs)

    def test_rejects_non_finite_values(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            inputs = self._inputs()
            inputs["joint_velocity"] = [bad] + [0.0] * (ACTION_DIM - 1)
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    observation_builder.build_observation(**inputs)
        with self.assertRaises(ValueError):
            observation_builder.validate_observation(
                [float("nan")] * observation_contract.OBSERVATION_DIM
            )

    def test_rejects_a_wrong_length_vector(self) -> None:
        with self.assertRaises(ValueError):
            observation_builder.validate_observation([0.0] * 68)


if __name__ == "__main__":
    unittest.main()
