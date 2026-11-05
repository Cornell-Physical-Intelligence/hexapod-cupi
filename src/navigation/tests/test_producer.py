"""Pure-math tests for the navigation command seam and its bank-conforming follower.

``WaypointFollower`` is an example, not a tuned planner, so what is tested here
is not path quality. It is the properties any producer has to have before it
may drive the trained policy: every command it emits is a training-bank entry
or stand, it holds the body heading at a bank bearing, it converges on an
eight-connected grid route, and it respects a consumer's envelope. A command
outside the bank asks a network for behavior it has never seen.

Everything is closed-form or a fixed-step kinematic rollout. No torch, no
simulator, no randomness.
"""

from __future__ import annotations

import math
import unittest

from contracts.command import (
    CommandEnvelope,
    Frame,
    VelocityCommand,
)
from navigation.producer import CommandProducer, Pose2D, wrap_to_pi
from navigation.waypoint import (
    BANK_ENVELOPE,
    BANK_SPEED_LEVELS_MPS,
    BANK_YAW_RATE_RAD_S,
    WaypointFollower,
    command_bank,
)

POLICY_STEP_DT_S = 0.02
CELL_M = 0.25
BANK = {tuple(entry) for entry in command_bank()}

# Eight-connected cell centers 0.25 m apart that visit all eight bearings:
# E, NE, N, NW, W, SW, S, SE, back to the origin.
STAIRCASE = [
    (CELL_M, 0.0),
    (2 * CELL_M, CELL_M),
    (2 * CELL_M, 2 * CELL_M),
    (CELL_M, 3 * CELL_M),
    (0.0, 3 * CELL_M),
    (-CELL_M, 2 * CELL_M),
    (-CELL_M, CELL_M),
    (0.0, 0.0),
]


def _is_bank_or_stand(command: VelocityCommand) -> bool:
    return command.is_stand() or command.as_tuple() in BANK


def _integrate(pose: Pose2D, command: VelocityCommand, dt: float) -> Pose2D:
    """Advance a holonomic body that tracks its body-frame command exactly."""

    heading = pose.heading_rad
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return Pose2D(
        x_m=pose.x_m + (command.vx_mps * cos_h - command.vy_mps * sin_h) * dt,
        y_m=pose.y_m + (command.vx_mps * sin_h + command.vy_mps * cos_h) * dt,
        heading_rad=wrap_to_pi(heading + command.wz_rad_s * dt),
    )


def _rollout(
    follower: WaypointFollower,
    pose: Pose2D,
    *,
    steps: int = 6000,
    dt: float = POLICY_STEP_DT_S,
) -> tuple[Pose2D, list[VelocityCommand], list[Pose2D]]:
    """Integrate at the policy rate and collect every command and pose.

    The robot is assumed to track its command exactly, which is generous but is
    the right abstraction here: this exercises the producer, not the gait.
    """

    emitted: list[VelocityCommand] = []
    poses: list[Pose2D] = [pose]
    for step in range(steps):
        command = follower.update(pose, step * dt)
        emitted.append(command)
        if follower.is_finished:
            break
        pose = _integrate(pose, command, dt)
        poses.append(pose)
    return pose, emitted, poses


class WrapToPiTests(unittest.TestCase):
    def test_wrapping(self) -> None:
        self.assertAlmostEqual(wrap_to_pi(0.0), 0.0, places=12)
        self.assertAlmostEqual(wrap_to_pi(math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(-math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(3.0 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(1.5 * math.pi), -0.5 * math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(-1.5 * math.pi), 0.5 * math.pi, places=12)
        for angle in (0.3, -0.3, 2.0, -2.0, 12.0, -12.0):
            self.assertLessEqual(abs(wrap_to_pi(angle)), math.pi + 1.0e-12)
        with self.assertRaises(ValueError):
            wrap_to_pi(float("nan"))


class PoseTests(unittest.TestCase):
    def test_rejects_non_finite_components(self) -> None:
        Pose2D(0.0, 0.0, 0.0)
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Pose2D(bad, 0.0, 0.0)
            with self.assertRaises(ValueError):
                Pose2D(0.0, 0.0, bad)

    def test_follower_satisfies_the_producer_protocol(self) -> None:
        follower = WaypointFollower([(1.0, 0.0)])
        self.assertIsInstance(follower, CommandProducer)


class CommandBankTests(unittest.TestCase):
    def test_navigation_bank_equals_the_training_bank(self) -> None:
        # Navigation must not import locomotion (ARCHITECTURE.md); the test may.
        from locomotion.omniwalk.mdp.commands import TaskConfig
        from locomotion.omniwalk.mdp.commands import command_bank as training_bank

        self.assertEqual(command_bank(), training_bank(TaskConfig()))

    def test_default_envelope_is_derived_from_the_bank(self) -> None:
        top = max(BANK_SPEED_LEVELS_MPS)
        self.assertEqual(BANK_ENVELOPE.forward_mps, (-top, top))
        self.assertEqual(BANK_ENVELOPE.lateral_mps, (-top, top))
        self.assertEqual(BANK_ENVELOPE.yaw_rate_rad_s, (-BANK_YAW_RATE_RAD_S, BANK_YAW_RATE_RAD_S))
        for entry in command_bank():
            VelocityCommand(*entry).validate(BANK_ENVELOPE)


class WaypointFollowerTests(unittest.TestCase):
    def assert_route_completes(
        self, follower: WaypointFollower, start: Pose2D, route: list[tuple[float, float]]
    ) -> tuple[list[VelocityCommand], list[Pose2D]]:
        final_pose, emitted, poses = _rollout(follower, start)
        self.assertTrue(follower.is_finished, "follower did not retire every waypoint")
        self.assertTrue(emitted[-1].is_stand())
        self.assertGreater(len(emitted), 10)
        self.assertLessEqual(
            math.hypot(final_pose.x_m - route[-1][0], final_pose.y_m - route[-1][1]),
            follower.stop_radius_m + 1.0e-9,
        )
        for command in emitted:
            self.assertTrue(_is_bank_or_stand(command), f"{command} is not a bank entry")
            self.assertIs(command.frame, Frame.NAVIGATION)
            command.validate(BANK_ENVELOPE)
        return emitted, poses

    def test_staircase_route_emits_bank_commands_only(self) -> None:
        follower = WaypointFollower(STAIRCASE, stop_radius_m=0.12)
        emitted, poses = self.assert_route_completes(follower, Pose2D(0.0, 0.0, 0.0), STAIRCASE)
        # The body heading is already a bank bearing: no yaw, heading held.
        for command in emitted:
            self.assertEqual(command.wz_rad_s, 0.0)
        for pose in poses:
            self.assertAlmostEqual(pose.heading_rad, 0.0, places=12)
        # Every one of the eight bearings is used along the way.
        bearings = {
            round(math.atan2(c.vy_mps, c.vx_mps) / (math.pi / 4)) % 8
            for c in emitted
            if not c.is_stand()
        }
        self.assertEqual(bearings, set(range(8)))

    def test_staircase_route_from_a_diagonal_held_heading(self) -> None:
        follower = WaypointFollower(STAIRCASE, stop_radius_m=0.12)
        start = Pose2D(0.0, 0.0, math.pi / 4)
        emitted, poses = self.assert_route_completes(follower, start, STAIRCASE)
        for command in emitted:
            self.assertEqual(command.wz_rad_s, 0.0)
        for pose in poses:
            self.assertAlmostEqual(pose.heading_rad, math.pi / 4, places=12)

    def test_staircase_route_turns_to_the_held_heading_first(self) -> None:
        gate = 0.05
        follower = WaypointFollower(STAIRCASE, stop_radius_m=0.12, heading_gate_rad=gate)
        start = Pose2D(0.0, 0.0, 0.30)
        emitted, poses = self.assert_route_completes(follower, start, STAIRCASE)
        first_drive = next(i for i, c in enumerate(emitted) if c.vx_mps or c.vy_mps)
        self.assertGreater(first_drive, 0, "expected an in-place turn before driving")
        for command in emitted[:first_drive]:
            self.assertEqual(command.as_tuple(), (0.0, 0.0, -BANK_YAW_RATE_RAD_S))
        for command in emitted[first_drive:]:
            self.assertEqual(command.wz_rad_s, 0.0)
        for pose in poses[first_drive:]:
            self.assertLess(abs(pose.heading_rad), gate)
        # The correction converges without overshoot: the heading error shrinks
        # on every turning tick and the pure-yaw commands stop once inside the gate.
        errors = [abs(pose.heading_rad) for pose in poses[: first_drive + 1]]
        self.assertAlmostEqual(errors[0], 0.30, places=12)
        for earlier, later in zip(errors, errors[1:]):
            self.assertLess(later, earlier, "heading error must shrink while turning")

    def test_heading_hold_turns_toward_the_nearest_bank_bearing(self) -> None:
        goal = [(3.0, 0.0)]
        # 0.30 rad is nearest 0: turn negative. -0.30 mirrors. 0.50 rad is
        # nearest pi/4: turn positive. All with zero translation.
        for heading, expected in ((0.30, -BANK_YAW_RATE_RAD_S), (-0.30, BANK_YAW_RATE_RAD_S)):
            command = WaypointFollower(goal).update(Pose2D(0.0, 0.0, heading), 0.0)
            self.assertEqual(command.as_tuple(), (0.0, 0.0, expected))
        command = WaypointFollower(goal).update(Pose2D(0.0, 0.0, 0.50), 0.0)
        self.assertEqual(command.as_tuple(), (0.0, 0.0, BANK_YAW_RATE_RAD_S))
        self.assertAlmostEqual(WaypointFollower.held_heading(0.50), math.pi / 4, places=12)
        self.assertAlmostEqual(WaypointFollower.held_heading(3.0), math.pi, places=12)
        # Inside the gate the follower drives instead of turning.
        command = WaypointFollower(goal, heading_gate_rad=0.05).update(Pose2D(0.0, 0.0, 0.04), 0.0)
        self.assertEqual(command.wz_rad_s, 0.0)
        self.assertGreater(command.vx_mps, 0.0)

    def test_drives_sideways_instead_of_turning(self) -> None:
        # Goal 90 degrees to the left of a held heading: pure lateral drive.
        command = WaypointFollower([(0.0, 2.0)]).update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertTrue(_is_bank_or_stand(command))
        self.assertEqual(command.wz_rad_s, 0.0)
        self.assertAlmostEqual(command.vx_mps, 0.0, places=12)
        self.assertAlmostEqual(command.vy_mps, max(BANK_SPEED_LEVELS_MPS), places=12)
        mirrored = WaypointFollower([(0.0, -2.0)]).update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(mirrored.vy_mps, -max(BANK_SPEED_LEVELS_MPS), places=12)
        # The goal direction is rotated by the body heading: facing north, a
        # goal to the north is straight ahead.
        ahead = WaypointFollower([(0.0, 2.0)]).update(Pose2D(0.0, 0.0, math.pi / 2), 0.0)
        self.assertAlmostEqual(ahead.vx_mps, max(BANK_SPEED_LEVELS_MPS), places=12)
        self.assertAlmostEqual(ahead.vy_mps, 0.0, places=12)

    def test_bearing_rounds_to_the_nearest_45_degrees(self) -> None:
        top = max(BANK_SPEED_LEVELS_MPS)
        at_30_deg = WaypointFollower([(math.cos(math.radians(30)), math.sin(math.radians(30)))])
        command = at_30_deg.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertTrue(_is_bank_or_stand(command))
        self.assertAlmostEqual(command.vx_mps, top * math.cos(math.pi / 4), places=12)
        self.assertAlmostEqual(command.vy_mps, top * math.sin(math.pi / 4), places=12)
        at_20_deg = WaypointFollower([(math.cos(math.radians(20)), math.sin(math.radians(20)))])
        command = at_20_deg.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(command.vx_mps, top, places=12)
        self.assertAlmostEqual(command.vy_mps, 0.0, places=12)

    def test_speed_rounds_to_a_bank_level(self) -> None:
        low, high = BANK_SPEED_LEVELS_MPS
        follower = WaypointFollower([(3.0, 0.0)], approach_gain_per_s=0.60, stop_radius_m=0.02)
        far = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(far.vx_mps, high, places=12, msg="0.60 * 3.0 rounds to the top")
        near = follower.update(Pose2D(2.95, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(near.vx_mps, low, places=12, msg="0.60 * 0.05 rounds to the low")
        self.assertTrue(_is_bank_or_stand(near))

    def test_stops_inside_the_stop_radius(self) -> None:
        follower = WaypointFollower([(1.0, 0.0)], stop_radius_m=0.12)
        command = follower.update(Pose2D(0.95, 0.02, 0.0), 0.0)
        self.assertTrue(command.is_stand())
        self.assertTrue(follower.is_finished)
        self.assertIsNone(follower.current_waypoint)
        # A finished follower keeps standing rather than drifting.
        self.assertTrue(follower.update(Pose2D(0.95, 0.02, 0.0), 1.0).is_stand())

    def test_no_waypoints_is_an_immediate_stand(self) -> None:
        follower = WaypointFollower([])
        self.assertTrue(follower.is_finished)
        self.assertTrue(follower.update(Pose2D(0.0, 0.0, 0.0), 0.0).is_stand())

    def test_retires_several_satisfied_waypoints_in_one_tick(self) -> None:
        follower = WaypointFollower([(0.02, 0.0), (0.0, 0.03), (1.0, 0.0)], stop_radius_m=0.12)
        command = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertEqual(follower.current_waypoint, (1.0, 0.0))
        self.assertGreater(command.vx_mps, 0.0)

    def test_off_grid_goals_still_complete_with_bank_commands(self) -> None:
        route = [(1.0, -0.7), (-0.5, -1.0), (0.2, 0.4)]
        follower = WaypointFollower(route, stop_radius_m=0.12)
        self.assert_route_completes(follower, Pose2D(0.0, 0.0, math.pi), route)

    def test_consumer_envelope_is_applied(self) -> None:
        narrow = CommandEnvelope(
            forward_mps=(0.0, 0.03),
            lateral_mps=(0.0, 0.0),
            yaw_rate_rad_s=(-0.1, 0.1),
            name="narrow",
        )
        follower = WaypointFollower([(2.0, 0.0)], envelope=narrow)
        command = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        command.validate(narrow)
        self.assertAlmostEqual(command.vx_mps, 0.03, places=12)
        turning = follower.update(Pose2D(0.0, 0.0, 0.30), 0.0)
        turning.validate(narrow)
        self.assertAlmostEqual(turning.wz_rad_s, -0.1, places=12)

    def test_rejects_bad_construction(self) -> None:
        with self.assertRaises(ValueError):
            WaypointFollower([(0.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            WaypointFollower([(float("nan"), 0.0)])
        with self.assertRaises(ValueError):
            WaypointFollower([(1.0, 0.0)], approach_gain_per_s=0.0)
        with self.assertRaises(ValueError):
            WaypointFollower([(1.0, 0.0)], stop_radius_m=-0.1)
        with self.assertRaises(ValueError):
            WaypointFollower([(1.0, 0.0)], heading_gate_rad=math.pi / 8)
        with self.assertRaises(TypeError):
            WaypointFollower([(1.0, 0.0)]).update((0.0, 0.0, 0.0), 0.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
